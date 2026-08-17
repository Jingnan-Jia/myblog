#!/usr/bin/env node
/**
 * agent-log-uploader.mjs — Agent 日志增量上传器（零依赖，Node 18+）
 *
 * 跑在运行 agent 的服务器上：持续 tail 指定的 log 文件，
 * 把新增的行批量写入 Supabase（agent_logs 表），并心跳更新 agents 表。
 *
 * 用法：
 *   SUPABASE_URL=https://xxx.supabase.co \
 *   SUPABASE_ANON_KEY=eyJ... \
 *   node scripts/agent-log-uploader.mjs --log /path/to/agent.log --agent my-agent [--run run-001]
 *
 * 参数：
 *   --log <path>      agent 日志文件路径（必填）
 *   --agent <name>    agent 标识（必填，页面上显示的名字）
 *   --run <id>        本次运行标识（可选，默认取启动时间戳）
 *   --interval <ms>   轮询间隔（默认 3000）
 *   --flush-lines <n> 攒够多少行批量上传（默认 20）
 *   --no-heartbeat    关闭空闲心跳（默认每 30s 心跳一次）
 *
 * 日志行格式约定（可选，便于页面分类展示）：
 *   普通行            → level=info, content=整行
 *   [TOOL] name args → level=tool, tool_name=name, args=JSON
 *   [ERROR] ...       → level=error
 * 不符合约定就按普通文本行处理。
 */

import fs from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const SUPABASE_URL = process.env.SUPABASE_URL || "";
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY || "";

function parseArgs(argv) {
  const args = { log: null, agent: null, run: null, interval: 3000, flushLines: 20, heartbeat: true };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--log") args.log = next();
    else if (a === "--agent") args.agent = next();
    else if (a === "--run") args.run = next();
    else if (a === "--interval") args.interval = Number(next()) || 3000;
    else if (a === "--flush-lines") args.flushLines = Number(next()) || 20;
    else if (a === "--no-heartbeat") args.heartbeat = false;
  }
  return args;
}

// ── 简单行解析：识别 [TOOL] / [ERROR] / [WARN] 前缀 ──
function parseLine(line) {
  const trimmed = line.trimEnd();
  const toolMatch = trimmed.match(/^\[TOOL\]\s+(\S+)\s+(.*)$/);
  if (toolMatch) {
    const row = { level: "tool", tool_name: toolMatch[1], content: trimmed };
    try {
      row.args = JSON.parse(toolMatch[2]);
    } catch {
      row.args = toolMatch[2] || null;
    }
    return row;
  }
  if (/^\[ERROR\]/.test(trimmed)) return { level: "error", content: trimmed };
  if (/^\[WARN\]/.test(trimmed)) return { level: "warn", content: trimmed };
  return { level: "info", content: trimmed };
}

async function insertRows(rows) {
  if (rows.length === 0) return;
  const res = await fetch(`${SUPABASE_URL}/rest/v1/agent_logs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: SUPABASE_ANON_KEY,
      Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
      Prefer: "return=minimal",
    },
    body: JSON.stringify(rows),
  });
  if (!res.ok) {
    throw new Error(`insert agent_logs 失败: ${res.status} ${await res.text()}`);
  }
}

async function upsertAgent(agentId, runId, lastLine, status = "active") {
  const res = await fetch(`${SUPABASE_URL}/rest/v1/agents?on_conflict=agent_id`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: SUPABASE_ANON_KEY,
      Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
      Prefer: "resolution=merge-duplicates,return=minimal",
    },
    body: JSON.stringify({
      agent_id: agentId,
      run_id: runId,
      status,
      last_ts: new Date().toISOString(),
      last_line: (lastLine || "").slice(0, 500),
    }),
  });
  if (!res.ok) {
    throw new Error(`upsert agents 失败: ${res.status} ${await res.text()}`);
  }
}

async function main() {
  const { log, agent, run, interval, flushLines, heartbeat } = parseArgs(process.argv);

  if (!log || !agent) {
    console.error("用法: node scripts/agent-log-uploader.mjs --log <path> --agent <name> [--run <id>]");
    console.error("且需设置环境变量 SUPABASE_URL 与 SUPABASE_ANON_KEY");
    process.exit(1);
  }
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
    console.error("错误: 缺少 SUPABASE_URL / SUPABASE_ANON_KEY 环境变量");
    process.exit(1);
  }
  if (!fs.existsSync(log)) {
    console.error(`错误: 日志文件不存在: ${log}`);
    process.exit(1);
  }

  const runId = run || `run-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`;
  console.log(`[uploader] agent=${agent} run=${runId} log=${log} interval=${interval}ms`);

  const fd = fs.openSync(log, "r");
  let position = fs.fstatSync(fd).size; // 从文件末尾开始，只上传之后的增量
  let buffer = [];

  async function flush() {
    if (buffer.length === 0) return;
    const rows = buffer;
    buffer = [];
    try {
      await insertRows(rows);
      const last = rows[rows.length - 1];
      await upsertAgent(agent, runId, last.content);
      console.log(`[uploader] +${rows.length} 行 (最后: ${last.content.slice(0, 60)})`);
    } catch (e) {
      console.error(`[uploader] 上传失败: ${e.message}`);
      buffer = rows; // 失败保留待下次重试
    }
  }

  let lastHeartbeat = 0;
  let shuttingDown = false;

  // 兜底：上传器退出时把 agent 置为 idle（等待请求完成再退出）
  const shutdown = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    try {
      const lastLine = buffer.length ? buffer[buffer.length - 1].content : "";
      await upsertAgent(agent, runId, lastLine, "idle");
    } catch {
      // 忽略
    }
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  console.log("[uploader] 开始监听日志增量 (Ctrl+C 退出)...");
  let lastFlush = Date.now();
  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      const size = fs.fstatSync(fd).size;
      if (size > position) {
        const buf = Buffer.alloc(size - position);
        fs.readSync(fd, buf, 0, buf.length, position);
        position = size;

        const text = buf.toString("utf8");
        const lines = text.split("\n");
        if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop(); // 去掉末尾空段
        for (const line of lines) {
          if (!line.trim()) continue;
          const row = parseLine(line);
          buffer.push({
            agent_id: agent,
            run_id: runId,
            ts: new Date().toISOString(),
            level: row.level,
            content: row.content,
            tool_name: row.tool_name ?? null, // 统一字段，PostgREST 批量 insert 要求各字段一致
            args: row.args ?? null,
          });
        }
      }

      // 批量上传：攒够 flushLines 行，或距上次上传超过 5s 且有数据（兜底，防止低吞吐卡缓冲）
      if (buffer.length && (buffer.length >= flushLines || Date.now() - lastFlush >= 5000)) {
        await flush();
        lastFlush = Date.now();
      }

      // 空闲心跳：无新行也定期刷新 last_ts（仪表盘据此判断运行中）
      if (heartbeat && Date.now() - lastHeartbeat > 30000) {
        lastHeartbeat = Date.now();
        try {
          await upsertAgent(agent, runId, buffer.length ? buffer[buffer.length - 1].content : undefined);
        } catch (e) {
          console.error(`[uploader] 心跳失败: ${e.message}`);
        }
      }
    } catch (e) {
      console.error(`[uploader] 读取异常: ${e.message}`);
    }
    await sleep(interval);
  }
}

main().catch((e) => {
  console.error(`[uploader] 致命错误: ${e.message}`);
  process.exit(1);
});
