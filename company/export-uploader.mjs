#!/usr/bin/env node
/* eslint-disable no-console -- 命令行工具，console 即输出通道 */
/**
 * export-uploader.mjs — 成果导出 CLI（公司侧，零依赖，Node 18+）
 *
 * 把一个目录整批经博客中转传到个人 GitHub 私有仓库（整批一个 commit）。
 * 协议与网页版一致：POST text/plain，首行 JSON 元数据 + '\n' + 正文。
 *
 * 【小分块协议】公司网关对请求体实测上限约 10KB（10240B），
 *   因此每块 body（meta 行 + 正文）必须 ≤ ~9.5KB：
 *   - 文本（合法 UTF-8）：按 3000 字符/块（最坏 3 字节/字符 → 9KB）
 *   - 二进制：按 6750 原始字节/块（base64 后 9000 字符）
 *
 * 用法：
 *   EXPORT_RELAY_SECRET=口令 node export-uploader.mjs \
 *     --dir /path/to/project --owner <github用户名> --repo <私有仓库名> \
 *     [--branch main] [--subdir ''] [--label 'exp-A'] [--concurrency 3] \
 *     [--endpoint https://www.jiajingnan.cn] [--batch <id>] [--dry-run]
 *
 * 断点续传：中断后【用同一个 --batch 重跑】即可，已收分块自动跳过。
 *   （不传 --batch 时按 目录名+时间戳 生成新批次，重跑会生成不同批次！）
 *
 * 速度预期：单块有效载荷 ~6.6KB（二进制），并发 3 时约 100-200KB/s。
 *   1MB ≈ 10-20 秒；10MB ≈ 2-3 分钟。超 10MB 建议拆批或压缩。
 *
 * 安全提示：
 *   - 口令优先用环境变量 EXPORT_RELAY_SECRET（--secret 会留在 shell 历史里）
 *   - 发送前请自查：不要包含密钥/密码/内部域名/专有标识（git 历史不可擦除）
 *   - 目标仓库请保持 private
 */

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

// 公司网关请求体上限约 10KB，以下参数保证每块 body（meta 行 ~400B + 正文）≤ ~9.5KB
const TEXT_CHUNK_CHARS = 3000; // 文本块字符数（最坏 3 字节/字符）
const BIN_CHUNK_RAW = 6750; // 二进制块原始字节（3 的倍数 → base64 恰好 9000 字符，无填充）
const MAX_FILES = 60; // 单批建议上限（服务端硬上限 200）
const MAX_TOTAL_BYTES = 10 * 1024 * 1024; // 单批建议上限 10MB（网关 10KB/请求，太大太慢；服务端硬上限 200MB）
const SKIP_DIRS = new Set([".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode", "dist", "build", ".next"]);

function parseArgs(argv) {
  const args = {
    dir: null, owner: null, repo: null,
    branch: "main", subdir: "", label: "",
    endpoint: "https://www.jiajingnan.cn",
    secret: process.env.EXPORT_RELAY_SECRET || "",
    batch: null, dryRun: false, concurrency: 3,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) { console.error(`参数 ${a} 缺少取值`); process.exit(1); }
      return argv[++i];
    };
    if (a === "--dir") args.dir = next();
    else if (a === "--owner") args.owner = next();
    else if (a === "--repo") args.repo = next();
    else if (a === "--branch") args.branch = next();
    else if (a === "--subdir") args.subdir = next();
    else if (a === "--label") args.label = next();
    else if (a === "--endpoint") args.endpoint = next().replace(/\/+$/, "");
    else if (a === "--secret") args.secret = next();
    else if (a === "--batch") args.batch = next();
    else if (a === "--concurrency") args.concurrency = Math.min(6, Math.max(1, Number(next()) || 3));
    else if (a === "--dry-run") args.dryRun = true;
    else if (a === "--help" || a === "-h") { usage(); process.exit(0); }
    else { console.error(`未知参数: ${a}`); usage(); process.exit(1); }
  }
  return args;
}

function usage() {
  console.log(`用法:
  EXPORT_RELAY_SECRET=口令 node export-uploader.mjs --dir <目录> --owner <用户名> --repo <仓库名> [选项]

选项:
  --branch <name>     目标分支（默认 main，需已存在；空仓库首次提交会自动建该分支）
  --subdir <path>     仓库内子目录（默认仓库根）
  --label <text>      批次标签，写入 commit message
  --concurrency <n>   并发上传请求数（默认 3，最大 6）
  --endpoint <url>    中转站点（默认 https://www.jiajingnan.cn）
  --batch <id>        指定批次 id（断点续传必须与上次相同）
  --dry-run           只列出将要上传的文件，不发起网络请求
  --secret <口令>     口令（不推荐，建议用环境变量 EXPORT_RELAY_SECRET）

说明:
  公司网关请求体上限约 10KB，所有内容自动分块（文本 3000 字符/块、二进制 6.6KB/块）。`);
}

// ── 递归收集文件（跳过隐藏项与常见非源码目录） ──
function collectFiles(rootDir) {
  const out = [];
  const walk = (dir) => {
    for (const name of fs.readdirSync(dir)) {
      if (name.startsWith(".")) continue;
      const full = path.join(dir, name);
      const stat = fs.statSync(full);
      if (stat.isDirectory()) {
        if (SKIP_DIRS.has(name)) continue;
        walk(full);
      } else if (stat.isFile()) {
        out.push({ abs: full, rel: path.relative(rootDir, full).split(path.sep).join("/") });
      }
    }
  };
  walk(rootDir);
  out.sort((a, b) => a.rel.localeCompare(b.rel));
  return out;
}

// ── 文件分类：合法 UTF-8 → 文本分块（按字符）；否则 base64 分块（按字节） ──
function classify(buf) {
  let text = null;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(buf);
  } catch {
    text = null;
  }
  const isText = text !== null;
  const chunks = isText
    ? Math.max(1, Math.ceil(text.length / TEXT_CHUNK_CHARS))
    : Math.max(1, Math.ceil(buf.length / BIN_CHUNK_RAW));
  return { isText, text: isText ? text : null, chunks };
}

async function relayPost(endpoint, fn, meta, payload, secret) {
  const res = await fetch(`${endpoint}/api/export/${fn}`, {
    method: "POST",
    headers: { "Content-Type": "text/plain", "X-Relay-Secret": secret },
    body: `${JSON.stringify(meta)}\n${payload ?? ""}`,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${data.message || data.error || "未知错误"}`);
  return data;
}

async function relayStatus(endpoint, batch, secret) {
  const res = await fetch(`${endpoint}/api/export/upload?batch=${encodeURIComponent(batch)}`, {
    headers: { "X-Relay-Secret": secret },
  });
  if (res.status === 404) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${data.message || data.error || "未知错误"}`);
  return data;
}

async function uploadChunkWithRetry(endpoint, meta, payload, secret, desc) {
  let attempt = 0;
  for (;;) {
    attempt += 1;
    try {
      await relayPost(endpoint, "upload", meta, payload, secret);
      return;
    } catch (err) {
      if (attempt >= 3) throw new Error(`${desc} 连续 ${attempt} 次失败: ${err.message}`);
      console.error(`  [重试 ${attempt}/3] ${desc}: ${err.message}`);
      await new Promise((r) => setTimeout(r, 1500 * attempt));
    }
  }
}

// ── 单文件并发上传：worker 池消费未收块索引，任一失败即中止该文件 ──
async function uploadFile(plan, args, batch, received) {
  const got = received.get(plan.rel) || new Set();
  const idxs = [];
  for (let i = 0; i < plan.chunks; i++) if (!got.has(i)) idxs.push(i);
  if (!idxs.length) return;

  const baseMeta = {
    v: 1, action: "upload", batch,
    repo_owner: args.owner, repo_name: args.repo,
    branch: args.branch, subdir: args.subdir, label: args.label,
    file_path: plan.rel, is_binary: !plan.isText,
    total: plan.chunks, bytes: plan.bytes, file_sha256: plan.sha256,
  };
  const payloadOf = (idx) =>
    plan.isText
      ? plan.text.slice(idx * TEXT_CHUNK_CHARS, (idx + 1) * TEXT_CHUNK_CHARS)
      : plan.buf.subarray(idx * BIN_CHUNK_RAW, Math.min((idx + 1) * BIN_CHUNK_RAW, plan.buf.length)).toString("base64");

  let cursor = 0;
  let done = 0;
  let failure = null;
  const worker = async () => {
    while (!failure) {
      const idx = cursor++;
      if (idx >= idxs.length) return;
      try {
        await uploadChunkWithRetry(args.endpoint, { ...baseMeta, idx }, payloadOf(idx), args.secret, `${plan.rel} 第 ${idx + 1}/${plan.chunks} 块`);
      } catch (err) {
        failure = err;
        return;
      }
      done += 1;
      if (idxs.length > 1) process.stdout.write(`\r  ${plan.rel}: ${done}/${idxs.length} 块`);
    }
  };
  await Promise.all(Array.from({ length: Math.min(args.concurrency, idxs.length) }, worker));
  if (idxs.length > 1) process.stdout.write("\n");
  if (failure) throw failure;
}

async function main() {
  const args = parseArgs(process.argv);
  if (!args.dir || !args.owner || !args.repo) { usage(); process.exit(1); }
  if (!fs.existsSync(args.dir) || !fs.statSync(args.dir).isDirectory()) {
    console.error(`错误: 目录不存在或不是目录: ${args.dir}`);
    process.exit(1);
  }

  const files = collectFiles(args.dir);
  if (!files.length) { console.error("错误: 目录中没有可上传的文件"); process.exit(1); }

  const plan = files.map((f) => {
    const buf = fs.readFileSync(f.abs);
    const { isText, text, chunks } = classify(buf);
    return {
      rel: f.rel, buf, isText, text, chunks,
      sha256: crypto.createHash("sha256").update(buf).digest("hex"),
      bytes: buf.length,
    };
  });

  const totalBytes = plan.reduce((s, f) => s + f.bytes, 0);
  const totalChunks = plan.reduce((s, f) => s + f.chunks, 0);
  console.log(`批次计划：${plan.length} 个文件，共 ${(totalBytes / 1048576).toFixed(2)} MB，${totalChunks} 块（每块 ≤10KB，适配网关）`);
  for (const f of plan) {
    console.log(`  ${f.isText ? "[文本]" : "[b64 ]"} ${String(f.bytes).padStart(9)} B  ${f.chunks} 块  ${f.rel}`);
  }
  if (plan.length > MAX_FILES) {
    console.warn(`警告: 文件数 ${plan.length} 超出单批建议上限 ${MAX_FILES}，请拆批`);
  }
  if (totalBytes > MAX_TOTAL_BYTES) {
    console.warn(`警告: 总量 ${(totalBytes / 1048576).toFixed(1)}MB 超出建议上限 10MB（网关 10KB/请求，传输会很慢），建议压缩或拆批`);
  }

  if (args.dryRun) {
    console.log("\n[dry-run] 到此为止，未发起任何网络请求。");
    return;
  }

  if (!args.secret) {
    console.error("错误: 缺少口令。请设置环境变量 EXPORT_RELAY_SECRET（或用 --secret，不推荐）");
    process.exit(1);
  }

  const batch = args.batch || `exp-${path.basename(path.resolve(args.dir))}-${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 12)}`;
  if (!args.batch) {
    console.log(`\n批次 id：${batch}`);
    console.log(`（如传输中断，重跑时请加 --batch ${batch} 以断点续传）`);
  }

  // ── 断点查询：跳过服务端已收分块（分块参数一致才认） ──
  const received = new Map(); // rel -> Set(idx)
  try {
    const st = await relayStatus(args.endpoint, batch, args.secret);
    if (st && st.files) {
      for (const f of st.files) {
        const hit = plan.find((p) => p.rel === f.file_path && f.total === p.chunks && Boolean(f.is_binary) !== p.isText);
        if (hit) received.set(f.file_path, new Set(f.received || []));
      }
      const n = Array.from(received.values()).reduce((s, x) => s + x.size, 0);
      if (n > 0) console.log(`断点续传：服务端已有 ${n} 块，将跳过`);
    }
  } catch (err) {
    console.log(`（未查询到历史批次，全新上传：${err.message}）`);
  }

  // ── 上传 ──
  const started = Date.now();
  let doneFiles = 0;
  for (const f of plan) {
    await uploadFile(f, args, batch, received);
    doneFiles += 1;
    const elapsed = Math.max(1, Math.round((Date.now() - started) / 1000));
    console.log(`[ ${doneFiles}/${plan.length} ] 已上传 ${f.rel}（累计 ${elapsed}s）`);
  }

  // ── finalize：拼装校验 → GitHub 单 commit → 清理暂存 ──
  console.log("\n全部上传完成，开始落盘（GitHub commit）…");
  const fin = await relayPost(args.endpoint, "finalize", { v: 1, action: "finalize", batch }, "", args.secret);
  console.log("\n✓ 导出成功");
  console.log(`  仓库:   ${fin.repo} @ ${fin.branch}`);
  console.log(`  Commit: ${fin.commit_sha}`);
  console.log(`  链接:   ${fin.html_url}`);
  if (Array.isArray(fin.files)) console.log(`  文件:   ${fin.files.length} 个`);
  console.log("\n中转暂存已清理。本次传输结束。");
}

main().catch((e) => {
  console.error(`\n✗ 失败: ${e.message}`);
  console.error("提示: 若是中途网络失败，用同一 --batch 重跑即可断点续传；若是 GitHub 侧失败，分块已保留，直接重跑（--batch 相同）。");
  process.exit(1);
});
