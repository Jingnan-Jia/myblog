#!/usr/bin/env node
/**
 * sync-ai-topics.mjs
 *
 * 将「AI副业选题库」的最新每日选题 / 每周计划部署到博客网站：
 *   - /Users/olivia/WorkBuddy/2026-08-07-23-15-06/选题库/daily/ 最新 md -> public/ai-topics/daily.html（固定文件名，直接替换）
 *   - /Users/olivia/WorkBuddy/2026-08-07-23-15-06/选题库/weekly/ 最新 md -> public/ai-topics/weekly.html（固定文件名，直接替换）
 *   - 生成 public/ai-topics/_index.json 供「选题总览」入口页（密码门）使用
 * 内容有变化则自动 git commit + push（Vercel 自动重新构建部署）。
 *
 * 与 sync-weekly.mjs 同款"单页替换"模式：每个类型只保留最新一份，不积累历史。
 * 用法：node scripts/sync-ai-topics.mjs
 */
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
  rmSync,
  statSync,
} from "node:fs";
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");
const topicsRoot = "/Users/olivia/WorkBuddy/2026-08-07-23-15-06/选题库";
const targetDir = join(projectRoot, "public", "ai-topics");
const indexPath = join(targetDir, "_index.json");

function run(cmd) {
  return execSync(cmd, { cwd: projectRoot, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
}
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function inline(s) {
  let out = esc(s);
  out = out.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  return out;
}
function mdToHtml(md) {
  const lines = md.split("\n");
  let html = "";
  let inList = false;
  const flushList = () => {
    if (inList) { html += "</ul>"; inList = false; }
  };
  for (const line of lines) {
    const t = line.trim();
    if (!t) { flushList(); continue; }
    if (t.startsWith("### ")) { flushList(); html += `<h3>${inline(t.slice(4))}</h3>`; }
    else if (t.startsWith("## ")) { flushList(); html += `<h2>${inline(t.slice(3))}</h2>`; }
    else if (t.startsWith("# ")) { flushList(); html += `<h1>${inline(t.slice(2))}</h1>`; }
    else if (t.startsWith("> ")) {
      flushList();
      html += `<blockquote>${inline(t.slice(2))}</blockquote>`;
    } else if (t.startsWith("- ") || /^\d+\.\s/.test(t)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(t.replace(/^- /, "").replace(/^\d+\.\s/, ""))}</li>`;
    } else { flushList(); html += `<p>${inline(t)}</p>`; }
  }
  flushList();
  return html;
}
function wrapPage(meta, bodyHtml) {
  const title = meta.title || "AI副业选题";
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${esc(title)}</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; background:#f5f6f8; color:#2c2c2a; line-height:1.75; }
  .wrap { max-width:760px; margin:0 auto; padding:32px 20px 80px; }
  .head { background:#fff; border:1px solid #e5e6e8; border-radius:16px; padding:20px 24px; margin-bottom:20px; }
  .head h1 { font-size:20px; font-weight:600; }
  .head .meta { color:#888780; font-size:13px; margin-top:6px; }
  .badge { display:inline-block; background:#E6F1FB; color:#185fa5; border-radius:999px; padding:2px 12px; font-size:12px; margin-right:6px; }
  .card { background:#fff; border:1px solid #e5e6e8; border-radius:16px; padding:24px; margin-bottom:16px; }
  .card h1 { font-size:18px; font-weight:600; margin:18px 0 8px; padding-bottom:8px; border-bottom:2px solid #185fa5; }
  .card h1:first-child { margin-top:0; }
  .card h2 { font-size:16px; font-weight:600; margin:22px 0 8px; padding-left:10px; border-left:4px solid #185fa5; }
  .card h3 { font-size:15px; font-weight:600; margin:18px 0 6px; color:#0c447c; }
  .card p { margin:8px 0; }
  .card ul { margin:8px 0 8px 4px; padding-left:20px; }
  .card li { margin:5px 0; }
  .card li strong { color:#185fa5; }
  .card blockquote { background:#f8f9fb; border-left:3px solid #d3d1c7; padding:8px 14px; margin:10px 0; color:#5f5e5a; font-size:13px; border-radius:0 8px 8px 0; }
  .card code { background:#F1EFE8; border-radius:4px; padding:1px 6px; font-size:13px; font-family:ui-monospace,Menlo,monospace; }
  .foot { text-align:center; color:#b4b2a9; font-size:12px; margin-top:24px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <h1>AI副业选题库</h1>
    <div class="meta"><span class="badge">每日选题</span><span class="badge">每周计划</span> 人设「懂AI的邻家姐姐」 · 仅供合作双方查看</div>
  </div>
  <div class="card">
    ${bodyHtml}
  </div>
  <div class="foot">由 WorkBuddy 自动化生成并部署 · ${meta.generatedAt || ""}</div>
</div>
</body>
</html>`;
}

function fileHashStr(str) {
  return createHash("sha256").update(str).digest("hex").slice(0, 12);
}

/** 同步最新每日选题/每周计划。@returns {{changed:boolean, reports:object[], error:string|null}} */
export function syncAiTopics() {
  if (!existsSync(topicsRoot)) {
    console.warn(`[sync-ai-topics] 跳过：选题库目录不存在 ${topicsRoot}`);
    return { changed: false, reports: [], error: `选题库目录不存在: ${topicsRoot}` };
  }
  mkdirSync(targetDir, { recursive: true });
  const reports = [];
  let changed = false;

  const SOURCES = [
    { key: "daily", label: "每日选题", dir: "daily", targetFile: "daily.html" },
    { key: "weekly", label: "每周计划", dir: "weekly", targetFile: "weekly.html" },
  ];

  for (const src of SOURCES) {
    const srcDir = join(topicsRoot, src.dir);
    if (!existsSync(srcDir)) {
      console.warn(`[sync-ai-topics] 跳过：${src.label}源目录不存在 ${srcDir}`);
      continue;
    }
    const files = readdirSync(srcDir).filter((f) => f.endsWith(".md")).sort().reverse();
    if (files.length === 0) {
      console.warn(`[sync-ai-topics] 跳过：${src.label}源目录中没有 md 文件`);
      continue;
    }
    const newest = files[0];
    const sourcePath = join(srcDir, newest);
    const targetPath = join(targetDir, src.targetFile);
    const md = readFileSync(sourcePath, "utf8");
    const stat = statSync(sourcePath);

    // 元信息：取 md 第一行 # 标题
    const titleMatch = md.match(/^#\s+(.+)$/m);
    const title = titleMatch ? titleMatch[1].trim() : newest.replace(/\.md$/, "");
    const generatedAt = new Date().toLocaleString("zh-CN", { hour12: false });

    reports.push({
      key: src.key,
      label: src.label,
      file: src.targetFile,
      url: `/ai-topics/${src.targetFile}`,
      title,
      period: newest.replace(/\.md$/, ""),
      generatedAt,
      size: stat.size,
      updatedAt: stat.mtimeMs,
      source: newest,
    });

    const html = wrapPage({ title, generatedAt }, mdToHtml(md));
    const fileChanged =
      !existsSync(targetPath) ||
      fileHashStr(readFileSync(targetPath, "utf8")) !== fileHashStr(html);
    writeFileSync(targetPath, html, "utf8");
    changed = changed || fileChanged;
    console.log(
      fileChanged
        ? `[sync-ai-topics] ${src.label}已替换: ${newest} -> ${src.targetFile}`
        : `[sync-ai-topics] ${src.label}内容无变化（${newest}）`
    );
  }

  const keep = new Set(["daily.html", "weekly.html", "_index.json"]);
  for (const f of readdirSync(targetDir)) {
    if (!keep.has(f)) {
      rmSync(join(targetDir, f), { force: true });
      console.log(`[sync-ai-topics] 清理旧文件: ${f}`);
    }
  }
  writeFileSync(indexPath, JSON.stringify(reports, null, 2), "utf8");

  if (reports.length === 0) {
    return { changed: false, reports, error: "选题库源目录均无 md 文件" };
  }
  return { changed, reports, error: null };
}

// CLI 直接运行时执行：同步 + 提交推送
const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  console.log("========== AI副业选题部署 ==========");
  const { changed, reports, error } = syncAiTopics();
  if (error) {
    console.error(`[sync-ai-topics] 同步失败：${error}`);
    process.exit(1);
  }
  if (!changed) {
    console.log("[sync-ai-topics] 内容均无变化，跳过提交推送");
    process.exit(0);
  }
  const periodInfo = reports.map((r) => `${r.label}:${r.period}`).join(" ");
  try {
    run("git add public/ai-topics/");
    const commitMsg = `chore: 更新AI副业选题（${periodInfo}）`;
    run(`git commit -m "${commitMsg}"`);
    console.log(`[sync-ai-topics] 已提交：${commitMsg}`);
    try {
      run("git -c http.version=HTTP/1.1 push origin main");
    } catch {
      console.warn("[sync-ai-topics] HTTP/1.1 推送失败，尝试默认协议重试...");
      run("git push origin main");
    }
    console.log("[sync-ai-topics] 已推送到 GitHub，Vercel 将自动重新部署");
  } catch (e) {
    console.error(`[sync-ai-topics] 提交/推送失败：${e.stderr || e.message}`);
    process.exit(1);
  }
  console.log("========== 完成 ==========");
}
