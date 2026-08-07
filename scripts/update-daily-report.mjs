#!/usr/bin/env node
/**
 * update-daily-report.mjs
 *
 * 定时任务全流程脚本（每晚 23:00 由自动化任务调用）：
 *   1. 同步 tracking/weekly 下最新的项目日报到 public/projects/latest.html（固定文件名，直接替换旧内容）
 *   2. 若内容有变化：git add + commit + push 到 GitHub（Vercel 会自动重新构建部署）
 *   3. 若内容无变化：跳过提交，直接结束（避免空提交）
 *
 * 本脚本只负责"部署"，日报内容由 tracking 项目的定时任务生成。
 * 用法：node scripts/update-daily-report.mjs
 */
import { execSync } from "node:child_process";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { syncReports } from "./sync-weekly.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");

function run(cmd) {
  return execSync(cmd, { cwd: projectRoot, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
}

console.log("========== 每日日报部署任务 ==========");

// 1. 同步最新日报（tracking/weekly -> public/projects/latest.html）
const { changed, latest, error } = syncReports();

if (error) {
  console.error(`[update-daily-report] 同步失败：${error}`);
  process.exit(1);
}

if (!changed) {
  console.log(`[update-daily-report] 日报内容无变化（${latest?.period ?? "未知"}），跳过提交推送`);
  process.exit(0);
}

// 2. 提交并推送（内容有变化时）
try {
  run("git add public/projects/");
  const commitMsg = `chore: 更新项目日报 ${latest.period}`;
  run(`git commit -m "${commitMsg}"`);
  console.log(`[update-daily-report] 已提交：${commitMsg}`);

  // 优先使用 HTTP/1.1 推送（此前遇到 HTTP/2 framing 不稳定问题）
  try {
    run("git -c http.version=HTTP/1.1 push origin main");
  } catch {
    console.warn("[update-daily-report] HTTP/1.1 推送失败，尝试默认协议重试...");
    run("git push origin main");
  }
  console.log("[update-daily-report] 已推送到 GitHub，Vercel 将自动重新部署");
} catch (e) {
  console.error(`[update-daily-report] 提交/推送失败：${e.stderr || e.message}`);
  process.exit(1);
}

console.log("========== 完成 ==========");
