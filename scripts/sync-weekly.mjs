#!/usr/bin/env node
/**
 * sync-weekly.mjs
 *
 * 将 /Users/olivia/Desktop/jiajingnan/tracking/weekly/ 下的项目周报 HTML
 * 同步到 public/projects/，并生成 _index.json 供"项目总揽"页面使用。
 *
 * 用法：node scripts/sync-weekly.mjs
 * 无外部路径时可安全跳过（打印提示，不报错）。
 */
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");
const sourceDir = "/Users/olivia/Desktop/jiajingnan/tracking/weekly";
const targetDir = join(projectRoot, "public", "projects");

if (!existsSync(sourceDir)) {
  console.warn(`[sync-weekly] 跳过：源目录不存在 ${sourceDir}`);
  process.exit(0);
}

mkdirSync(targetDir, { recursive: true });

const files = readdirSync(sourceDir)
  .filter((f) => f.endsWith(".html"))
  .sort()
  .reverse();

const reports = [];

for (const file of files) {
  const src = join(sourceDir, file);
  const dest = join(targetDir, file);

  // 复制文件（保留时间戳）
  cpSync(src, dest, { force: true });

  const html = readFileSync(src, "utf8");
  const stat = statSync(src);

  // 解析 <title>，如 "2026-W32 项目周报"
  const titleMatch = html.match(/<title>([^<]+)<\/title>/);
  const title = titleMatch ? titleMatch[1].trim() : file.replace(/\.html$/, "");

  // 解析生成时间，如 "生成时间：2026-08-07 11:53"
  const genMatch = html.match(/生成时间[：:]\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s*[0-9]{2}:[0-9]{2})/);
  const generatedAt = genMatch ? genMatch[1].trim() : "";

  reports.push({
    file,
    url: `/projects/${file}`,
    title,
    period: file.replace(/\.html$/, ""),
    generatedAt,
    size: stat.size,
    updatedAt: stat.mtimeMs,
  });
}

writeFileSync(join(targetDir, "_index.json"), JSON.stringify(reports, null, 2), "utf8");
console.log(`[sync-weekly] 已同步 ${reports.length} 份周报 -> public/projects/`);
