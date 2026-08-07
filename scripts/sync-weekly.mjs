#!/usr/bin/env node
/**
 * sync-weekly.mjs
 *
 * 将 /Users/olivia/Desktop/jiajingnan/tracking/weekly/ 下最新的项目日报 HTML
 * 同步为 public/projects/latest.html（固定文件名，直接替换旧内容），
 * 并生成 _index.json 供"项目总揽"页面使用。
 *
 * 设计为"单页替换"模式：每次只保留最新一份，不积累历史。
 * 可直接运行，也可作为模块被 update-daily-report.mjs 导入。
 *
 * 用法：node scripts/sync-weekly.mjs
 * 无外部路径时可安全跳过（打印提示，不报错）。
 */
import {
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");
const sourceDir = "/Users/olivia/Desktop/jiajingnan/tracking/weekly";
const targetDir = join(projectRoot, "public", "projects");
const targetFile = "latest.html";
const targetPath = join(targetDir, targetFile);
const indexPath = join(targetDir, "_index.json");

function fileHash(filePath) {
  return createHash("sha256")
    .update(readFileSync(filePath))
    .digest("hex")
    .slice(0, 12);
}

/**
 * 同步最新日报到 public/projects/latest.html。
 * @returns {{ changed: boolean, latest: object|null, error: string|null }}
 */
export function syncReports() {
  if (!existsSync(sourceDir)) {
    console.warn(`[sync-weekly] 跳过：源目录不存在 ${sourceDir}`);
    return { changed: false, latest: null, error: `源目录不存在: ${sourceDir}` };
  }

  const files = readdirSync(sourceDir)
    .filter((f) => f.endsWith(".html"))
    .sort()
    .reverse();

  if (files.length === 0) {
    console.warn("[sync-weekly] 跳过：源目录中没有 HTML 文件");
    return { changed: false, latest: null, error: "源目录中没有 HTML 文件" };
  }

  const newest = files[0]; // 文件名倒序的第一个即最新
  const src = join(sourceDir, newest);
  const stat = statSync(src);
  const html = readFileSync(src, "utf8");

  // 解析 <title>，如 "2026-W32 项目周报"
  const titleMatch = html.match(/<title>([^<]+)<\/title>/);
  const title = titleMatch ? titleMatch[1].trim() : newest.replace(/\.html$/, "");

  // 解析生成时间，如 "生成时间：2026-08-07 11:53"
  const genMatch = html.match(/生成时间[：:]\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s*[0-9]{2}:[0-9]{2})/);
  const generatedAt = genMatch ? genMatch[1].trim() : "";

  const latest = {
    file: targetFile,
    url: `/projects/${targetFile}`,
    title,
    period: newest.replace(/\.html$/, ""),
    generatedAt,
    size: stat.size,
    updatedAt: stat.mtimeMs,
    source: newest,
  };

  // 内容未变化时跳过（避免空提交）
  const changed = !existsSync(targetPath) || fileHash(src) !== fileHash(targetPath);

  mkdirSync(targetDir, { recursive: true });
  cpSync(src, targetPath, { force: true }); // 固定文件名覆盖 = 直接替换

  // 清理旧文件：只保留 latest.html 与 _index.json
  for (const f of readdirSync(targetDir)) {
    if (f !== targetFile && f !== "_index.json") {
      rmSync(join(targetDir, f), { force: true });
      console.log(`[sync-weekly] 清理旧文件: ${f}`);
    }
  }

  writeFileSync(indexPath, JSON.stringify([latest], null, 2), "utf8");

  console.log(
    changed
      ? `[sync-weekly] 已替换为最新日报: ${newest} -> ${targetFile}`
      : `[sync-weekly] 内容无变化（最新仍为 ${newest}）`
  );
  return { changed, latest, error: null };
}

// CLI 直接运行时执行
const isMain =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) {
  syncReports();
}
