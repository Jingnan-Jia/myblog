#!/usr/bin/env node
/**
 * sync-weekly.mjs
 *
 * 将 tracking 项目生成的最新日报/周报 HTML 加密同步到博客 public/projects/：
 *   - tracking/daily/ 下最新 HTML   -> public/projects/daily.enc    （XOR+Base64 加密，直接替换）
 *   - tracking/weekly/ 下最新 HTML  -> public/projects/weekly.enc   （XOR+Base64 加密，直接替换）
 *   - 生成 _check.enc 校验文件（用于页面验证密码是否正确）
 * 并生成 _index.json 供"项目总揽"页面使用（期数/时间信息，不含内容）。
 *
 * 内容加密存储，密钥由密码派生（SHA-256），浏览器端输入密码后解密查看。
 * 设计为"单页替换"模式：每个类型只保留最新一份，不积累历史。
 * 可直接运行，也可作为模块被 update-daily-report.mjs 导入。
 *
 * 用法：node scripts/sync-weekly.mjs
 * 无外部路径时可安全跳过（打印提示，不报错）。
 */
import {
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
const trackingRoot = "/Users/olivia/Desktop/jiajingnan/tracking";
const targetDir = join(projectRoot, "public", "projects");
const indexPath = join(targetDir, "_index.json");

// ===== 访问密码配置 =====
// 修改密码后需重新运行本脚本（或 update-daily-report.mjs）重新生成 .enc 文件，
// 再构建推送即可生效。
const PASSWORD = "090909";
const GATE_CHECK = "PROJECTS_GATE_OK"; // 校验文件内固定内容

// 日报与周报的源目录映射（tracking 的 gen_report.py 双周期输出）
const SOURCES = [
  { key: "daily", label: "日报", dir: "daily", targetFile: "daily.enc" },
  { key: "weekly", label: "周报", dir: "weekly", targetFile: "weekly.enc" },
];

// 加密：SHA-256(密码) 派生密钥，逐字节 XOR 后 Base64
function deriveKey(pwd) {
  return createHash("sha256").update(pwd).digest();
}

function xorCrypt(text, key) {
  const buf = Buffer.from(text, "utf8");
  for (let i = 0; i < buf.length; i++) buf[i] ^= key[i % key.length];
  return buf.toString("base64");
}

function xorDecrypt(b64, key) {
  const buf = Buffer.from(b64, "base64");
  for (let i = 0; i < buf.length; i++) buf[i] ^= key[i % key.length];
  return buf.toString("utf8");
}

function fileHashStr(str) {
  return createHash("sha256").update(str).digest("hex").slice(0, 12);
}

/**
 * 同步最新日报/周报（加密）到 public/projects/。
 * @returns {{ changed: boolean, reports: object[], error: string|null }}
 */
export function syncReports() {
  if (!existsSync(trackingRoot)) {
    console.warn(`[sync-weekly] 跳过：源目录不存在 ${trackingRoot}`);
    return { changed: false, reports: [], error: `源目录不存在: ${trackingRoot}` };
  }

  mkdirSync(targetDir, { recursive: true });
  const key = deriveKey(PASSWORD);
  const reports = [];
  let changed = false;

  for (const src of SOURCES) {
    const srcDir = join(trackingRoot, src.dir);
    if (!existsSync(srcDir)) {
      console.warn(`[sync-weekly] 跳过：${src.label}源目录不存在 ${srcDir}`);
      continue;
    }

    const files = readdirSync(srcDir)
      .filter((f) => f.endsWith(".html"))
      .sort()
      .reverse();

    if (files.length === 0) {
      console.warn(`[sync-weekly] 跳过：${src.label}源目录中没有 HTML 文件`);
      continue;
    }

    const newest = files[0]; // 文件名倒序的第一个即最新
    const sourcePath = join(srcDir, newest);
    const targetPath = join(targetDir, src.targetFile);
    const stat = statSync(sourcePath);
    const html = readFileSync(sourcePath, "utf8");

    // 解析 <title>（取第一个），如 "今日项目汇报（2026-08-07）" / "本周项目总览（2026-W32）"
    const titleMatch = html.match(/<title>([^<]+)<\/title>/);
    const title = titleMatch ? titleMatch[1].trim() : newest.replace(/\.html$/, "");

    // 解析生成时间，如 "生成时间：2026-08-07 14:47"
    const genMatch = html.match(/生成时间[：:]\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s*[0-9]{2}:[0-9]{2})/);
    const generatedAt = genMatch ? genMatch[1].trim() : "";

    reports.push({
      key: src.key,
      label: src.label,
      file: src.targetFile,
      url: `/projects/${src.targetFile}`,
      title,
      period: newest.replace(/\.html$/, ""),
      generatedAt,
      size: stat.size,
      updatedAt: stat.mtimeMs,
      source: newest,
    });

    // 内容未变化时跳过（对比源文件与已存在密文的内容哈希）
    const fileChanged =
      !existsSync(targetPath) ||
      fileHashStr(xorDecrypt(readFileSync(targetPath, "utf8"), key)) !==
        fileHashStr(html);

    writeFileSync(targetPath, xorCrypt(html, key), "utf8"); // 加密覆盖 = 直接替换
    changed = changed || fileChanged;
    console.log(
      fileChanged
        ? `[sync-weekly] ${src.label}已替换(加密): ${newest} -> ${src.targetFile}`
        : `[sync-weekly] ${src.label}内容无变化（${newest}）`
    );
  }

  // 写入密码校验文件（固定内容，页面据此判断密码是否正确）
  writeFileSync(join(targetDir, "_check.enc"), xorCrypt(GATE_CHECK, key), "utf8");

  // 清理旧文件：只保留 .enc 文件与 _index.json
  const keep = new Set(["daily.enc", "weekly.enc", "_check.enc", "_index.json"]);
  for (const f of readdirSync(targetDir)) {
    if (!keep.has(f)) {
      rmSync(join(targetDir, f), { force: true });
      console.log(`[sync-weekly] 清理旧文件: ${f}`);
    }
  }

  writeFileSync(indexPath, JSON.stringify(reports, null, 2), "utf8");

  if (reports.length === 0) {
    return { changed: false, reports, error: "日报/周报源目录均无 HTML 文件" };
  }
  return { changed, reports, error: null };
}

// CLI 直接运行时执行
const isMain =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) {
  syncReports();
}
