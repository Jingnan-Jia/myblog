/* eslint-disable no-console -- 服务端函数，日志只记元数据 */
/**
 * 成果导出通道 — 落盘函数 (Vercel Serverless, 零依赖 Node ESM)
 *
 * 流程：租约 claim 防卡死 → 读回分块拼装 → 逐文件 sha256/bytes 校验 →
 *       GitHub Git Data API 整批单 commit 写入指定私有仓库 → 清理暂存
 *
 * 协议：POST，请求体 = 首行 JSON {v:1,action:'finalize',batch,...}，
 *       Content-Type text/plain（公司网关拦截 application/json）。
 *
 * 幂等：done → 直接返回既有 commit；committing 且租约未过期 → 409；
 *       committing 且租约超 90s（函数被超时杀掉）→ 自动接管重跑。
 *
 * 环境：EXPORT_RELAY_SECRET（未设则回退 RELAY_SECRET）、SUPABASE_URL、
 *       SUPABASE_SERVICE_KEY、GITHUB_TOKEN、EXPORT_GIT_NAME、EXPORT_GIT_EMAIL
 */

import crypto from 'node:crypto';

const SUPABASE_URL = (process.env.SUPABASE_URL || '').replace(/\/$/, '');
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || '';
// 独立口令优先：EXPORT_RELAY_SECRET（导出通道专用）；未配置则沿用博客共用 RELAY_SECRET
const RELAY_SECRET = process.env.EXPORT_RELAY_SECRET || process.env.RELAY_SECRET || '';
const GITHUB_TOKEN = process.env.GITHUB_TOKEN || '';
const GIT_NAME = process.env.EXPORT_GIT_NAME || 'export-relay';
const GIT_EMAIL = process.env.EXPORT_GIT_EMAIL || 'export-relay@users.noreply.github.com';

const EMPTY_TREE_SHA = '4b825dc642cb6eb9a060e54bf8d69288fbee4904';
const LEASE_SECONDS = 90; // finalize 租约时长：超过则视为函数已被杀，允许接管
const MAX_FILES = 200; // 硬上限（文档建议单批 ≤60 文件 / ≤60MB）
const MAX_TOTAL_BYTES = 200 * 1024 * 1024;
const FIRST_LINE_MAX = 4096;

const GH_HEADERS = {
  Authorization: `Bearer ${GITHUB_TOKEN}`,
  Accept: 'application/vnd.github+json',
  'X-GitHub-Api-Version': '2022-11-28',
  'Content-Type': 'application/json',
  'User-Agent': 'export-relay',
};

function safeEqual(a, b) {
  const ba = Buffer.from(String(a));
  const bb = Buffer.from(String(b));
  if (ba.length !== bb.length) return false;
  return crypto.timingSafeEqual(ba, bb);
}

function authorized(req, meta) {
  if (!RELAY_SECRET) return false;
  const header = req.headers['x-relay-secret'];
  if (typeof header === 'string' && safeEqual(header, RELAY_SECRET)) return true;
  const expectedHash = crypto.createHash('sha256').update(RELAY_SECRET).digest('hex');
  const token = meta && meta.web_token;
  if (typeof token === 'string' && safeEqual(token, expectedHash)) return true;
  return false;
}

async function readBody(req) {
  const b = req.body;
  if (typeof b === 'string') return b;
  if (Buffer.isBuffer(b)) return b.toString('utf8');
  if (b && typeof b === 'object') return JSON.stringify(b);
  const chunks = [];
  for await (const c of req) chunks.push(typeof c === 'string' ? Buffer.from(c) : c);
  return Buffer.concat(chunks).toString('utf8');
}

function sbHeaders(extra = {}) {
  return {
    apikey: SUPABASE_SERVICE_KEY,
    Authorization: `Bearer ${SUPABASE_SERVICE_KEY}`,
    'Content-Type': 'application/json',
    ...extra,
  };
}

async function sbFetch(path, options = {}) {
  const res = await fetch(`${SUPABASE_URL}/rest/v1/${path}`, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`supabase ${path.split('?')[0]} -> ${res.status}: ${text.slice(0, 300)}`);
  }
  return res;
}

async function ghFetch(path, options = {}) {
  const res = await fetch(`https://api.github.com${path}`, { ...options, headers: GH_HEADERS });
  const bodyText = await res.text();
  let body = null;
  try {
    body = bodyText ? JSON.parse(bodyText) : null;
  } catch {
    body = null;
  }
  if (!res.ok) {
    const msg = (body && body.message) || res.statusText;
    const err = new Error(`github ${path.split('?')[0]} -> ${res.status}: ${msg}`);
    err.status = res.status;
    throw err;
  }
  return body;
}

// ── 批次状态落库 ──
async function setBatch(batchId, patch) {
  await sbFetch(`export_batches?batch_id=eq.${encodeURIComponent(batchId)}`, {
    method: 'PATCH',
    headers: sbHeaders({ Prefer: 'return=minimal' }),
    body: JSON.stringify({ ...patch, updated_at: new Date().toISOString() }),
  });
}

function hasPathEscape(p) {
  return !p || p.includes('\\') || p.includes('..') || p.startsWith('/') || p.includes('\0');
}

function joinPath(subdir, filePath) {
  const combined = subdir ? `${subdir.replace(/\/+$/, '')}/${filePath}` : filePath;
  if (hasPathEscape(combined)) throw new Error(`不安全的目标路径: ${combined}`);
  return combined;
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Relay-Secret');
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method Not Allowed' });
    return;
  }

  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY || !RELAY_SECRET || !GITHUB_TOKEN) {
    res.status(500).json({ error: 'Server Configuration Error', message: 'missing env' });
    return;
  }

  let meta;
  try {
    const raw = await readBody(req);
    const nl = raw.indexOf('\n');
    if (nl < 0 || nl > FIRST_LINE_MAX) throw new Error('first line (JSON meta) missing or too long');
    meta = JSON.parse(raw.slice(0, nl));
  } catch (e) {
    res.status(400).json({ error: 'Bad Request', message: e.message });
    return;
  }

  if (!authorized(req, meta)) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  const batchId = String(meta.batch || '');
  if (!/^[A-Za-z0-9._-]{1,80}$/.test(batchId)) {
    res.status(400).json({ error: 'Bad Request', message: 'bad batch id' });
    return;
  }

  try {
    await finalize(req, res, batchId);
  } catch (err) {
    console.error(`[export-finalize] batch=${batchId} error:`, err.message);
    try {
      await setBatch(batchId, { status: 'failed', error: err.message.slice(0, 1000) });
    } catch {
      // 状态更新失败不掩盖原始错误
    }
    res.status(500).json({ error: 'Finalize Failed', message: err.message.slice(0, 500), batch: batchId });
  }
}

async function finalize(req, res, batchId) {
  // ── 1) 租约 claim：pending/failed 可跑；committing 且租约超时视为死锁可接管 ──
  // 时间戳去掉毫秒（值中的 "." 会被 PostgREST 当作操作符分隔符）
  const leaseExpired = new Date(Date.now() - LEASE_SECONDS * 1000).toISOString().replace(/\.\d{3}Z$/, 'Z');
  const claimPath = `export_batches?batch_id=eq.${encodeURIComponent(batchId)}&or=(status.in.(pending,failed),and(status.eq.committing,committing_at.lt.${leaseExpired}))`;
  const claimRes = await sbFetch(claimPath, {
    method: 'PATCH',
    headers: sbHeaders({ Prefer: 'return=representation' }),
    body: JSON.stringify({ status: 'committing', committing_at: new Date().toISOString(), error: null }),
  });
  const claimed = await claimRes.json();

  if (!claimed.length) {
    const batchRes = await sbFetch(
      `export_batches?batch_id=eq.${encodeURIComponent(batchId)}&select=status,commit_sha,repo_owner,repo_name`,
      { headers: sbHeaders() }
    );
    const rows = await batchRes.json();
    if (!rows.length) {
      res.status(404).json({ error: 'Not Found', message: 'batch not found（未上传过分块或已完成清理）' });
      return;
    }
    const row = rows[0];
    if (row.status === 'done' && row.commit_sha) {
      res.status(200).json({
        ok: true,
        batch: batchId,
        commit_sha: row.commit_sha,
        html_url: `https://github.com/${row.repo_owner}/${row.repo_name}/commit/${row.commit_sha}`,
        already_done: true,
      });
      return;
    }
    res.status(409).json({ error: 'Conflict', message: `batch 状态为 ${row.status}，仍在处理中，请稍后再试` });
    return;
  }

  const batch = claimed[0];
  const { repo_owner: owner, repo_name: repo, branch, subdir, label } = batch;

  try {
    // ── 2) 读回分块并按文件拼装校验 ──
    const chunksRes = await sbFetch(
      `export_chunks?batch_id=eq.${encodeURIComponent(batchId)}&select=file_path,is_binary,idx,total,data,bytes,file_sha256&order=file_path.asc,idx.asc`,
      { headers: sbHeaders() }
    );
    const chunks = await chunksRes.json();
    if (!chunks.length) throw new Error('batch 无任何分块（请先上传文件）');

    const filesMap = new Map();
    for (const c of chunks) {
      let f = filesMap.get(c.file_path);
      if (!f) {
        f = { file_path: c.file_path, is_binary: c.is_binary, total: c.total, bytes: c.bytes, sha256: c.file_sha256, parts: [] };
        filesMap.set(c.file_path, f);
      }
      if (f.total !== c.total || f.sha256 !== c.file_sha256 || f.bytes !== c.bytes) {
        throw new Error(`文件 ${c.file_path} 的分块元数据不一致，请整文件重传`);
      }
      f.parts[c.idx] = c.data;
    }

    const files = Array.from(filesMap.values());
    if (files.length > MAX_FILES) throw new Error(`单批文件数 ${files.length} 超过硬上限 ${MAX_FILES}，请拆批`);
    let totalBytes = 0;

    const blobs = []; // {path, content: Buffer}
    for (const f of files) {
      for (let i = 0; i < f.total; i++) {
        if (typeof f.parts[i] !== 'string') {
          throw new Error(`文件 ${f.file_path} 缺少分块 ${i}/${f.total}，请重跑上传补齐`);
        }
      }
      const buf = f.is_binary ? Buffer.from(f.parts.join(''), 'base64') : Buffer.from(f.parts.join(''), 'utf8');
      if (buf.length !== f.bytes) {
        throw new Error(`文件 ${f.file_path} 字节数不符（期望 ${f.bytes}，实得 ${buf.length}）`);
      }
      const sha = crypto.createHash('sha256').update(buf).digest('hex');
      if (sha !== f.sha256) {
        throw new Error(`文件 ${f.file_path} sha256 校验失败，请整文件重传`);
      }
      totalBytes += buf.length;
      blobs.push({ path: joinPath(subdir, f.file_path), buf });
    }
    if (totalBytes > MAX_TOTAL_BYTES) throw new Error(`单批总量超硬上限 200MB，请拆批`);

    // ── 3) GitHub：仓库/分支探测 ──
    let repoInfo;
    try {
      repoInfo = await ghFetch(`/repos/${owner}/${repo}`);
    } catch (e) {
      if (e.status === 404) {
        throw new Error(`仓库 ${owner}/${repo} 不存在或 token 未授权该仓库。请先在 GitHub 手工建好私有仓库、并在 fine-grained token 中授权后重试`);
      }
      throw e;
    }

    let baseTreeSha = null;
    let parentSha = null;
    let branchExists = false;
    try {
      const ref = await ghFetch(`/repos/${owner}/${repo}/git/ref/heads/${branch}`);
      branchExists = true;
      parentSha = ref.object.sha;
      const commit = await ghFetch(`/repos/${owner}/${repo}/git/commits/${parentSha}`);
      baseTreeSha = commit.tree.sha;
    } catch (e) {
      if (e.status !== 404) throw e;
      // 分支不存在：仓库为空 → 以空树建首 commit；仓库非空 → 让用户改分支
      const branches = await ghFetch(`/repos/${owner}/${repo}/branches?per_page=1`);
      if (Array.isArray(branches) && branches.length > 0) {
        throw new Error(`分支 ${branch} 不存在且仓库非空。请改用已有分支，或先在 GitHub 手工建好该分支`);
      }
    }

    // ── 4) 逐文件 blob → tree → commit → ref ──
    const treeItems = [];
    for (const b of blobs) {
      const blob = await ghFetch(`/repos/${owner}/${repo}/git/blobs`, {
        method: 'POST',
        body: JSON.stringify({ content: b.buf.toString('base64'), encoding: 'base64' }),
      });
      treeItems.push({ path: b.path, mode: '100644', type: 'blob', sha: blob.sha });
    }

    const treePayload = { tree: treeItems };
    if (baseTreeSha) treePayload.base_tree = baseTreeSha;
    const tree = await ghFetch(`/repos/${owner}/${repo}/git/trees`, {
      method: 'POST',
      body: JSON.stringify(treePayload),
    });

    const labelPart = label ? ` - ${label}` : '';
    const commitPayload = {
      message: `export(batch): ${batchId} - ${blobs.length} files${labelPart}`,
      tree: tree.sha,
      author: { name: GIT_NAME, email: GIT_EMAIL },
      committer: { name: GIT_NAME, email: GIT_EMAIL },
    };
    if (parentSha) commitPayload.parents = [parentSha];
    const commit = await ghFetch(`/repos/${owner}/${repo}/git/commits`, {
      method: 'POST',
      body: JSON.stringify(commitPayload),
    });

    if (branchExists) {
      await ghFetch(`/repos/${owner}/${repo}/git/refs/heads/${branch}`, {
        method: 'PATCH',
        body: JSON.stringify({ sha: commit.sha }),
      });
    } else {
      await ghFetch(`/repos/${owner}/${repo}/git/refs`, {
        method: 'POST',
        body: JSON.stringify({ ref: `refs/heads/${branch}`, sha: commit.sha }),
      });
    }

    const htmlUrl = `https://github.com/${owner}/${repo}/commit/${commit.sha}`;

    // ── 5) 成功：清理暂存（chunks + batch 行），中转即到即走 ──
    await sbFetch(`export_chunks?batch_id=eq.${encodeURIComponent(batchId)}`, {
      method: 'DELETE',
      headers: sbHeaders({ Prefer: 'return=minimal' }),
    });
    await sbFetch(`export_batches?batch_id=eq.${encodeURIComponent(batchId)}`, {
      method: 'DELETE',
      headers: sbHeaders({ Prefer: 'return=minimal' }),
    });

    console.log(
      `[export-finalize] batch=${batchId} repo=${owner}/${repo} branch=${branch} files=${blobs.length} bytes=${totalBytes} commit=${commit.sha}`
    );
    res.status(200).json({
      ok: true,
      batch: batchId,
      repo: `${owner}/${repo}`,
      branch,
      commit_sha: commit.sha,
      html_url: htmlUrl,
      files: blobs.map((b) => ({ path: b.path, bytes: b.buf.length })),
    });
  } catch (err) {
    // 失败：保留分块供重试，仅置 failed
    await setBatch(batchId, { status: 'failed', error: err.message.slice(0, 1000) });
    throw err;
  }
}
