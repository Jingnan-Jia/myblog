/* eslint-disable no-console -- 服务端函数，日志只记元数据 */
/**
 * 成果导出通道 — 上传中转函数 (Vercel Serverless, 零依赖 Node ESM)
 *
 * 协议：请求体 = 首行 JSON 元数据 + '\n' + 正文，Content-Type 一律 text/plain
 *   （公司网关拦截 POST application/json，见 api/feishu/index.js 同类注释）。
 *   POST：正文 = 文本原文段（binary=false）或 base64 分块段（binary=true）
 *   GET ?batch=xxx：断点续传查询，只返回已收分块索引与元数据，绝不回传内容
 *
 * 鉴权（单一口令双通道）：
 *   CLI：请求头 X-Relay-Secret = RELAY_SECRET 明文
 *   网页：meta.web_token = sha256(口令)，与 sha256(RELAY_SECRET) 常数时间比较
 *
 * 环境：RELAY_SECRET、SUPABASE_URL、SUPABASE_SERVICE_KEY（service_role）
 * 暂存表 export_batches / export_chunks 仅 service_role 可访问（RLS 零 anon 策略）
 */

import crypto from 'node:crypto';

const SUPABASE_URL = (process.env.SUPABASE_URL || '').replace(/\/$/, '');
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || '';
const RELAY_SECRET = process.env.RELAY_SECRET || '';

const FIRST_LINE_MAX = 4096; // 首行元数据长度上限，防滥用
const BODY_MAX = 32 * 1024; // 公司网关对请求体实测上限约 10KB（见 company/ 探索记录），
// 客户端每块 body ≤ ~9.5KB；此处 32KB 仅作服务端防未分片大 body 的兜底

// ── 常数时间比较（长度不同直接 false，不泄露时序信息） ──
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
  const token = (meta && meta.web_token) || (req.query && req.query.token);
  if (typeof token === 'string' && safeEqual(token, expectedHash)) return true;
  return false;
}

// ── 读取请求体（兼容 Vercel 预解析 / 原始流两种形态） ──
async function readBody(req) {
  const b = req.body;
  if (typeof b === 'string') return b;
  if (Buffer.isBuffer(b)) return b.toString('utf8');
  if (b && typeof b === 'object') return JSON.stringify(b);
  const chunks = [];
  for await (const c of req) chunks.push(typeof c === 'string' ? Buffer.from(c) : c);
  return Buffer.concat(chunks).toString('utf8');
}

// ── Supabase REST (service_role) ──
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
    throw new Error(`supabase ${path} -> ${res.status}: ${text.slice(0, 300)}`);
  }
  return res;
}

// ── 元数据字段校验 ──
const RE_NAME = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62})$/; // owner / repo
const RE_BRANCH = /^[A-Za-z0-9._/-]{1,120}$/;
const RE_BATCH = /^[A-Za-z0-9._-]{1,80}$/;

function validateRelPath(p, maxLen = 512) {
  if (typeof p !== 'string' || !p || p.length > maxLen) return false;
  if (p.includes('\\') || p.includes('..') || p.startsWith('/')) return false;
  return p.split('/').every((seg) => seg.length > 0 && /[A-Za-z0-9._\-+ ()\[\]]/.test(seg[0]));
}

function validMeta(meta) {
  if (!meta || meta.v !== 1 || meta.action !== 'upload') return 'bad action/version';
  if (!RE_BATCH.test(String(meta.batch || ''))) return 'bad batch id';
  if (!RE_NAME.test(String(meta.repo_owner || ''))) return 'bad repo_owner';
  if (!RE_NAME.test(String(meta.repo_name || ''))) return 'bad repo_name';
  if (!RE_BRANCH.test(String(meta.branch || 'main'))) return 'bad branch';
  const subdir = meta.subdir || '';
  if (subdir && !validateRelPath(subdir, 200)) return 'bad subdir';
  if (typeof meta.label === 'string' && meta.label.length > 120) return 'label too long';
  if (!validateRelPath(meta.file_path)) return 'bad file_path';
  const idx = Number(meta.idx);
  const total = Number(meta.total);
  if (!Number.isInteger(idx) || !Number.isInteger(total) || idx < 0 || total < 1 || idx >= total)
    return 'bad idx/total';
  if (total > 100000) return 'too many chunks';
  if (typeof meta.bytes !== 'number' || meta.bytes < 0 || meta.bytes > 1024 * 1024 * 1024)
    return 'bad bytes';
  if (typeof meta.file_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(meta.file_sha256))
    return 'bad file_sha256';
  return null;
}

export default async function handler(req, res) {
  // ── CORS ──
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Relay-Secret');
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }

  // ── 配置自检 ──
  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY || !RELAY_SECRET) {
    res.status(500).json({ error: 'Server Configuration Error', message: 'missing env' });
    return;
  }

  // ── 鉴权（GET 用 query token，POST 用 header 或首行 meta.web_token） ──
  let meta = null;
  if (req.method === 'POST') {
    const raw = await readBody(req);
    if (raw.length > BODY_MAX) {
      res.status(413).json({ error: 'Payload Too Large' });
      return;
    }
    const nl = raw.indexOf('\n');
    if (nl < 0 || nl > FIRST_LINE_MAX) {
      res.status(400).json({ error: 'Bad Request', message: 'first line (JSON meta) missing or too long' });
      return;
    }
    try {
      meta = JSON.parse(raw.slice(0, nl));
    } catch {
      res.status(400).json({ error: 'Bad Request', message: 'first line is not valid JSON' });
      return;
    }
    if (!authorized(req, meta)) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    await handleUpload(req, res, meta, raw.slice(nl + 1));
    return;
  }

  if (req.method === 'GET') {
    if (!authorized(req, null)) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    await handleStatus(req, res);
    return;
  }

  res.status(405).json({ error: 'Method Not Allowed' });
}

// ── POST：登记批次 + 落库分块（upsert 覆盖式断点续传） ──
async function handleUpload(req, res, meta, payload) {
  try {
    const invalid = validMeta(meta);
    if (invalid) {
      res.status(400).json({ error: 'Bad Request', message: invalid });
      return;
    }
    if (!payload.length) {
      res.status(400).json({ error: 'Bad Request', message: 'empty payload' });
      return;
    }
    if (meta.binary && !/^[A-Za-z0-9+/=\r\n]+$/.test(payload.slice(0, 1024))) {
      res.status(400).json({ error: 'Bad Request', message: 'binary chunk must be base64' });
      return;
    }

    // 1) 登记批次（幂等 upsert；已 done 的批次行已删除，重传会重建为 pending）
    await sbFetch('export_batches?on_conflict=batch_id', {
      method: 'POST',
      headers: sbHeaders({ Prefer: 'resolution=merge-duplicates,return=minimal' }),
      body: JSON.stringify({
        batch_id: meta.batch,
        repo_owner: meta.repo_owner,
        repo_name: meta.repo_name,
        branch: meta.branch || 'main',
        subdir: meta.subdir || '',
        label: meta.label || '',
        status: 'pending',
        total_files: 0,
        total_bytes: 0,
        updated_at: new Date().toISOString(),
      }),
    });

    // 2) 落库分块（唯一约束 batch_id+file_path+idx，冲突即覆盖 → 断点重跑安全）
    await sbFetch('export_chunks?on_conflict=batch_id,file_path,idx', {
      method: 'POST',
      headers: sbHeaders({ Prefer: 'resolution=merge-duplicates,return=minimal' }),
      body: JSON.stringify({
        batch_id: meta.batch,
        file_path: meta.file_path,
        binary: Boolean(meta.binary),
        idx: Number(meta.idx),
        total: Number(meta.total),
        data: payload,
        bytes: Number(meta.bytes),
        file_sha256: meta.file_sha256,
      }),
    });

    // 日志只记元数据，不打印正文
    console.log(
      `[export-upload] batch=${meta.batch} file=${meta.file_path} idx=${meta.idx}/${meta.total} binary=${meta.binary}`
    );
    res.status(200).json({ ok: true, batch: meta.batch, file_path: meta.file_path, idx: meta.idx });
  } catch (err) {
    console.error('[export-upload] error:', err.message);
    res.status(502).json({ error: 'Upload Failed', message: err.message.slice(0, 500) });
  }
}

// ── GET ?batch=xxx：断点查询，只返回已收 idx 与文件元数据，不回传内容 ──
async function handleStatus(req, res) {
  try {
    const batch = req.query && req.query.batch;
    if (!batch || !RE_BATCH.test(batch)) {
      res.status(400).json({ error: 'Bad Request', message: 'bad batch id' });
      return;
    }

    const batchRes = await sbFetch(
      `export_batches?batch_id=eq.${encodeURIComponent(batch)}&select=batch_id,repo_owner,repo_name,branch,subdir,label,status,commit_sha,error,updated_at`,
      { headers: sbHeaders() }
    );
    const batchRows = await batchRes.json();
    if (!batchRows.length) {
      res.status(404).json({ error: 'Not Found', message: 'batch not found (未开始或已完成清理)' });
      return;
    }

    const chunksRes = await sbFetch(
      `export_chunks?batch_id=eq.${encodeURIComponent(batch)}&select=file_path,binary,total,bytes,file_sha256,idx&order=file_path.asc,idx.asc`,
      { headers: sbHeaders() }
    );
    const chunks = await chunksRes.json();

    const files = new Map();
    for (const c of chunks) {
      let f = files.get(c.file_path);
      if (!f) {
        f = { file_path: c.file_path, binary: c.binary, total: c.total, bytes: c.bytes, file_sha256: c.file_sha256, received: [] };
        files.set(c.file_path, f);
      }
      f.received.push(c.idx);
    }

    res.status(200).json({ batch: batchRows[0], files: Array.from(files.values()) });
  } catch (err) {
    console.error('[export-status] error:', err.message);
    res.status(502).json({ error: 'Status Query Failed', message: err.message.slice(0, 500) });
  }
}
