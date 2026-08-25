/**
* Supabase API 中转代理 — Vercel Serverless Function
*
* 将 /api/supabase/* 请求转发到 Supabase REST API
* 解决大华内网无法直接访问 *.supabase.co 的问题
*
* 部署: 放到 Vercel 项目的 api/supabase/index.js
*       vercel.json 配置 rewrites: /api/supabase/:path* → /api/supabase
*
* 环境变量:
*   SUPABASE_URL      - Supabase 项目地址 (如 https://xxx.supabase.co)
*   RELAY_SECRET      - 可选鉴权密钥
*/

const SUPABASE_URL = (process.env.SUPABASE_URL || '').replace(/\/$/, '');
const RELAY_SECRET = process.env.RELAY_SECRET || '';

export default async function handler(req, res) {
// ── CORS ──
res.setHeader('Access-Control-Allow-Origin', '*');
res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, PATCH, DELETE, OPTIONS');
res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, apikey, Prefer, X-Relay-Secret');

if (req.method === 'OPTIONS') {
res.status(204).end();
return;
}

// ── 鉴权 ──
if (RELAY_SECRET) {
const provided = req.headers['x-relay-secret'];
if (provided !== RELAY_SECRET) {
res.status(401).json({ error: 'Unauthorized: invalid relay secret' });
return;
}
}

if (!SUPABASE_URL) {
res.status(500).json({ error: 'Server Configuration Error', message: 'SUPABASE_URL not set' });
return;
}

// ── 解析路径 ──
// Vercel rewrite /api/supabase/:path* → /api/supabase 会丢失路径段
// :path* 会作为 req.query.path 传入
const parsedUrl = new URL(req.url || '', 'http://localhost');
let pathPart = parsedUrl.pathname;
const prefix = '/api/supabase';
if (pathPart.startsWith(prefix)) {
pathPart = pathPart.slice(prefix.length);
}

// Vercel rewrite 丢失了路径段，从 req.query.path 恢复
if (!pathPart || pathPart === '' || pathPart === '/') {
const queryPath = req.query && req.query.path;
if (queryPath) {
const pathStr = Array.isArray(queryPath) ? queryPath.join('/') : queryPath;
pathPart = '/' + pathStr;
}
}

// 重建 query string（去掉 Vercel 注入的 path 参数）
const searchParams = new URLSearchParams(parsedUrl.searchParams);
searchParams.delete('path');
const queryString = searchParams.toString();
const targetUrl = `${SUPABASE_URL}${pathPart}${queryString ? '?' + queryString : ''}`;

// ── 构建转发 headers ──
const forwardHeaders = {};
const allowHeaders = [
'authorization',
'apikey',
'prefer',
'x-client-info',
'x-supabase-api-version',
];
for (const h of allowHeaders) {
if (req.headers[h]) {
forwardHeaders[h] = req.headers[h];
}
}
// 客户端统一用 text/plain 绕过 McAfee，转发到 Supabase 时强制覆盖为 application/json
if (req.method !== 'GET' && req.method !== 'HEAD') {
forwardHeaders['content-type'] = 'application/json';
}

// ── 发起请求 ──
try {
const fetchOptions = {
method: req.method,
headers: forwardHeaders,
};

if (req.method !== 'GET' && req.method !== 'HEAD') {
if (req.body) {
if (typeof req.body === 'string') {
fetchOptions.body = req.body;
} else {
fetchOptions.body = JSON.stringify(req.body);
}
}
}

const apiResp = await fetch(targetUrl, fetchOptions);

// ── 透传响应 ──
const respContentType = apiResp.headers.get('content-type');
if (respContentType) {
res.setHeader('Content-Type', respContentType);
}

const respBody = await apiResp.text();
res.status(apiResp.status).send(respBody);
} catch (err) {
console.error('Supabase relay error:', err);
res.status(502).json({
error: 'Bad Gateway',
message: err && err.message ? err.message : 'Failed to reach Supabase',
});
}
}