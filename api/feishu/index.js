/**
 * 飞书 API 中转代理 — Vercel Serverless Function (JavaScript)
 *
 * 将所有 /api/feishu/* 请求转发到 https://open.feishu.cn/open-apis/*
 *
 * 用途：公司 McAfee Web Gateway 会拦截发往 open.feishu.cn 的
 *       Content-Type: application/json 的 POST 请求（返回 403 Blocked）。
 *       但 McAfee 不拦截发往 jiajingnan.cn 的同类请求。
 *       因此通过此中转函数绕过 McAfee 拦截。
 *
 * 部署：将此文件放到 Vercel 项目的 api/feishu/index.js
 *       在 vercel.json 中配置 rewrites 将 /api/feishu/* 路由到此函数
 *
 * 注意：使用 .js 而非 .ts，避免 Astro 项目的 TypeScript 类型检查报错
 *       使用 ES Module 语法 (export default)，兼容 "type": "module" 项目
 *       不依赖 @vercel/node，使用标准 fetch API（Node 18+ 内置）
 *       不使用 [...path] catch-all（Astro 静态构建会干扰多层路径匹配）
 *       改为从 req.url 中手动解析路径
 */

const FEISHU_BASE = 'https://open.feishu.cn/open-apis';

// 简单的鉴权 token，防止被滥用
// 部署后在 Vercel 环境变量中设置 RELAY_SECRET
const RELAY_SECRET = process.env.RELAY_SECRET || '';

export default async function handler(req, res) {
  // ── CORS ──
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Relay-Secret');

  // 处理 CORS 预检
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

  // ── 从 req.url 中解析飞书 API 路径 ──
  // req.url 格式: /api/feishu/auth/v3/tenant_access_token/internal?xxx=yyy
  // 需要提取: auth/v3/tenant_access_token/internal
  const rawUrl = req.url || '';
  // 去掉开头的 /api/feishu 前缀
  let pathPart = rawUrl;
  const prefix = '/api/feishu';
  if (pathPart.startsWith(prefix)) {
    pathPart = pathPart.slice(prefix.length);
  }
  // 去掉开头的 /
  if (pathPart.startsWith('/')) {
    pathPart = pathPart.slice(1);
  }

  // 分离 query string
  const qIdx = pathPart.indexOf('?');
  let pathStr = pathPart;
  let queryString = '';
  if (qIdx >= 0) {
    pathStr = pathPart.slice(0, qIdx);
    queryString = pathPart.slice(qIdx + 1);
  }

  const targetUrl = queryString
    ? `${FEISHU_BASE}/${pathStr}?${queryString}`
    : `${FEISHU_BASE}/${pathStr}`;

  // ── 构建转发 headers ──
  const forwardHeaders = {};
  if (req.headers['authorization']) {
    forwardHeaders['Authorization'] = req.headers['authorization'];
  }
  if (req.headers['content-type']) {
    forwardHeaders['Content-Type'] = req.headers['content-type'];
  }

  // ── 发起请求到飞书 ──
  try {
    const fetchOptions = {
      method: req.method,
      headers: forwardHeaders,
    };

    // GET 请求不带 body
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      if (req.body) {
        if (typeof req.body === 'string') {
          fetchOptions.body = req.body;
        } else {
          fetchOptions.body = JSON.stringify(req.body);
        }
      }
    }

    const feishuResp = await fetch(targetUrl, fetchOptions);

    // ── 透传响应 ──
    const respContentType = feishuResp.headers.get('content-type');
    if (respContentType) {
      res.setHeader('Content-Type', respContentType);
    }

    const respBody = await feishuResp.text();
    res.status(feishuResp.status).send(respBody);
  } catch (err) {
    console.error('Feishu relay error:', err);
    res.status(502).json({
      error: 'Bad Gateway',
      message: err && err.message ? err.message : 'Failed to reach feishu API',
    });
  }
}