/**
 * 学术 API 中转代理 — Vercel Serverless Function (JavaScript)
 *
 * 支持的学术 API:
 *   /api/academic/openalex/*    → https://api.openalex.org/*
 *   /api/academic/s2/*          → https://api.semanticscholar.org/*
 *   /api/academic/firecrawl/*   → https://api.firecrawl.dev/*
 *
 * API Key 自动注入（从 Vercel 环境变量读取，不暴露在前端）:
 *   OPENALEX_API_KEY    → OpenAlex api_key 查询参数
 *   S2_API_KEY          → Semantic Scholar x-api-key 请求头
 *   FIRECRAWL_API_KEY   → Firecrawl Authorization: Bearer 请求头
 *
 * 鉴权: X-Relay-Secret 头（与飞书中转共用）
 *
 * 部署: 将此文件放到 Vercel 项目的 api/academic/index.js
 *       vercel.json 配置 rewrites: /api/academic/:path* → /api/academic
 */

const RELAY_SECRET = process.env.RELAY_SECRET || '';

// 服务后端映射
const BACKENDS = {
  openalex: {
    baseUrl: 'https://api.openalex.org',
    keyEnv: 'OPENALEX_API_KEY',
    // OpenAlex 用查询参数 ?api_key=xxx
    injectKey: (headers, url, key) => {
      const sep = url.includes('?') ? '&' : '?';
      return { headers, url: `${url}${sep}api_key=${key}` };
    },
  },
  s2: {
    baseUrl: 'https://api.semanticscholar.org',
    keyEnv: 'S2_API_KEY',
    // Semantic Scholar 用 x-api-key 头
    injectKey: (headers, url, key) => {
      headers['x-api-key'] = key;
      return { headers, url };
    },
  },
  firecrawl: {
    baseUrl: 'https://api.firecrawl.dev',
    keyEnv: 'FIRECRAWL_API_KEY',
    // Firecrawl 用 Authorization: Bearer 头
    injectKey: (headers, url, key) => {
      headers['Authorization'] = `Bearer ${key}`;
      return { headers, url };
    },
  },
};

export default async function handler(req, res) {
  // ── CORS ──
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Relay-Secret');

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

  // ── 从 req.url 解析服务名和路径 ──
  // req.url 格式: /api/academic/openalex/works?search=deep+learning
  // 需要提取: service=openalex, path=works, query=search=deep+learning
  const rawUrl = req.url || '';
  let pathPart = rawUrl;
  const prefix = '/api/academic';
  if (pathPart.startsWith(prefix)) {
    pathPart = pathPart.slice(prefix.length);
  }
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

  // Vercel rewrite (:path*) 会把捕获的路径作为 path= 查询参数注入
  // 需要过滤掉这个多余的参数，否则目标 API 会收到非法参数
  if (queryString) {
    const cleanParams = queryString
      .split('&')
      .filter(kv => !kv.startsWith('path=') && !kv.startsWith('path%3D'));
    queryString = cleanParams.join('&');
  }
  // 第一段是服务名，剩下的是 API 路径
  const segments = pathStr.split('/').filter(Boolean);
  if (segments.length === 0) {
    res.status(400).json({
      error: 'Bad Request',
      message: 'Missing service name. Use /api/academic/{openalex|s2|firecrawl}/...',
      availableServices: Object.keys(BACKENDS),
    });
    return;
  }

  const serviceName = segments[0].toLowerCase();
  const apiPath = segments.slice(1).join('/');

  const backend = BACKENDS[serviceName];
  if (!backend) {
    res.status(400).json({
      error: 'Bad Request',
      message: `Unknown service: ${serviceName}`,
      availableServices: Object.keys(BACKENDS),
    });
    return;
  }

  // ── 构建目标 URL ──
  let targetUrl = queryString
    ? `${backend.baseUrl}/${apiPath}?${queryString}`
    : `${backend.baseUrl}/${apiPath}`;

  // ── 构建转发 headers ──
  let forwardHeaders = {};
  if (req.headers['content-type']) {
    forwardHeaders['Content-Type'] = req.headers['content-type'];
  }
  // 不透传原始 Authorization（用各服务自己的 key 注入）

  // ── 注入 API Key ──
  const apiKey = process.env[backend.keyEnv] || '';
  if (apiKey) {
    const result = backend.injectKey(forwardHeaders, targetUrl, apiKey);
    forwardHeaders = result.headers || forwardHeaders;
    targetUrl = result.url || targetUrl;
  } else {
    // API key 未配置，返回提示
    res.status(500).json({
      error: 'Server Configuration Error',
      message: `Environment variable ${backend.keyEnv} is not set on Vercel`,
    });
    return;
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
    console.error(`Academic relay error (${serviceName}):`, err);
    res.status(502).json({
      error: 'Bad Gateway',
      message: err && err.message ? err.message : `Failed to reach ${serviceName} API`,
    });
  }
}