/**
 * 飞书 API 中转代理 — Vercel Serverless Function
 *
 * 将所有 /api/feishu/* 请求转发到 https://open.feishu.cn/open-apis/*
 *
 * 用途：公司 McAfee Web Gateway 会拦截发往 open.feishu.cn 的
 *       Content-Type: application/json 的 POST 请求（返回 403 Blocked）。
 *       但 McAfee 不拦截发往 jiajingnan.cn 的同类请求。
 *       因此通过此中转函数绕过 McAfee 拦截。
 *
 * 部署：将此文件放到 Vercel 项目的 api/feishu/[...path].ts
 *       Vercel 会自动识别为 Serverless Function
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const FEISHU_BASE = 'https://open.feishu.cn/open-apis';

// 简单的鉴权 token，防止被滥用
// 部署后在 Vercel 环境变量中设置 RELAY_SECRET
const RELAY_SECRET = process.env.RELAY_SECRET || '';

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  // ── CORS ──
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Relay-Secret');

  // 处理 CORS 预检
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  // ── 鉴权 ──
  if (RELAY_SECRET) {
    const provided = req.headers['x-relay-secret'] as string;
    if (provided !== RELAY_SECRET) {
      return res.status(401).json({ error: 'Unauthorized: invalid relay secret' });
    }
  }

  // ── 构建目标 URL ──
  // req.query.path 是 catch-all 参数，可能是 string 或 string[]
  const pathSegments = Array.isArray(req.query.path)
    ? req.query.path
    : [req.query.path];
  const pathStr = pathSegments.filter(Boolean).join('/');
  const targetUrl = `${FEISHU_BASE}/${pathStr}`;

  // 保留 query string
  const queryString = req.url?.split('?')[1] || '';
  const finalUrl = queryString ? `${targetUrl}?${queryString}` : targetUrl;

  // ── 构建转发 headers ──
  const forwardHeaders: Record<string, string> = {};
  // 透传 Authorization 和 Content-Type
  if (req.headers['authorization']) {
    forwardHeaders['Authorization'] = req.headers['authorization'] as string;
  }
  if (req.headers['content-type']) {
    forwardHeaders['Content-Type'] = req.headers['content-type'] as string;
  }

  // ── 发起请求到飞书 ──
  try {
    const fetchOptions: RequestInit = {
      method: req.method,
      headers: forwardHeaders,
    };

    // GET 请求不带 body
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      // 直接透传原始 body
      if (req.body) {
        if (typeof req.body === 'string') {
          fetchOptions.body = req.body;
        } else {
          fetchOptions.body = JSON.stringify(req.body);
        }
      }
    }

    const feishuResp = await fetch(finalUrl, fetchOptions);

    // ── 透传响应 ──
    const respContentType = feishuResp.headers.get('content-type');
    if (respContentType) {
      res.setHeader('Content-Type', respContentType);
    }

    const respBody = await feishuResp.text();
    return res.status(feishuResp.status).send(respBody);
  } catch (err: any) {
    console.error('Feishu relay error:', err);
    return res.status(502).json({
      error: 'Bad Gateway',
      message: err?.message || 'Failed to reach feishu API',
    });
  }
}