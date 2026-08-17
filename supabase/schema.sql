-- =============================================================
-- Agent 日志在线阅读 - Supabase Schema
-- 执行方式：Supabase 控制台 → SQL Editor → 粘贴本文件全部内容 → Run
-- 说明：anon 角色允许 insert/select（上传器与前端直连用），
--       密码保护在前端页面层（防君子不防小人）。
-- =============================================================

-- 1. 日志表：每条 agent 思考/工具调用记录一行
create table if not exists public.agent_logs (
  id bigint generated always as identity primary key,
  agent_id text not null,
  run_id text not null default 'default',
  ts timestamptz not null default now(),
  level text not null default 'info',            -- info / tool / thinking / error
  content text not null,                          -- 日志内容（思考文本或描述）
  tool_name text,                                 -- 工具调用时的工具名
  args jsonb                                      -- 工具调用参数
);

-- 2. 代理状态表：每个 agent 一行，仪表盘用
create table if not exists public.agents (
  agent_id text primary key,
  run_id text not null default 'default',
  status text not null default 'active',          -- active / idle / finished
  last_ts timestamptz not null default now(),     -- 最后一次心跳/写入时间
  last_line text                                  -- 最新一行内容（截断）
);

-- 3. 索引：按 agent+run+时间查询
create index if not exists idx_agent_logs_agent_ts
  on public.agent_logs (agent_id, run_id, ts desc);

-- 4. RLS 策略（anon 可写可读，无服务端密钥）
alter table public.agent_logs enable row level security;
drop policy if exists "agent_logs_anon_insert" on public.agent_logs;
create policy "agent_logs_anon_insert" on public.agent_logs
  for insert to anon with check (true);
drop policy if exists "agent_logs_anon_select" on public.agent_logs;
create policy "agent_logs_anon_select" on public.agent_logs
  for select to anon using (true);

alter table public.agents enable row level security;
drop policy if exists "agents_anon_all" on public.agents;
drop policy if exists "agents_anon_read" on public.agents;
create policy "agents_anon_read" on public.agents
  for select to anon using (true);
drop policy if exists "agents_anon_insert" on public.agents;
create policy "agents_anon_insert" on public.agents
  for insert to anon with check (true);
drop policy if exists "agents_anon_update" on public.agents;
create policy "agents_anon_update" on public.agents
  for update to anon using (true) with check (true);
-- 注意：上传器写 agents 走 upsert（insert + 冲突时 update），
-- 故需要 insert 与 update 权限；不开放 delete，防止 anon 删表。
