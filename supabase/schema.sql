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

-- =============================================================
-- 成果导出通道（export pipeline）— 临时暂存表
-- 与上面 agent_logs/agents 的 anon 可写模式【不同】：
--   这两张表承载代码/论文/数据等敏感内容，只允许云端中转函数
--   （持有 SUPABASE_SERVICE_KEY 的 service_role）读写。
--   RLS enabled 但【不建任何 anon policy】= 匿名完全不可访问，
--   service_role 默认绕过 RLS，无需额外策略。
-- 数据生命周期：finalize 成功写入 GitHub 后即整批删除，
--   中转侧不留任何内容副本；失败批次保留分块供重试。
-- =============================================================

-- 5. 导出批次表：一次上传会话一行
create table if not exists public.export_batches (
  batch_id     text primary key,
  repo_owner   text not null,
  repo_name    text not null,
  branch       text not null default 'main',
  subdir       text not null default '',
  label        text not null default '',
  status       text not null default 'pending',   -- pending / committing / done / failed
  committing_at timestamptz,                      -- finalize 租约时间戳，防函数超时被杀后批次永久卡 committing
  total_files  integer not null default 0,
  total_bytes  bigint not null default 0,
  commit_sha   text,
  error        text,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

-- 6. 导出分块表：文本直传存原文段，二进制存 base64 段
--    唯一约束 (batch_id, file_path, idx) + on conflict do update
--    实现覆盖式断点续传（重跑只补缺失/损坏分块）
create table if not exists public.export_chunks (
  id          bigint generated always as identity primary key,
  batch_id    text not null,
  file_path   text not null,
  is_binary   boolean not null default false,
  idx         integer not null,
  total       integer not null,
  data        text not null,
  bytes       bigint not null,                    -- 整个文件的字节数（非分块字节数）
  file_sha256 text not null,
  created_at  timestamptz not null default now(),
  constraint export_chunks_unique unique (batch_id, file_path, idx)
);

create index if not exists idx_export_chunks_batch
  on public.export_chunks (batch_id);

-- RLS：enabled + 零 policy = 仅 service_role（云端函数）可访问
alter table public.export_batches enable row level security;
alter table public.export_chunks enable row level security;
