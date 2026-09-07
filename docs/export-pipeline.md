# 成果导出通道部署与使用手册

经领导批准的"公司 → 个人 GitHub"导出管道。三段式架构：

```
公司侧（网页 /export/ 或 company/export-uploader.mjs CLI）
  └─POST text/plain→ Vercel 中转函数（api/export/upload.js → Supabase 暂存；api/export/finalize.js → GitHub 单 commit）
                      └─成功后暂存即清理，不在博客侧留存任何内容
```

> 为什么 text/plain：公司网关拦截 POST application/json（见 `api/feishu/index.js` 注释）。
> 为什么一切请求只访问博客域名：公司内网无法直连 *.supabase.co / api.github.com，云端再代为访问。

## 一、一次性部署步骤

### 1. 建表（Supabase SQL Editor）

粘贴执行 `supabase/schema.sql` 末尾"成果导出通道"部分（export_batches / export_chunks 两表）。
RLS 已启用且**零 anon 策略**，只有持有 service_key 的云端函数能读写。

### 2. 手工建好目标私有仓库（每个项目一个，不自动建仓）

GitHub → New repository → 命名 → 勾选 **Private** → 勾选 **Add a README**（初始化 main 分支）→ Create。

### 3. 创建 fine-grained PAT

GitHub → Settings → Developer settings → Fine-grained tokens → Generate：

- Repository access：Only select repositories → 勾选所有目标导出仓库（天然白名单）
- Permissions：**Contents: Read and write**、Metadata: Read（必选）

### 4. 配置 Vercel 环境变量（Project Settings → Environment Variables）

| 变量 | 值 | 说明 |
|---|---|---|
| `RELAY_SECRET` | `口令明文` | 已有；CLI 直接用它当请求头 |
| `SUPABASE_URL` | 已有 | |
| `SUPABASE_SERVICE_KEY` | Supabase → Settings → API → service_role key | 仅云端使用，**绝不给公司侧** |
| `GITHUB_TOKEN` | 上一步的 fine-grained PAT | |
| `EXPORT_GIT_NAME` | 如 `export-relay` | commit 作者名 |
| `EXPORT_GIT_EMAIL` | 如 `xxx@users.noreply.github.com` | commit 作者邮箱 |

构建环境变量（Build Env，供网页密码门）：

| 变量 | 值 |
|---|---|
| `PUBLIC_EXPORT_PASSWORD_HASH` | `sha256(口令)` —— **与 RELAY_SECRET 同一口令** |
| `PUBLIC_EXPORT_DEFAULT_OWNER` | 可选，页面预填 GitHub 用户名 |
| `PUBLIC_EXPORT_DEFAULT_BRANCH` | 可选，默认 `main` |

口令哈希生成（任一机器）：

```bash
echo -n '你的口令' | shasum -a 256   # macOS
echo -n '你的口令' | sha256sum        # Linux
```

改口令 = 同时改 `RELAY_SECRET` 与 `PUBLIC_EXPORT_PASSWORD_HASH` 两个 env 并重新部署，立即全局生效。

### 5. 部署

推送到 main 触发 Vercel 部署（本仓库已含 `api/export/*.js`、`src/pages/export/index.astro`、`vercel.json` 函数登记，无需额外操作）。

## 二、使用

### 网页（零星手工提交）

`https://www.jiajingnan.cn/export/` → 输口令解锁 → 填 owner/仓库名/分支 → 拖文件或粘贴文本 → 开始传输 → 完成后展示 commit 链接。

### CLI（批量，公司侧）

把博客仓库的 `company/` 目录整个拷到公司机器，用法见 `company/README.md`：

```bash
node export-uploader.mjs --dir <目录> --owner <用户名> --repo <仓库名> --label <批次标签>
```

- 文本（合法 UTF-8）按 3000 字符/块、二进制按 6.6KB/块 base64 自动分块（适配网关 10KB 请求体上限）
- `--concurrency 3`（默认）并发上传；断点续传：中断后用**同一个 `--batch`** 重跑
- `--dry-run` 只列清单不联网

### 规模限制

**小分块协议**：公司网关对请求体实测上限约 **10KB**（10240B，见 `company/` 探索记录），因此所有内容自动分块，每块完整 body（meta 行 + 正文）≤ ~9.5KB：文本按 3000 字符/块，二进制按 6750 原始字节/块（base64 后 9000 字符）。断点续传按块跳过。

| 项 | 建议 | 硬上限 |
|---|---|---|
| 单批文件数 | ≤60 | 200 |
| 单批总量 | ≤10MB | 200MB |
| 传输速度 | 并发 3 约 100-200KB/s：1MB≈10-20s，10MB≈2-3 分钟 | — |

超限请拆批（换新 `--batch` 或网页新建会话）。每次批次 = 一个 commit。大二进制建议先压缩（zip/tar.gz）再传。

## 三、发送前自检清单（每次都过一遍）

git 历史不可擦除，推错无法真正撤回：

- [ ] 无密钥/密码/token（搜 `.env`、`secret`、`token`、`id_rsa`…）
- [ ] 无公司内部域名/IP/系统名等专有标识
- [ ] 目标仓库保持 private
- [ ] 内容在已批准范围内

## 四、运维

### 端到端冒烟验收（部署后做一次）

1. 手工建一个测试私有仓库（勾 README 初始化）
2. `echo hello > /tmp/t.txt`，CLI `--dir /tmp --repo 测试仓库 --label smoke`
3. 确认输出 commit 链接可打开、内容正确、仓库根就是 `t.txt`
4. Supabase 确认 `export_chunks` 已清空该批次

### 孤儿批次手工清理

失败批次会保留分块供重试；确认不再重试后可清理：

```sql
delete from export_chunks where batch_id = '<批次id>';
delete from export_batches where batch_id = '<批次id>';
```

### 卡在 committing 的批次

不需要处理：租约 90 秒自动过期，重跑 finalize 即可接管。也可用上面的 SQL 直接清理。

### 常见报错

| 报错 | 原因与处理 |
|---|---|
| `仓库 xx 不存在或 token 未授权` | 先在 GitHub 建好私有仓库、把仓库加进 PAT 授权列表，再重跑（分块已保留） |
| `分支 xx 不存在且仓库非空` | 网页/CLI 改用已有分支，或先在 GitHub 手工建分支 |
| `409 batch 状态为 committing` | 正在被处理（或 90 秒内被杀），稍等重试 |
| `sha256 校验失败` | 该文件传坏了，重跑上传（同 batch 会覆盖式重传该文件） |
