# 公司侧导出工具（拷走这个目录即可）

这个目录里是**要在公司电脑/服务器上运行**的代码，与博客本体无关，单独存放。

## 文件

- `export-uploader.mjs` — 批量上传 CLI（零依赖，只需 Node 18+）

## 快速用法

```bash
# 1. 预览将要上传什么（不发起网络请求）
node export-uploader.mjs --dir /path/to/项目目录 --owner <GitHub用户名> --repo <私有仓库名> --dry-run

# 2. 正式上传（口令 = 博客 Vercel 的 RELAY_SECRET）
EXPORT_RELAY_SECRET=口令 node export-uploader.mjs \
  --dir /path/to/项目目录 --owner <GitHub用户名> --repo <私有仓库名> --label exp-A

# 3. 中断后续传（批次 id 用第 2 步打印的那个）
EXPORT_RELAY_SECRET=口令 node export-uploader.mjs --dir ... --owner ... --repo ... --batch <批次id>
```

零星手工提交（粘贴 LaTeX/单个文件）用网页版：`https://www.jiajingnan.cn/export/`（同一口令）。

**分块协议**：公司网关请求体上限约 10KB（实测），脚本自动分块——文本 3000 字符/块、二进制 6.6KB/块（base64），并发 3 上传。速度约 100-200KB/s：1MB≈10-20 秒，10MB≈2-3 分钟。大二进制建议先 zip 压缩再传。

完整部署与配置说明见博客仓库 `docs/export-pipeline.md`。

## 发送前自检清单（git 历史不可擦，务必过一遍）

- [ ] 目录里没有密钥/密码/token（`.env`、`credentials`、`id_rsa`…）
- [ ] 没有公司内部域名、IP、系统名等专有标识
- [ ] 目标仓库保持 **private**
- [ ] 已获领导批准的范围内
