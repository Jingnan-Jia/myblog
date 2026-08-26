# myblog 项目进展日志

## 2026-08-07
- 阶段：网页开发
- 进展：新增「项目总揽」页面（src/pages/projects.astro），把追踪系统的日报/周报部署为博客页面（public/projects/daily.html、weekly.html），博客新增项目总览入口，可在线查看每日/每周项目进展。
- 关键文件：src/pages/projects.astro、public/projects/daily.html、public/projects/weekly.html

## 2026-08-07（21:37 扫描确认）
- 阶段：网页开发
- 进展：扫描确认「项目总揽加密版」已上线——日报/周报以 daily.enc / weekly.enc 密文同步（18:59 更新至 2026-08-07 日报与 2026-W32 周报），_check.enc 密码校验正常，线上无明文泄漏。
- 关键文件：src/pages/projects.astro、scripts/sync-weekly.mjs、public/projects/daily.enc、public/projects/_index.json

## 2026-08-09
- 阶段：网页开发
- 进展：新增「AI 副业选题」页面——sync-ai-topics.mjs 把选题库每日/每周 md 同步为 public/ai-topics/daily.html 并生成 _index.json（密码门入口），00:55 重新生成 projects 日报/周报页面，Header/Breadcrumb/i18n 组件同步更新。
- 关键文件：src/pages/ai-topics.astro、scripts/sync-ai-topics.mjs、public/ai-topics/_index.json、public/projects/daily.html

## 2026-08-10
- 阶段：网页开发
- 进展：「AI 副业选题」页面例行更新——08:51 重新生成 2026-08-10 每日选题与每周选题 HTML 并刷新 _index.json；projects 日报/周报页面于 08-09 23:24 更新。无实质代码改动。
- 关键文件：public/ai-topics/daily.html、public/ai-topics/weekly.html、public/ai-topics/_index.json

## 2026-08-11
- 阶段：网页开发
- 进展：「AI 副业选题」页面例行更新——08:51 重新生成 2026-08-11 每日选题与 2026-W33 每周内容日历并刷新 _index.json；projects 日报/周报页面更新至 08-10 版本。无实质代码改动，仅例行页面重生成。
- 关键文件：public/ai-topics/daily.html、public/ai-topics/weekly.html、public/ai-topics/_index.json、public/projects/daily.html

## 2026-08-12
- 阶段：网页开发
- 进展：「AI 副业选题」页面例行更新——08:52 重新生成 2026-08-12 每日选题与 2026-W33 每周内容日历并刷新 _index.json；projects 日报/周报页面更新至 08-11 版本。无实质代码改动，仅例行页面重生成。
- 关键文件：public/ai-topics/daily.html、public/ai-topics/weekly.html、public/ai-topics/_index.json、public/projects/daily.html

## 2026-08-15
- 阶段：网页开发
- 进展：08-14 23:11 例行部署检查——update-daily-report.mjs 执行成功，日报（tracking 源最新仍为 08-12）与周报（W32）内容均无变化，跳过 git commit/push；public/projects 下 _index.json/daily.html/weekly.html 仅时间戳刷新。无实质代码改动。
- 关键文件：public/projects/_index.json、public/projects/daily.html、public/projects/weekly.html

## 2026-08-16
- 阶段：网页开发
- 进展：「AI 副业选题」页面例行更新——20:31 重新生成每日选题与每周内容日历 HTML 并刷新 _index.json。无实质代码改动，仅例行页面重生成。
- 关键文件：public/ai-topics/daily.html、public/ai-topics/weekly.html、public/ai-topics/_index.json

## 2026-08-19（补记 08-17~08-18 积累变化）
- 阶段：网页开发
- 进展：新增「Agents 监控」功能——服务器智能体实时状态与日志监控页（agents/index.astro + [...slug].astro，Supabase 数据源，每 5 秒刷新，运行中=60 秒内有心跳）；配套零依赖 agent-log-uploader.mjs 日志增量上传器（tail 服务器 log → 批量写 Supabase agent_logs 表 + 心跳）+ supabase/schema.sql（agent_logs/agents 表 + RLS）与 api/supabase/index.js；新增 PasswordGate 密码门组件（密码 090909）保护页面。文件修改时间 08-17 19:34~08-18 12:48，因 8-17/8-18 追踪未执行而积累补记。
- 关键文件：src/pages/agents/index.astro、src/pages/agents/[...slug].astro、scripts/agent-log-uploader.mjs、supabase/schema.sql、src/components/PasswordGate.astro

## 2026-08-20（补记 08-19 23:14 例行部署）
- 阶段：网页开发
- 进展：无实质进展（例行页面重生成）——08-19 23:14 projects 日报/周报页面（daily.html / weekly.html / _index.json）由部署自动化例行重生成并刷新索引，23:15 项目内自动化部署记录 memory.md 更新；今日扫描补记，无代码改动。
- 关键文件：public/projects/daily.html、public/projects/weekly.html、public/projects/_index.json

## 2026-08-24
- 阶段：网页开发
- 进展：「AI 副业选题」页面例行更新——09:22 重新生成每日选题（2026-08-17）与下周内容日历 2026-W35（8月24日—30日）HTML 并刷新 _index.json。无实质代码改动，仅例行页面重生成。
- 关键文件：public/ai-topics/daily.html、public/ai-topics/weekly.html、public/ai-topics/_index.json

## 2026-08-25
- 阶段：网页开发
- 进展：① AI 选题页面（src/pages/ai-topics.astro）移除密码保护——删除 #gate 密码门与密码校验脚本，每日选题/每周计划/AI 热点新闻 3 个子入口均免密（projects/agents 等页面保留密码门），并在两个卡片下方新增通栏卡片「AI 热点新闻」链接到 CloudBase 托管的 ai_select 站点（https://ai2-d6ge8qoxj6157b724-1256053800.tcloudbaseapp.com），commit c6b9bb9 已推送触发 Vercel 部署；② 修复 api/supabase/index.js 与 api/feishu/index.js 在 Vercel rewrite 下丢失路径段问题（:path* 改从 req.query.path 恢复），commit 286c328 / fa303f1；③ 三个代理（feishu/supabase/academic）统一转发时强制覆盖 Content-Type 为 application/json，绕过公司 McAfee 拦截，commit 3665c15；④ ai-topics 每日/每周选题页 09:19 例行重生成。
- 关键文件：src/pages/ai-topics.astro、api/supabase/index.js、api/feishu/index.js、api/academic/index.js、public/ai-topics/daily.html
