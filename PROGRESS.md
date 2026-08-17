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
