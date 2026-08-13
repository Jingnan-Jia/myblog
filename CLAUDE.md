# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

| Command | Description |
|---|---|
| `pnpm dev` | Start dev server at `localhost:4321` |
| `pnpm build` | Type-check, build, generate Pagefind search index |
| `pnpm preview` | Preview production build locally |
| `pnpm sync` | Generate TypeScript types for Astro modules |
| `pnpm lint` | Lint with ESLint |
| `pnpm format` | Format with Prettier |
| `pnpm format:check` | Check formatting without writing |

No test framework is configured.

## Architecture

This is a personal blog (jiajingnan.cn) built on **AstroPaper v6**, an Astro static site generator theme. It uses Tailwind CSS v4 for styling and Pagefind for static search.

### Two-Tier Configuration

1. **User config**: `astro-paper.config.ts` — defines site metadata, features, social links, pagination settings. Uses `defineAstroPaperConfig()` helper.
2. **Resolved config**: `src/config.ts` — imports user config, applies defaults for every field, exposes `ResolvedAstroPaperConfig` (no optional fields).

### Content Collections (defined in `src/content.config.ts`)

- **`posts`** — Markdown/MDX in `src/content/posts/`. Subdirectory names become part of post URLs. Files starting with `_` are excluded from all collections (useful for drafts/organizational content). Frontmatter: `title`, `pubDatetime`, `description`, `tags`, `featured`, `draft`, `ogImage`, `canonicalURL`, `hideEditPost`.
- **`pages`** — Markdown in `src/content/pages/` for standalone pages.
- **`publications`** — Markdown in `src/content/publications/`. Frontmatter: `title`, `year`, `venue`, `excerpt`, `citation`, `url`.

### Post Visibility

`src/utils/postFilter.ts` controls visibility: drafts are always excluded; in production, scheduled posts are hidden until their `pubDatetime` minus a configurable margin (default 15 min). In dev, all non-draft posts are shown regardless of date.

### i18n

Built on Astro's i18n routing (`en`, `zh-CN`). Default locale is `zh-CN` with no URL prefix. Custom `useTranslations()` system in `src/i18n/` provides UI string translations with implementations in `lang/en.ts` and `lang/zh-CN.ts`. All translatable strings must conform to the `UIStrings` interface in `types.ts`.

### Dynamic OG Images

When `features.dynamicOgImage` is enabled and a post lacks a static `ogImage`, `posts/[...slug]/index.png.ts` generates one at build time using Satori + Sharp.

### Static Search

Pagefind index is generated at build time (`pagefind --site dist`) and stored in `public/pagefind/`. The search page lazily loads `@pagefind/default-ui`.

### Markdown Processing

Remark plugins: math, TOC, collapsible TOC. Rehype plugins: callouts, KaTeX. Syntax highlighting via Shiki (min-light / night-owl themes) with transformers for diff, highlight, word-highlight, and file names.

### Client Interactivity

Client-side scripts use Astro custom events (`astro:page-load`, `astro:after-swap`, `astro:before-swap`). The theme toggle script is in `src/scripts/theme.ts`.
