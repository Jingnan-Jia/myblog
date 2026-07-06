import {
  defineConfig,
  envField,
  fontProviders,
  svgoOptimizer,
} from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import { unified } from "@astrojs/markdown-remark";
import remarkToc from "remark-toc";
import remarkCollapse from "remark-collapse";
import rehypeCallouts from "rehype-callouts";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
} from "@shikijs/transformers";
import { transformerFileName } from "./src/utils/transformers/fileName";
import config from "./astro-paper.config";

// Convert relative (non-root, non-absolute) markdown image references into
// plain HTML <img> tags so Astro's asset pipeline does not fail to resolve
// missing local images at build time. Real images can be dropped into
// /public later without touching the markdown.
function remarkPassthroughLocalImages() {
  const walk = (node: any) => {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    if (node.type === "image") {
      const src: string = node.url ?? "";
      const isExternal = /^(https?:)?\/\//i.test(src);
      const isRoot = src.startsWith("/") || src.startsWith("@/") || src.startsWith("assets/");
      if (!isExternal && !isRoot) {
        const alt = (node.alt ?? "").replace(/"/g, "&quot;");
        const title = node.title ? ` title="${(node.title ?? "").replace(/"/g, "&quot;")}"` : "";
        node.type = "html";
        node.value = `<img src="${src}" alt="${alt}"${title}>`;
        delete node.url;
        delete node.alt;
        delete node.title;
        return;
      }
    }
    for (const key of Object.keys(node)) {
      if (key === "type" || key === "value") continue;
      const child = (node as any)[key];
      if (child && typeof child === "object") walk(child);
    }
  };
  return (tree: any) => walk(tree);
}

export default defineConfig({
  site: config.site.url,
  integrations: [
    mdx(),
    sitemap({
      filter: page =>
        config.features?.showArchives !== false || !page.endsWith("/archives/"),
    }),
  ],
  i18n: {
    locales: ["en", "zh-CN"],
    defaultLocale: "zh-CN",
    routing: {
      prefixDefaultLocale: false,
    },
  },
  markdown: {
    processor: unified({
      remarkPlugins: [
        remarkMath,
        remarkPassthroughLocalImages,
        remarkToc,
        [remarkCollapse, { test: "Table of contents" }],
      ],
      rehypePlugins: [rehypeCallouts, rehypeKatex],
    }),
    shikiConfig: {
      themes: { light: "min-light", dark: "night-owl" },
      defaultColor: false,
      wrap: false,
      transformers: [
        transformerFileName({ style: "v2", hideDot: false }),
        transformerNotationHighlight(),
        transformerNotationWordHighlight(),
        transformerNotationDiff({ matchAlgorithm: "v3" }),
      ],
    },
  },
  vite: {
    plugins: [tailwindcss()],
  },
  fonts: [
    {
      name: "Google Sans Code",
      cssVariable: "--font-google-sans-code",
      provider: fontProviders.google(),
      fallbacks: ["monospace"],
      weights: [300, 400, 500, 600, 700],
      styles: ["normal", "italic"],
      formats: ["woff", "ttf"],
    },
  ],
  env: {
    schema: {
      PUBLIC_GOOGLE_SITE_VERIFICATION: envField.string({
        access: "public",
        context: "client",
        optional: true,
      }),
    },
  },
  experimental: {
    svgOptimizer: svgoOptimizer(),
  },
});
