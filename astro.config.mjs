import { readdirSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import mdx from "@astrojs/mdx";
import pagefind from "astro-pagefind";
import tailwindcss from "@tailwindcss/vite";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const site = "https://jinnpan.com";

// Pages under public/sources are copied as is, so the sitemap integration does not see them as routes.
function sourcePages(dir = "public/sources") {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return sourcePages(path);
    if (!entry.name.endsWith(".html")) return [];
    const url = "/" + relative("public", path).split(sep).join("/");
    return [new URL(url.replace(/index\.html$/, ""), site).href];
  });
}

export default defineConfig({
  site,
  integrations: [sitemap({ customPages: sourcePages() }), mdx(), pagefind()],
  vite: {
    plugins: [tailwindcss()],
  },
  markdown: {
    shikiConfig: {
      theme: "css-variables",
    },
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  i18n: {
    defaultLocale: "en",
    locales: ["zh", "en"],
    routing: {
      prefixDefaultLocale: true,
      redirectToDefaultLocale: false,
    },
  },
});
