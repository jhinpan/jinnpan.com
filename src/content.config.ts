import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const blog = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/blog" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    draft: z.boolean().optional(),
    tags: z.array(z.string()).optional(),
    category: z.string().optional(),
    lang: z.enum(["zh", "en"]).optional(),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/projects" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    draft: z.boolean().optional(),
    demoURL: z.string().optional(),
    repoURL: z.string().optional(),
    // Work page: "now" entries are current work, listed by `order`.
    group: z.enum(["now", "earlier"]).default("earlier"),
    order: z.number().optional(),
    role: z.string().optional(),
    // owner/name on GitHub; the Work page shows its merged pull request count.
    repo: z.string().optional(),
    // A Library page about this project.
    reading: z.string().optional(),
  }),
});

export const collections = { blog, projects };
