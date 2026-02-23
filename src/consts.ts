import type { Metadata, Site, Socials } from "@types";

export const SITE: Site = {
  TITLE: "Jhin Pan",
  DESCRIPTION: "RL Infra / LLM Inference / Kernel Automation",
  EMAIL: "jhinpan@outlook.com",
  NUM_POSTS_ON_HOMEPAGE: 5,
  NUM_PROJECTS_ON_HOMEPAGE: 3,
};

export const HOME: Metadata = {
  TITLE: "Home",
  DESCRIPTION: "RL Infra / LLM Inference / Kernel Automation",
};

export const BLOG: Metadata = {
  TITLE: "Blog",
  DESCRIPTION: "Technical writing on MLSys, LLM inference, and kernel optimization.",
};

export const PROJECTS: Metadata = {
  TITLE: "Projects",
  DESCRIPTION: "A collection of my projects with links to repositories and demos.",
};

export const SOCIALS: Socials = [
  {
    NAME: "GitHub",
    HREF: "https://github.com/jhinpan",
  },
  {
    NAME: "LinkedIn",
    HREF: "https://linkedin.com/in/jhinpan",
  },
];
