import type { Metadata, Site, Socials } from "@types";

export const SITE: Site = {
  TITLE: "Jinn's Hub",
  DESCRIPTION: "ML Systems / LLM Inference / RL Infrastructure",
  EMAIL: "jpan236@wisc.edu",
  NUM_POSTS_ON_HOMEPAGE: 5,
  NUM_PROJECTS_ON_HOMEPAGE: 5,
};

export const HOME: Metadata = {
  TITLE: "Home",
  DESCRIPTION: "ML Systems / LLM Inference / RL Infrastructure",
};

export const BLOG: Metadata = {
  TITLE: "Blog",
  DESCRIPTION: "Technical writing on MLSys, LLM inference, and kernel optimization.",
};

export const PROJECTS: Metadata = {
  TITLE: "Projects",
  DESCRIPTION: "A collection of my projects with links to repositories and demos.",
};

export const ABOUT: Metadata = {
  TITLE: "About",
  DESCRIPTION: "About Jin Pan — ML Systems researcher and engineer.",
};

export const SOCIALS: Socials = [
  {
    NAME: "GitHub",
    HREF: "https://github.com/jhinpan",
  },
  {
    NAME: "LinkedIn",
    HREF: "https://www.linkedin.com/in/jinnpan/",
  },
];
