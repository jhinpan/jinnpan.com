import type { Metadata, Site, Socials } from "@types";

export const SITE: Site = {
  TITLE: "Jin Pan",
  DESCRIPTION: "ML Systems / GPU Kernels / LLM Serving",
  EMAIL: "panjin1221@gmail.com",
};

export const HOME: Metadata = {
  TITLE: "Jin Pan · ML Systems",
  DESCRIPTION:
    "Jin Pan works on ML systems at AMD: GPU kernels in FlyDSL, LLM serving in SGLang, and measurements published with their raw data.",
};

export const BLOG: Metadata = {
  TITLE: "Blog",
  DESCRIPTION: "Technical writing on MLSys, LLM inference, and kernel optimization.",
};

export const PROJECTS: Metadata = {
  TITLE: "Work",
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
