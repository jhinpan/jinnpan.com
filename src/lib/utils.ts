import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(date: Date) {
  return Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

export function readingTime(html: string) {
  const textOnly = html.replace(/<[^>]+>/g, "");
  const wordCount = textOnly.split(/\s+/).length;
  const readingTimeMinutes = (wordCount / 200 + 1).toFixed();
  return `${readingTimeMinutes} min read`;
}

export type Lang = "zh" | "en";

export function getPostLang(id: string): Lang {
  return id.startsWith("en/") ? "en" : "zh";
}

export function getPostSlug(id: string): string {
  return id.replace(/^(zh|en)\//, "");
}

export function getPostUrl(entry: { collection: string; id: string }): string {
  const lang = getPostLang(entry.id);
  const slug = getPostSlug(entry.id);
  return `/${lang}/${entry.collection}/${slug}`;
}
