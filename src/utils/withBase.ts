/**
 * Prepends the site base path to a given URL.
 * Handles trailing/leading slash edge cases.
 *
 * Usage: import { withBase } from "@/utils/withBase";
 *        href={withBase("/posts")}
 */
export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, ""); // strip trailing slash
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${base}${p}`;
}
