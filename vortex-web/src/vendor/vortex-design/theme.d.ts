// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// A copy of the Vortex design system's components/lib/theme.d.ts. Edit the system, then copy it again.
// Types for theme.js, shipped beside it.

export type ThemeChoice = "light" | "dark" | "system";
export type Theme = "light" | "dark";

/** Path prefixes whose pages follow the reader's choice: "/blog" covers /blog and /blog/…. */
export type Themeable = readonly string[];

export function isThemeable(path: string, themeable?: Themeable): boolean;
/** `legacy` names older storage keys whose choice moves to vx-theme (bench's "bench-theme"). */
export function themeBootstrap(options?: { themeable?: Themeable; legacy?: readonly string[] }): string;
export const THEME_BOOTSTRAP: string;
/** The cookie's choice when it holds one, else storage's, else "system". */
export function readChoice(storage: Pick<Storage, "getItem"> | null | undefined, cookie?: string): ThemeChoice;
/** The cookie text: on vortex.dev and its subdomains there, on the host only elsewhere. */
export function themeCookie(choice: string, hostname: string | undefined, secure: boolean): string | null;
export function saveChoice(
  storage: Pick<Storage, "setItem"> | null | undefined,
  choice: ThemeChoice,
  options?: { jar?: { cookie: string } | null; hostname?: string; secure?: boolean },
): void;
export function resolveTheme(choice: string, prefersDark: boolean): Theme;
export function themeFor(choice: string, prefersDark: boolean, options?: { path?: string; themeable?: Themeable }): Theme;
export function applyTheme(choice?: ThemeChoice, options?: { path?: string; themeable?: Themeable }): ThemeChoice;
export function mountThemeSwitch(el: Element, options?: { apply?: boolean; themeable?: Themeable }): () => void;
