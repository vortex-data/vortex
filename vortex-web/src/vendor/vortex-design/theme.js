// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// A copy of the Vortex design system's components/lib/theme.js. Edit the system, then copy it again.

// The theme choice every surface shares (DECISIONS 37, 43): light, dark or system. The default is Dark.
// It lives in a vx-theme cookie on vortex.dev, so a choice made on the blog holds on docs, bench and
// the Explorer too, and in localStorage["vx-theme"] as a fallback for hosts outside vortex.dev. It
// lands on <html data-theme>. The choice itself also lands on <html data-theme-choice>, so the switch
// can show it pressed before its script runs.
//
// A site whose other pages are Dark only passes `themeable`, a list of path prefixes: "/blog"
// covers /blog and every page under it. Pages outside the list stay Dark whatever the choice.
// It is a list and not a function because the bootstrap carries it into an inline script as data.
const KEY = "vx-theme";
const CHOICES = ["light", "dark", "system"];
const PLAIN_PATH = /^\/[\w.~\/-]*$/;
const DEFAULT = "dark";
const YEAR = 31536000;
const COOKIE = /(?:^|;\s*)vx-theme=([^;]*)/;

// "/blog/" and "/blog" are one prefix. "/" stays "/".
const trim = (prefix) => (prefix.length > 1 ? prefix.replace(/\/+$/, "") : prefix);

/** Whether `path` takes the reader's theme. With no list, every path does. */
export function isThemeable(path, themeable) {
  if (!themeable) return true;
  return themeable.map(trim).some((p) => path === p || path.startsWith(p === "/" ? "/" : `${p}/`));
}

/**
 * The no-flash snippet for a site where only `themeable` paths follow the choice (every path when
 * it is left out). Paste it into an inline classic <script> in <head>, before the stylesheets, so
 * the theme lands on <html> before first paint. A module script runs too late for this. It repeats
 * readChoice, isThemeable and resolveTheme in plain ES5, because it cannot import them. The paths
 * go in as JSON, and only plain paths are allowed, so no value becomes code.
 */
export function themeBootstrap({ themeable } = {}) {
  let list = "null";
  if (themeable) {
    for (const p of themeable) if (typeof p !== "string" || !PLAIN_PATH.test(p)) throw new Error(`themeable takes a plain path such as "/blog", not ${JSON.stringify(p)}`);
    list = JSON.stringify(themeable.map(trim));
  }
  return (
    `(function(){var r=document.documentElement,c="${DEFAULT}",v=null,t=${list},ok=!t;` +
    'if(t){var p=location.pathname;for(var i=0;i<t.length;i++)if(p===t[i]||p.indexOf(t[i]==="/"?"/":t[i]+"/")===0)ok=true;}' +
    'try{var k=/(?:^|;\\s*)vx-theme=(light|dark|system)(?:;|$)/.exec(document.cookie);if(k)v=k[1];}catch(e){}' +
    'if(!v)try{var s=localStorage.getItem("vx-theme");if(s==="light"||s==="dark"||s==="system")v=s;}catch(e){}' +
    'if(v)c=v;' +
    'r.dataset.themeChoice=c;var m=c;if(m==="system")m=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";' +
    'r.dataset.theme=ok?m:"dark";})();'
  );
}

/** The snippet for a site where every page follows the choice. */
export const THEME_BOOTSTRAP = themeBootstrap();

/** The choice: the cookie's when it holds one, else this origin's storage, else Dark. */
export function readChoice(storage, cookie = "") {
  const c = COOKIE.exec(cookie ?? "")?.[1];
  if (CHOICES.includes(c)) return c;
  try {
    const v = storage?.getItem(KEY);
    return CHOICES.includes(v) ? v : DEFAULT;
  } catch {
    return DEFAULT;
  }
}

/**
 * The Set-Cookie text for `choice`: on vortex.dev and every subdomain when `hostname` is one of
 * them, else on the host only (localhost, a preview deploy). Null for a value that is no choice.
 */
export function themeCookie(choice, hostname, secure) {
  if (!CHOICES.includes(choice)) return null;
  const shared = hostname === "vortex.dev" || hostname?.endsWith(".vortex.dev");
  return `${KEY}=${choice}; path=/; max-age=${YEAR}; samesite=lax${shared ? "; domain=vortex.dev" : ""}${secure ? "; secure" : ""}`;
}

const page = () => (typeof document === "undefined" ? null : document);

/** Writes the choice to storage and to the cookie. `jar` is document by default. */
export function saveChoice(storage, choice, { jar = page(), hostname = globalThis.location?.hostname, secure = globalThis.location?.protocol === "https:" } = {}) {
  try {
    storage?.setItem(KEY, choice);
  } catch {
    // Blocked storage only costs this origin's copy of the choice.
  }
  const text = themeCookie(choice, hostname, secure);
  try {
    if (jar && text) jar.cookie = text;
  } catch {
    // A blocked cookie only costs the other surfaces' copy.
  }
}

export function resolveTheme(choice, prefersDark) {
  if (choice === "system") return prefersDark ? "dark" : "light";
  return choice === "light" ? "light" : "dark";
}

/** The theme a page shows: the choice on a themeable path, else Dark. */
export function themeFor(choice, prefersDark, { path = "/", themeable } = {}) {
  return isThemeable(path, themeable) ? resolveTheme(choice, prefersDark) : "dark";
}

const store = () => {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
};
const dark = () => matchMedia("(prefers-color-scheme: dark)");
const cookies = () => {
  try {
    return document.cookie;
  } catch {
    return "";
  }
};
// Fired on document when a switch changes the choice, so every switch on the page shows it.
const CHANGE = "vx-theme:change";

// Sets data-theme and data-theme-choice from the stored choice. `themeable` and `path` (the
// current page's by default) keep a page outside the list Dark.
export function applyTheme(choice = readChoice(store(), cookies()), { themeable, path = location.pathname } = {}) {
  const root = document.documentElement;
  root.dataset.themeChoice = choice;
  root.dataset.theme = themeFor(choice, dark().matches, { path, themeable });
  return choice;
}

// Wires a .vx-theme-switch: buttons carry data-choice and aria-pressed.
// With { apply: false } it shows the current choice (the bootstrap's data-theme-choice, else the
// theme) without changing it on load. `themeable` is the same list the bootstrap took. Every switch
// on the page follows a change made in any of them (the header and its Menu panel both hold one).
// Returns a function that removes the listeners.
export function mountThemeSwitch(el, { apply = true, themeable } = {}) {
  const buttons = [...el.querySelectorAll("button[data-choice]")];
  const root = document.documentElement.dataset;
  let choice = apply ? applyTheme(undefined, { themeable }) : CHOICES.includes(root.themeChoice) ? root.themeChoice : DEFAULT;
  const press = () => buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.choice === choice)));
  const onClick = (e) => {
    choice = e.currentTarget.dataset.choice;
    saveChoice(store(), choice);
    applyTheme(choice, { themeable });
    document.dispatchEvent(new CustomEvent(CHANGE, { detail: choice }));
  };
  const onChange = (e) => {
    if (CHOICES.includes(e.detail)) choice = e.detail;
    press();
  };
  const onOs = () => choice === "system" && applyTheme(choice, { themeable });
  const media = dark();
  for (const b of buttons) b.addEventListener("click", onClick);
  document.addEventListener(CHANGE, onChange);
  media.addEventListener("change", onOs);
  press();
  return () => {
    for (const b of buttons) b.removeEventListener("click", onClick);
    document.removeEventListener(CHANGE, onChange);
    media.removeEventListener("change", onOs);
  };
}
