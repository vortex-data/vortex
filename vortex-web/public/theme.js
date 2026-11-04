// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Sets the reader's theme before the first paint (the design system's ThemeSwitch): the vx-theme
// cookie that every vortex.dev surface shares, else this origin's storage, else Dark. A file,
// not an inline script, because the Content-Security-Policy allows scripts from this origin only.
// A choice saved under the Explorer's old key, 'vortex-theme', moves to 'vx-theme' first.
try {
  const old = localStorage.getItem('vortex-theme');
  if (old !== null) {
    if (localStorage.getItem('vx-theme') === null) localStorage.setItem('vx-theme', old);
    localStorage.removeItem('vortex-theme');
  }
} catch {
  // Storage can be blocked; the bootstrap then reads the cookie alone.
}
// The design system's THEME_BOOTSTRAP (components/lib/theme.js).
(function(){var r=document.documentElement,c="dark",v=null,t=null,ok=!t;if(t){var p=location.pathname;for(var i=0;i<t.length;i++)if(p===t[i]||p.indexOf(t[i]==="/"?"/":t[i]+"/")===0)ok=true;}try{var k=/(?:^|;\s*)vx-theme=(light|dark|system)(?:;|$)/.exec(document.cookie);if(k)v=k[1];}catch(e){}if(!v)try{var s=localStorage.getItem("vx-theme");if(s==="light"||s==="dark"||s==="system")v=s;}catch(e){}if(v)c=v;r.dataset.themeChoice=c;var m=c;if(m==="system")m=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";r.dataset.theme=ok?m:"dark";})();
