// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// The design system's THEME_BOOTSTRAP (components/lib/theme.js), as a file, because the docs' CSP
// allows only scripts from their own origin. Load it in <head>, before the stylesheets.
(function(){var r=document.documentElement,c="dark",v=null,t=null,ok=!t;if(t){var p=location.pathname;for(var i=0;i<t.length;i++)if(p===t[i]||p.indexOf(t[i]==="/"?"/":t[i]+"/")===0)ok=true;}try{var k=/(?:^|;\s*)vx-theme=(light|dark|system)(?:;|$)/.exec(document.cookie);if(k)v=k[1];}catch(e){}if(!v)try{var s=localStorage.getItem("vx-theme");if(s==="light"||s==="dark"||s==="system")v=s;}catch(e){}if(v)c=v;r.dataset.themeChoice=c;var m=c;if(m==="system")m=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";r.dataset.theme=ok?m:"dark";})();
