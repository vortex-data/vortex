// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Sets the reader's theme before the first paint (the design system's ThemeSwitch): the vx-theme
// cookie that every vortex.dev surface shares, else this origin's storage, else Dark. A file,
// not an inline script, because the Content-Security-Policy allows scripts from this origin only.
// The design system's themeBootstrap (components/lib/theme.js), with the Explorer's old key,
// 'vortex-theme', as a legacy key: its choice moves to 'vx-theme'.
(function(){var r=document.documentElement,c="dark",v=null,f=false,t=null,L=["vortex-theme"],ok=!t;function q(x){return x==="light"||x==="dark"||x==="system"}if(t){var p=location.pathname;for(var i=0;i<t.length;i++)if(p===t[i]||p.indexOf(t[i]==="/"?"/":t[i]+"/")===0)ok=true;}try{var k=/(?:^|;\s*)vx-theme=(light|dark|system)(?:;|$)/.exec(document.cookie);if(k){v=k[1];f=true;}}catch(e){}if(!v)try{var s=localStorage.getItem("vx-theme");if(q(s))v=s;}catch(e){}if(L)for(var j=0;j<L.length;j++)try{var o=localStorage.getItem(L[j]);if(o!==null){if(!v&&q(o)){v=o;try{localStorage.setItem("vx-theme",o)}catch(e){}}localStorage.removeItem(L[j]);}}catch(e){}if(v&&!f)try{var h=location.hostname;document.cookie="vx-theme="+v+"; path=/; max-age=31536000; samesite=lax"+(h==="vortex.dev"||/\.vortex\.dev$/.test(h)?"; domain=vortex.dev":"")+(location.protocol==="https:"?"; secure":"");}catch(e){}if(v)c=v;r.dataset.themeChoice=c;var m=c;if(m==="system")m=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";r.dataset.theme=ok?m:"dark";})();
