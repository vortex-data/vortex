// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// The shared header's scripts (the design system's Header and ThemeSwitch): every theme switch on
// the page, and the Menu, which closes on Escape (focus goes back to its button) and on a click
// outside it. Without scripts the Menu still opens and closes, and the theme follows the OS.
import { mountThemeSwitch } from "./vortex-theme.js";

for (const el of document.querySelectorAll(".vx-theme-switch")) mountThemeSwitch(el);

const menu = document.querySelector(".vx-header-menu");
if (menu) {
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !menu.open) return;
    menu.open = false;
    menu.querySelector("summary")?.focus();
  });
  document.addEventListener("pointerdown", (e) => {
    if (menu.open && !menu.contains(e.target)) menu.open = false;
  });
}
