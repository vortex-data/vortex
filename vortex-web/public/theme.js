// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Sets the saved theme before the first paint, so a light theme never flashes dark. A file, not an
// inline script, because the Content-Security-Policy allows scripts from this origin only.
try {
  const choice = localStorage.getItem('vortex-theme');
  if (choice === 'light' || choice === 'dark' || choice === 'system') {
    document.documentElement.dataset.theme = choice;
  }
} catch {
  // Storage can be blocked; the page then stays on the default, dark.
}
