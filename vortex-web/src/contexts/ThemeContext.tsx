// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { ThemeContext } from './ThemeContextCore';

export type ThemeChoice = 'light' | 'dark' | 'system';

export interface ThemeContextValue {
  theme: ThemeChoice;
  setTheme: (theme: ThemeChoice) => void;
}

const STORAGE_KEY = 'vortex-theme';

/**
 * The design system's tokens read the choice from `data-theme` on the root: `dark`, `light`, or
 * `system`, which follows the operating system through prefers-color-scheme. `public/theme.js`
 * sets it before the first paint; this keeps it in step with the switch.
 */
function applyTheme(choice: ThemeChoice) {
  document.documentElement.dataset.theme = choice;
}

function getStoredTheme(): ThemeChoice {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark' || stored === 'system') return stored;
  } catch {
    // Storage can be blocked; fall back to the default.
  }
  return 'dark';
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeChoice>(getStoredTheme);

  const setTheme = useCallback((choice: ThemeChoice) => {
    try {
      localStorage.setItem(STORAGE_KEY, choice);
    } catch {
      // The choice then lasts for this page only.
    }
    setThemeState(choice);
  }, []);

  useEffect(() => applyTheme(theme), [theme]);

  return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>;
}
