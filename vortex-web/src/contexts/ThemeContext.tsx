// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { applyTheme, readChoice, saveChoice } from '../vendor/vortex-design/theme.js';
import { ThemeContext } from './ThemeContextCore';

export type ThemeChoice = 'light' | 'dark' | 'system';

export interface ThemeContextValue {
  theme: ThemeChoice;
  setTheme: (theme: ThemeChoice) => void;
}

function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function cookie(): string {
  try {
    return document.cookie;
  } catch {
    return '';
  }
}

/**
 * The theme every Vortex surface shares (the design system's ThemeSwitch): the vx-theme cookie on
 * vortex.dev, else this origin's storage, else Dark. `public/theme.js` sets it before the first
 * paint; this keeps it in step with the switches, and follows the OS while the choice is System.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeChoice>(
    () => readChoice(storage(), cookie()) as ThemeChoice,
  );

  const setTheme = useCallback((choice: ThemeChoice) => {
    saveChoice(storage(), choice);
    setThemeState(choice);
  }, []);

  useEffect(() => {
    applyTheme(theme);
    if (theme !== 'system') return;
    const media = matchMedia('(prefers-color-scheme: dark)');
    const follow = () => applyTheme('system');
    media.addEventListener('change', follow);
    return () => media.removeEventListener('change', follow);
  }, [theme]);

  return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>;
}
