// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useTheme, type ThemeChoice } from '../contexts/ThemeContextCore';
import { Icon, type IconName } from './Icon';

const CHOICES: { choice: ThemeChoice; label: string; icon: IconName }[] = [
  { choice: 'light', label: 'Light', icon: 'light_mode' },
  { choice: 'dark', label: 'Dark', icon: 'dark_mode' },
  { choice: 'system', label: 'System', icon: 'contrast' },
];

/** The design system's theme switch: three buttons, the current one pressed. */
export function ThemePicker() {
  const { theme, setTheme } = useTheme();
  return (
    <fieldset className="vx-theme-switch" aria-label="Theme">
      {CHOICES.map(({ choice, label, icon }) => (
        <button
          type="button"
          key={choice}
          aria-pressed={theme === choice}
          aria-label={label}
          title={label}
          onClick={() => setTheme(choice)}
        >
          <Icon name={icon} />
        </button>
      ))}
    </fieldset>
  );
}
