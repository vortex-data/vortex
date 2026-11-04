// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { createElement } from 'react';
import type { Preview } from '@storybook/react-vite';
import { ThemeProvider } from '../src/contexts/ThemeContext';
import '../src/index.css';

const preview: Preview = {
  globalTypes: {
    theme: {
      description: 'Color theme',
      toolbar: {
        title: 'Theme',
        icon: 'paintbrush',
        items: [
          { value: 'light', title: 'Light', icon: 'sun' },
          { value: 'dark', title: 'Dark', icon: 'moon' },
          { value: 'system', title: 'System', icon: 'mirror' },
        ],
        dynamicTitle: true,
      },
    },
  },
  initialGlobals: {
    theme: 'dark',
  },
  decorators: [
    (Story, context) => {
      // Wrap every story in the real ThemeProvider so theme-aware components
      // (e.g. FileHeader/ThemePicker, which call useTheme) have their context.
      // Seed it from the toolbar's theme global; `key` remounts the provider
      // when the toolbar theme changes so the switch takes effect.
      const theme = (context.globals.theme as string) || 'dark';
      try {
        localStorage.setItem('vx-theme', theme);
      } catch {
        // Storage can be blocked; the provider then starts from its default.
      }
      return createElement(ThemeProvider, { key: theme }, Story());
    },
  ],
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
};

export default preview;
