// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { StorybookConfig } from '@storybook/react-vite';

const config: StorybookConfig = {
  stories: ['../src/**/*.mdx', '../src/**/*.stories.@(js|jsx|mjs|ts|tsx)'],
  addons: ['@storybook/addon-docs'],
  framework: '@storybook/react-vite',
  // The app's public files, so components that load /vortex_wordmark.svg find it, as in the app.
  staticDirs: ['../public'],
};
export default config;
