// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Re-export constants from utils for backward compat, plus any rendering-only constants.
export {
  ENCODING_STYLES,
  getEncodingStyle,
  DTYPE_COLORS,
  DTYPE_CATEGORIES,
  ROW_HEIGHT,
  MIN_LABEL_WIDTH,
  GROUP_SIZE,
} from './utils';

export const TREE_WIDTH = 260;
export const DEFAULT_SWIMLANE_MIN_WIDTH = 800;
export const DEFAULT_HEIGHT = 360;
