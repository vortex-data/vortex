// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Main component
export { LayoutSwimlane, default } from './LayoutSwimlane';
export type { LayoutSwimlaneProps } from './LayoutSwimlane';

// Types
export type {
  LayoutNode,
  StructLayout,
  ChunkedLayout,
  ZonemapLayout,
  DictLayout,
  FlatLayout,
  ChunkNode,
  ZoneNode,
  Split,
  DtypeCategory,
  LayoutType,
  FlatMeta,
  ZoneMeta,
} from './types';

// Utilities (for advanced usage)
export {
  getDtypeCategory,
  rangesOverlap,
  createSplits,
  formatBytes,
  formatRowRange,
  formatRowCount,
  LAYOUT_STYLES,
  DTYPE_COLORS,
  DTYPE_CATEGORIES,
} from './utils';
