// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Main component
export { LayoutSwimlane, default } from './LayoutSwimlane';
export type { LayoutSwimlaneProps } from './LayoutSwimlane';

// Types
export type {
  LayoutTreeNode,
  LayoutChildKind,
  SegmentMapEntry,
  FileStructureInfo,
  FlattenedRow,
  DisplayKind,
  Split,
  DtypeCategory,
} from './types';

// Sub-components
export { TreeRow } from './TreeRow';
export { TreeSearch } from './TreeSearch';
export { SwimlaneBar } from './SwimlaneBar';
export { AxisBar } from './AxisBar';
export { Tooltip } from './Tooltip';
export { DtypeLegend } from './DtypeLegend';
export { SplitRegion } from './SplitRegion';

// Utilities
export {
  getDtypeCategory,
  rangesOverlap,
  createSplits,
  formatBytes,
  formatRowRange,
  formatRowCount,
  getNodeDisplayName,
  getNodeRowRange,
  getEncodingStyle,
  hasExpandableChildren,
  flattenTree,
  filterTreeBySearch,
  findNodeById,
  findPathToNode,
  collectSubtreeIds,
  collectSubtreeSegments,
  ENCODING_STYLES,
  DTYPE_COLORS,
  DTYPE_CATEGORIES,
} from './utils';
