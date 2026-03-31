// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Unified layout tree type mirroring the Rust Layout trait.
// Each node represents a layout in the Vortex file format.

export interface LayoutTreeNode {
  /** Path-based ID, e.g. "root.customer.id.[0]" */
  id: string;
  /** Encoding name, e.g. "vortex.flat", "vortex.chunked" */
  encoding: string;
  /** DType string, e.g. "i64", "utf8", "{name=utf8, age=i32}" */
  dtype: string;
  /** Number of rows in this layout */
  rowCount: number;
  /** Absolute row offset in the file */
  rowOffset: number;
  /** Size of metadata for this layout in bytes */
  metadataBytes: number;
  /** Segment IDs referenced by this layout */
  segmentIds: number[];
  /** Relationship of this node to its parent */
  childType: LayoutChildKind;
  /** Child layout nodes */
  children: LayoutTreeNode[];
  /** For flat layouts: the array encoding tree inside this layout */
  arrayEncodingTree?: ArrayEncodingNode;
  /** True if this node represents an array encoding node (not a layout node) */
  isArrayNode?: boolean;
  /** Buffer byte lengths for array nodes */
  bufferLengths?: number[];
  /** Buffer names for array nodes */
  bufferNames?: string[];
}

export interface ArrayEncodingNode {
  encoding: string;
  dtype: string;
  metadataBytes: number;
  numBuffers: number;
  bufferLengths: number[];
  bufferNames: string[];
  children: ArrayEncodingNode[];
  childNames: string[];
}

export type LayoutChildKind =
  | { kind: 'root' }
  | { kind: 'field'; fieldName: string }
  | { kind: 'chunk'; chunkIndex: number; rowOffset: number }
  | { kind: 'transparent'; name: string }
  | { kind: 'auxiliary'; name: string };

export interface SegmentMapEntry {
  index: number;
  byteOffset: number;
  byteLength: number;
  alignment: number;
  column: string | null;
  /** Node ID path in the layout tree */
  layoutPath: string;
}

export interface FileStructureInfo {
  fileSize: number;
  version: number;
  postscriptSize: number;
  totalDataBytes: number;
  totalMetadataBytes: number;
}

// Rendering types (internal to swimlane)

export type DisplayKind = 'normal' | 'group' | 'hiddenIndicator';

export interface FlattenedRow {
  node: LayoutTreeNode;
  depth: number;
  displayKind: DisplayKind;
  groupedChildren?: LayoutTreeNode[];
  rowRange: [number, number];
}

// Retained from original types
export type DtypeCategory =
  | 'bool'
  | 'int'
  | 'float'
  | 'utf8'
  | 'datetime'
  | 'struct'
  | 'list'
  | 'other';

export interface Split {
  id: string;
  rowRange: [number, number];
}
