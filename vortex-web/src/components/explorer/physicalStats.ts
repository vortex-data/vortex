// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode, SegmentMapEntry } from '../swimlane/types';
import { collectSubtreeSegments } from '../swimlane/utils';

/**
 * Physical (on-disk) properties of a layout or array-encoding node, aggregated
 * over its subtree. These describe how the logical data is laid out as bytes in
 * the file — the focus of the treemap explorer.
 */
export interface PhysicalStats {
  /** Bytes of actual encoded data (segment bytes for layouts, buffer bytes for array nodes). */
  dataBytes: number;
  /** Bytes of metadata summed over the subtree. */
  metadataBytes: number;
  /** dataBytes + metadataBytes. */
  totalBytes: number;
  /** Fraction of the whole file occupied by this subtree (0..1). */
  fractionOfFile: number;
  /** Average encoded bytes per logical row, or null when the node spans no rows. */
  bytesPerRow: number | null;
  /** Logical rows spanned by this node. */
  rowCount: number;
  /** Number of file segments reachable from this subtree (layout nodes). */
  segmentCount: number;
  /** Number of buffers in this array node (array nodes only). */
  bufferCount: number;
}

/** Total buffer bytes for an array-encoding node subtree. */
export function arraySubtreeBytes(node: LayoutTreeNode): number {
  const own = (node.bufferLengths ?? []).reduce((sum, b) => sum + b, 0);
  const childBytes = node.children
    .filter((c) => c.isArrayNode)
    .reduce((sum, c) => sum + arraySubtreeBytes(c), 0);
  return own + childBytes;
}

/** Sum of metadata bytes across an entire subtree. */
function subtreeMetadataBytes(node: LayoutTreeNode): number {
  return node.children.reduce((sum, c) => sum + subtreeMetadataBytes(c), node.metadataBytes);
}

/** Count buffers across an array-node subtree. */
function arraySubtreeBufferCount(node: LayoutTreeNode): number {
  const own = (node.bufferLengths ?? []).length;
  return node.children
    .filter((c) => c.isArrayNode)
    .reduce((sum, c) => sum + arraySubtreeBufferCount(c), own);
}

/**
 * Compute the physical statistics for a node, aggregated over its subtree.
 *
 * @param node the layout or array node to describe
 * @param segmentMap segment index → entry, used to resolve layout byte sizes
 * @param fileSize total file size in bytes, used for the percentage-of-file metric
 */
export function nodePhysicalStats(
  node: LayoutTreeNode,
  segmentMap: Map<number, SegmentMapEntry>,
  fileSize: number,
): PhysicalStats {
  const isArray = node.isArrayNode ?? false;

  let dataBytes: number;
  let segmentCount: number;
  let bufferCount: number;

  if (isArray) {
    dataBytes = arraySubtreeBytes(node);
    bufferCount = arraySubtreeBufferCount(node);
    segmentCount = 0;
  } else {
    const ids = new Set(collectSubtreeSegments(node));
    dataBytes = 0;
    for (const id of ids) {
      dataBytes += segmentMap.get(id)?.byteLength ?? 0;
    }
    segmentCount = ids.size;
    bufferCount = 0;
  }

  const metadataBytes = subtreeMetadataBytes(node);
  const totalBytes = dataBytes + metadataBytes;

  return {
    dataBytes,
    metadataBytes,
    totalBytes,
    fractionOfFile: fileSize > 0 ? totalBytes / fileSize : 0,
    bytesPerRow: node.rowCount > 0 ? dataBytes / node.rowCount : null,
    rowCount: node.rowCount,
    segmentCount,
    bufferCount,
  };
}

/** Format a fraction (0..1) as a percentage string, e.g. "12.3%". */
export function formatPercent(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

/** Format bytes-per-row density, e.g. "4.0 B/row" or "—" when unknown. */
export function formatBytesPerRow(bytesPerRow: number | null): string {
  if (bytesPerRow === null) return '—';
  if (bytesPerRow >= 1024) return `${(bytesPerRow / 1024).toFixed(1)} KB/row`;
  if (bytesPerRow >= 10) return `${bytesPerRow.toFixed(0)} B/row`;
  return `${bytesPerRow.toFixed(2)} B/row`;
}
