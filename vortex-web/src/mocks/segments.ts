// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode, SegmentMapEntry } from '../components/swimlane/types';
import { getNodeDisplayName } from '../components/swimlane/utils';

/**
 * Generate SegmentMapEntry[] by walking a layout tree and assigning byte offsets.
 */
export function generateSegments(root: LayoutTreeNode, fileSize: number): SegmentMapEntry[] {
  const entries: SegmentMapEntry[] = [];

  // Collect all segment IDs with their layout context
  function walk(node: LayoutTreeNode, columnPath: string) {
    const name = getNodeDisplayName(node);
    const currentPath = columnPath ? `${columnPath}.${name}` : name;

    for (const segId of node.segmentIds) {
      entries.push({
        index: segId,
        byteOffset: 0, // filled in below
        byteLength: 0,
        alignment: 64,
        column: node.childType.kind === 'field' ? node.childType.fieldName : null,
        layoutPath: node.id,
      });
    }

    for (const child of node.children) {
      walk(child, currentPath);
    }
  }

  walk(root, '');

  // Sort by index and assign byte offsets proportionally
  entries.sort((a, b) => a.index - b.index);
  const totalSegments = entries.length;
  if (totalSegments === 0) return entries;

  // Reserve ~10% for metadata at the end
  const dataRegionSize = Math.floor(fileSize * 0.9);
  const avgSegmentSize = Math.floor(dataRegionSize / totalSegments);

  let offset = 0;
  for (const entry of entries) {
    // Add some variance: segments between 0.5x and 1.5x average size
    const variance = 0.5 + hashIndex(entry.index) / 0xffff;
    const segmentSize = Math.max(64, Math.floor(avgSegmentSize * variance));
    const alignment = 64;
    // Align offset
    offset = Math.ceil(offset / alignment) * alignment;

    entry.byteOffset = offset;
    entry.byteLength = segmentSize;
    offset += segmentSize;
  }

  return entries;
}

function hashIndex(n: number): number {
  // Simple deterministic hash for variance
  let h = n * 2654435761;
  h = ((h >>> 16) ^ h) * 0x45d9f3b;
  return (h >>> 0) & 0xffff;
}
