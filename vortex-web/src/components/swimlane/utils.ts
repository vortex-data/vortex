// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type {
  LayoutTreeNode,
  ArrayEncodingNode,
  Split,
  DtypeCategory,
  FlattenedRow,
  DisplayKind,
} from './types';

// Encoding styles — keyed by encoding string, with fallback for unknown encodings
export const ENCODING_STYLES: Record<string, { color: string; label: string }> = {
  'vortex.struct': { color: '#5971FD', label: 'struct' },
  'vortex.chunked': { color: '#CEE562', label: 'chunked' },
  'vortex.flat': { color: '#2CB9D1', label: 'flat' },
  'vortex.dict': { color: '#EEB3E1', label: 'dict' },
  'vortex.zonemap': { color: '#FB863D', label: 'zonemap' },
  'vortex.fsst': { color: '#EEB3E1', label: 'fsst' },
  'vortex.roaring_bool': { color: '#EEB3E1', label: 'roaring' },
  'vortex.roaring_int': { color: '#2CB9D1', label: 'roaring' },
  'vortex.alp': { color: '#FB863D', label: 'alp' },
  'vortex.alp_rd': { color: '#FB863D', label: 'alp-rd' },
  'vortex.for': { color: '#CEE562', label: 'for' },
  'vortex.bitpacked': { color: '#CEE562', label: 'bitpacked' },
  'vortex.runend': { color: '#CEE562', label: 'run-end' },
  'vortex.runend_bool': { color: '#CEE562', label: 'run-end' },
  'vortex.zigzag': { color: '#2CB9D1', label: 'zigzag' },
  'vortex.constant': { color: '#8F8F8F', label: 'const' },
  'vortex.sparse': { color: '#8F8F8F', label: 'sparse' },
};

const DEFAULT_ENCODING_STYLE = { color: '#8F8F8F', label: 'unknown' };

export function getEncodingStyle(encoding: string): { color: string; label: string } {
  return (
    ENCODING_STYLES[encoding] ?? {
      ...DEFAULT_ENCODING_STYLE,
      label: encoding.split('.').pop() ?? encoding,
    }
  );
}

// Dtype colors (for flat chunk bars in swimlane)
export const DTYPE_COLORS: Record<DtypeCategory, string> = {
  bool: '#D97BC6',
  int: '#2CB9D1',
  float: '#FB863D',
  utf8: '#5971FD',
  datetime: '#8BB536',
  list: '#A78BFA',
  struct: '#999999',
  other: '#777777',
};

export const DTYPE_CATEGORIES: DtypeCategory[] = [
  'bool',
  'int',
  'float',
  'utf8',
  'datetime',
  'list',
  'struct',
  'other',
];

export const ROW_HEIGHT = 26;
export const MIN_LABEL_WIDTH = 36;
export const GROUP_SIZE = 10;

/**
 * Determine dtype category from dtype string
 */
export function getDtypeCategory(dtype?: string): DtypeCategory {
  if (!dtype) return 'other';
  const d = dtype.toLowerCase();
  if (d === 'bool' || d === 'boolean') return 'bool';
  // Struct/list checks first — composite dtypes contain field names that would false-match others.
  if (d.startsWith('{') || d === 'struct') return 'struct';
  if (d.includes('list') || d.includes('array')) return 'list';
  if (d.includes('utf8') || d.includes('string') || d.includes('binary')) return 'utf8';
  if (d.includes('timestamp') || d.includes('date') || d.includes('time')) return 'datetime';
  if (d.includes('int') || d.includes('uint') || d.startsWith('i') || d.startsWith('u'))
    return 'int';
  if (d.includes('float') || d.includes('double') || d.includes('decimal') || d.startsWith('f'))
    return 'float';
  return 'other';
}

/**
 * Check if two ranges overlap
 */
export function rangesOverlap(a: [number, number], b: [number, number]): boolean {
  return a[0] < b[1] && b[0] < a[1];
}

/**
 * Collect all row boundaries from a LayoutTreeNode tree
 */
export function collectBoundaries(node: LayoutTreeNode, set: Set<number> = new Set()): Set<number> {
  set.add(node.rowOffset);
  set.add(node.rowOffset + node.rowCount);

  for (const child of node.children) {
    collectBoundaries(child, set);
  }

  return set;
}

/**
 * Create splits from layout boundaries
 */
export function createSplits(layout: LayoutTreeNode): Split[] {
  const boundaries = Array.from(collectBoundaries(layout)).sort((a, b) => a - b);
  return boundaries.slice(0, -1).map((start, i) => ({
    id: `s${i}`,
    rowRange: [start, boundaries[i + 1]] as [number, number],
  }));
}

/**
 * Get the combined row range of selected splits
 */
export function getSelectedRowRange(
  splits: Split[],
  selectedSplits: Set<string>,
): [number, number] | null {
  if (selectedSplits.size === 0) return null;
  const selected = splits.filter((s) => selectedSplits.has(s.id));
  const min = Math.min(...selected.map((s) => s.rowRange[0]));
  const max = Math.max(...selected.map((s) => s.rowRange[1]));
  return [min, max];
}

/**
 * Get the display name for a layout tree node based on its child type
 */
export function getNodeDisplayName(node: LayoutTreeNode): string {
  const ct = node.childType;
  switch (ct.kind) {
    case 'root':
      return 'root';
    case 'field':
      return ct.fieldName;
    case 'chunk':
      return `[${ct.chunkIndex}]`;
    case 'transparent':
      return ct.name;
    case 'auxiliary':
      return ct.name;
  }
}

/**
 * Get a badge label for schema mode
 */
export function getSchemaLabel(node: LayoutTreeNode): string {
  return node.dtype;
}

/**
 * Get a badge label for layout mode
 */
export function getLayoutLabel(node: LayoutTreeNode): string {
  const ct = node.childType;
  switch (ct.kind) {
    case 'root':
      return getEncodingStyle(node.encoding).label;
    case 'field':
      return `[field] ${getEncodingStyle(node.encoding).label}`;
    case 'chunk':
      return `[chunk ${ct.chunkIndex}]`;
    case 'transparent':
      return `[transparent: ${ct.name}]`;
    case 'auxiliary':
      return `[aux: ${ct.name}]`;
  }
}

/**
 * Check if a node has expandable children
 */
export function hasExpandableChildren(node: LayoutTreeNode): boolean {
  return node.children.length > 0;
}

/**
 * Get the row range tuple for a node
 */
export function getNodeRowRange(node: LayoutTreeNode): [number, number] {
  return [node.rowOffset, node.rowOffset + node.rowCount];
}

/**
 * Check if a node is a "field" child — used in schema mode to identify logical columns
 */
function isFieldChild(node: LayoutTreeNode): boolean {
  return node.childType.kind === 'field';
}

/**
 * In schema mode, find the field-level children, skipping intermediate layout nodes.
 * Returns the node itself if it has field children, or walks through transparent/chunk/aux
 * nodes to find them.
 */
function collectSchemaChildren(node: LayoutTreeNode): LayoutTreeNode[] {
  const fieldChildren = node.children.filter(isFieldChild);
  if (fieldChildren.length > 0) return fieldChildren;
  // No field children — this is a leaf or layout-only node
  return [];
}

/**
 * Group children into groups of GROUP_SIZE when there are too many.
 * Returns null if grouping is not needed.
 */
export function groupChildren(
  children: LayoutTreeNode[],
  parentId: string,
): { groups: LayoutTreeNode[]; isGrouped: true } | null {
  if (children.length <= GROUP_SIZE) return null;

  const groups: LayoutTreeNode[] = [];
  for (let i = 0; i < children.length; i += GROUP_SIZE) {
    const groupNodes = children.slice(i, Math.min(i + GROUP_SIZE, children.length));
    const startIdx = i;
    const endIdx = Math.min(i + GROUP_SIZE - 1, children.length - 1);
    const firstNode = groupNodes[0];
    const lastNode = groupNodes[groupNodes.length - 1];

    groups.push({
      id: `${parentId}_group_${startIdx}_${endIdx}`,
      encoding: 'group',
      dtype: '',
      rowCount: lastNode.rowOffset + lastNode.rowCount - firstNode.rowOffset,
      rowOffset: firstNode.rowOffset,
      metadataBytes: 0,
      segmentIds: [],
      childType: { kind: 'transparent', name: `chunks ${startIdx}–${endIdx}` },
      children: groupNodes,
    });
  }
  return { groups, isGrouped: true };
}

/**
 * Flatten a layout tree into rows for rendering.
 *
 * @param root - The root layout tree node
 * @param expanded - Set of expanded node IDs
 * @param selectedRange - Optional selected row range for filtering
 * @param mode - 'schema' shows logical column hierarchy, 'layout' shows full layout tree
 */
export function flattenTree(
  root: LayoutTreeNode,
  expanded: Set<string>,
  selectedRange: [number, number] | null,
  mode: 'schema' | 'layout',
): FlattenedRow[] {
  const result: FlattenedRow[] = [];

  function walk(node: LayoutTreeNode, depth: number) {
    const rowRange = getNodeRowRange(node);
    result.push({ node, depth, displayKind: 'normal', rowRange });

    if (!expanded.has(node.id)) return;

    let childrenToShow: LayoutTreeNode[];

    if (mode === 'schema') {
      // In schema mode, show field children at the top level.
      // If a field child has no field sub-children, show its layout children when expanded.
      const schemaChildren = collectSchemaChildren(node);
      childrenToShow = schemaChildren.length > 0 ? schemaChildren : node.children;
    } else {
      childrenToShow = node.children;
    }

    // Apply chunk grouping if there are many children of the same type
    const chunkChildren = childrenToShow.filter((c) => c.childType.kind === 'chunk');
    const nonChunkChildren = childrenToShow.filter((c) => c.childType.kind !== 'chunk');

    // Show non-chunk children first
    for (const child of nonChunkChildren) {
      walk(child, depth + 1);
    }

    // Group chunk children if needed
    if (chunkChildren.length > 0) {
      const groupResult = groupChildren(chunkChildren, node.id);

      if (groupResult) {
        const visibleGroups = selectedRange
          ? groupResult.groups.filter((g) => rangesOverlap(getNodeRowRange(g), selectedRange))
          : groupResult.groups;

        for (const group of visibleGroups) {
          const groupRowRange = getNodeRowRange(group);
          result.push({
            node: group,
            depth: depth + 1,
            displayKind: 'group',
            groupedChildren: group.children,
            rowRange: groupRowRange,
          });

          if (expanded.has(group.id)) {
            const visibleInGroup = selectedRange
              ? group.children.filter((c) => rangesOverlap(getNodeRowRange(c), selectedRange))
              : group.children;

            for (const child of visibleInGroup) {
              walk(child, depth + 2);
            }

            if (selectedRange && visibleInGroup.length < group.children.length) {
              addHiddenIndicator(
                group,
                group.children.length - visibleInGroup.length,
                depth + 2,
                'in group',
              );
            }
          }
        }

        if (selectedRange && visibleGroups.length < groupResult.groups.length) {
          addHiddenIndicator(
            node,
            groupResult.groups.length - visibleGroups.length,
            depth + 1,
            'groups',
          );
        }
      } else {
        const visible = selectedRange
          ? chunkChildren.filter((c) => rangesOverlap(getNodeRowRange(c), selectedRange))
          : chunkChildren;

        for (const child of visible) {
          walk(child, depth + 1);
        }

        if (selectedRange && visible.length < chunkChildren.length) {
          addHiddenIndicator(node, chunkChildren.length - visible.length, depth + 1, 'chunks');
        }
      }
    }
  }

  function addHiddenIndicator(
    parent: LayoutTreeNode,
    hiddenCount: number,
    depth: number,
    label: string,
  ) {
    const indicator: LayoutTreeNode = {
      id: `${parent.id}_hidden_${label}`,
      encoding: '',
      dtype: '',
      rowCount: parent.rowCount,
      rowOffset: parent.rowOffset,
      metadataBytes: 0,
      segmentIds: [],
      childType: { kind: 'transparent', name: `${hiddenCount} more ${label}` },
      children: [],
    };
    result.push({
      node: indicator,
      depth,
      displayKind: 'hiddenIndicator' as DisplayKind,
      rowRange: getNodeRowRange(parent),
    });
  }

  walk(root, 0);
  return result;
}

/**
 * Find a node by ID in a layout tree
 */
export function findNodeById(root: LayoutTreeNode, id: string): LayoutTreeNode | null {
  if (root.id === id) return root;
  for (const child of root.children) {
    const found = findNodeById(child, id);
    if (found) return found;
  }
  return null;
}

/**
 * Find the path from root to a node (inclusive of both endpoints).
 * Returns an empty array if the node is not found.
 */
export function findPathToNode(root: LayoutTreeNode, id: string): LayoutTreeNode[] {
  if (root.id === id) return [root];
  for (const child of root.children) {
    const path = findPathToNode(child, id);
    if (path.length > 0) return [root, ...path];
  }
  return [];
}

/**
 * Collect all node IDs in a subtree
 */
export function collectSubtreeIds(node: LayoutTreeNode): Set<string> {
  const ids = new Set<string>();
  function walk(n: LayoutTreeNode) {
    ids.add(n.id);
    for (const child of n.children) walk(child);
  }
  walk(node);
  return ids;
}

/**
 * Collect all segment IDs reachable from a subtree
 */
export function collectSubtreeSegments(node: LayoutTreeNode): number[] {
  const segments: number[] = [];
  function walk(n: LayoutTreeNode) {
    segments.push(...n.segmentIds);
    for (const child of n.children) walk(child);
  }
  walk(node);
  return segments;
}

/**
 * Filter nodes matching a search query (and their ancestors)
 */
export function filterTreeBySearch(
  rows: FlattenedRow[],
  query: string,
  root: LayoutTreeNode,
): FlattenedRow[] {
  if (!query.trim()) return rows;

  const lowerQuery = query.toLowerCase();
  const matchingIds = new Set<string>();

  // Find all matching nodes
  function findMatches(node: LayoutTreeNode) {
    const name = getNodeDisplayName(node).toLowerCase();
    const dtype = node.dtype.toLowerCase();
    const encoding = node.encoding.toLowerCase();
    if (name.includes(lowerQuery) || dtype.includes(lowerQuery) || encoding.includes(lowerQuery)) {
      matchingIds.add(node.id);
    }
    for (const child of node.children) findMatches(child);
  }
  findMatches(root);

  // Collect ancestors of matching nodes
  const ancestorIds = new Set<string>();
  function collectAncestors(node: LayoutTreeNode, path: string[]) {
    if (matchingIds.has(node.id)) {
      for (const id of path) ancestorIds.add(id);
    }
    for (const child of node.children) {
      collectAncestors(child, [...path, node.id]);
    }
  }
  collectAncestors(root, []);

  // Collect descendants of matching nodes so expanded children are visible.
  const descendantIds = new Set<string>();
  function collectDescendants(node: LayoutTreeNode) {
    descendantIds.add(node.id);
    for (const child of node.children) collectDescendants(child);
  }
  function findDescendantsOfMatches(node: LayoutTreeNode) {
    if (matchingIds.has(node.id)) {
      collectDescendants(node);
    } else {
      for (const child of node.children) findDescendantsOfMatches(child);
    }
  }
  findDescendantsOfMatches(root);

  const visibleIds = new Set([...matchingIds, ...ancestorIds, ...descendantIds]);
  return rows.filter((row) => visibleIds.has(row.node.id));
}

/**
 * Strip the `vortex.` prefix from an encoding name for display.
 */
export function shortEncoding(encoding: string): string {
  return encoding.startsWith('vortex.') ? encoding.slice(7) : encoding;
}

/**
 * Format bytes to human readable string
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

/**
 * Format row range to compact string (e.g., "0k–25k")
 */
export function formatRowRange(range: [number, number]): string {
  const fmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
  return `${fmt(range[0])}–${fmt(range[1])}`;
}

/**
 * Format row count to compact string (e.g., "27.1k")
 */
export function formatRowCount(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1000000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1000000).toFixed(1)}M`;
}

/**
 * Convert an ArrayEncodingNode tree into LayoutTreeNode children
 * so they can appear in the layout tree under a flat layout node.
 */
export function arrayTreeToLayoutChildren(
  arrayTree: ArrayEncodingNode,
  parentNode: LayoutTreeNode,
): LayoutTreeNode[] {
  function convert(node: ArrayEncodingNode, parentId: string, name: string): LayoutTreeNode {
    const id = `${parentId}.$${name}`;

    const children = node.children.map((child, i) => {
      const childName = node.childNames[i] ?? `child ${i}`;
      return convert(child, id, childName);
    });

    return {
      id,
      encoding: node.encoding,
      dtype: node.dtype || parentNode.dtype,
      rowCount: parentNode.rowCount,
      rowOffset: parentNode.rowOffset,
      metadataBytes: node.metadataBytes,
      segmentIds: [],
      childType: { kind: 'field', fieldName: name },
      children,
      isArrayNode: true,
      bufferLengths: node.bufferLengths,
      bufferNames: node.bufferNames,
    };
  }

  // Wrap the entire array tree as a single "array" child of the flat layout.
  return [convert(arrayTree, parentNode.id, 'array')];
}

/**
 * Check if a layout node is a flat layout that can have array children.
 */
export function isFlatLayout(node: LayoutTreeNode): boolean {
  return node.encoding === 'vortex.flat' && !node.isArrayNode;
}

/**
 * Parse an array node ID into its layout node ID and array child path.
 * Array node IDs look like "root.col.$array.$values.$encoded".
 * The first `$array` segment represents the root of the decoded array tree,
 * so the WASM-side path skips it: → layoutNodeId: "root.col", arrayPath: ["values", "encoded"]
 */
export function parseArrayNodeId(nodeId: string): { layoutNodeId: string; arrayPath: string[] } {
  const parts = nodeId.split('.');
  const firstArrayIdx = parts.findIndex((p) => p.startsWith('$'));
  if (firstArrayIdx === -1) {
    return { layoutNodeId: nodeId, arrayPath: [] };
  }
  // Skip the first $array segment — it represents the decoded root, not a child to navigate to.
  const arraySegments = parts.slice(firstArrayIdx).map((p) => p.slice(1));
  return {
    layoutNodeId: parts.slice(0, firstArrayIdx).join('.'),
    arrayPath: arraySegments.slice(1),
  };
}
