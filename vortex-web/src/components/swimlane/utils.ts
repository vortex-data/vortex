import type {
  LayoutNode,
  Split,
  DtypeCategory,
  LayoutType,
  ChunkNode,
  ZoneNode,
  ChunkGroup,
} from './types';

// Layout type styles (for tree badges) — using Vortex palette as hex
export const LAYOUT_STYLES: Record<
  LayoutType | 'chunk' | 'zone' | 'chunkGroup',
  { color: string; label: string }
> = {
  struct: { color: '#5971FD', label: 'struct' }, // vortex-blue
  chunked: { color: '#CEE562', label: 'chunked' }, // vortex-green
  chunk: { color: '#CEE562', label: 'chunk' }, // vortex-green
  chunkGroup: { color: '#CEE562', label: '···' }, // vortex-green
  zonemap: { color: '#FB863D', label: 'zonemap' }, // vortex-orange
  zone: { color: '#FB863D', label: 'zone' }, // vortex-orange
  dict: { color: '#EEB3E1', label: 'dict' }, // vortex-pink
  flat: { color: '#2CB9D1', label: 'flat' }, // vortex-light-blue
};

// Dtype colors (for flat chunk bars in swimlane) — using Vortex palette as hex
export const DTYPE_COLORS: Record<DtypeCategory, string> = {
  bool: '#EEB3E1', // vortex-pink
  int: '#2CB9D1', // vortex-light-blue
  float: '#FB863D', // vortex-orange
  struct: '#5971FD', // vortex-blue
  list: '#CEE562', // vortex-green
  other: '#8F8F8F', // vortex-grey-dark
};

export const DTYPE_CATEGORIES: DtypeCategory[] = [
  'bool',
  'int',
  'float',
  'struct',
  'list',
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
  if (d.includes('int') || d.includes('uint')) return 'int';
  if (d.includes('float') || d.includes('double') || d.includes('decimal')) return 'float';
  if (d.includes('struct') && !d.includes('list')) return 'struct';
  if (d.includes('list') || d.includes('array')) return 'list';
  return 'other';
}

/**
 * Check if two ranges overlap
 */
export function rangesOverlap(a: [number, number], b: [number, number]): boolean {
  return a[0] < b[1] && b[0] < a[1];
}

/**
 * Collect all row boundaries from a layout tree
 */
export function collectBoundaries(
  node: LayoutNode | ChunkNode | ZoneNode,
  set: Set<number> = new Set(),
): Set<number> {
  set.add(node.rowRange[0]);
  set.add(node.rowRange[1]);

  if ('chunks' in node && node.chunks) {
    node.chunks.forEach((c) => collectBoundaries(c, set));
  }
  if ('zones' in node && node.zones) {
    node.zones.forEach((z) => collectBoundaries(z, set));
  }
  if ('children' in node && node.children) {
    node.children.forEach((c) => collectBoundaries(c, set));
  }
  if ('child' in node && node.child) {
    collectBoundaries(node.child, set);
  }

  return set;
}

/**
 * Create splits from layout boundaries
 */
export function createSplits(layout: LayoutNode): Split[] {
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
 * Group chunks into hierarchical groups of GROUP_SIZE
 */
export function groupChunks(chunks: ChunkNode[], parentId: string): ChunkGroup[] | null {
  if (chunks.length <= GROUP_SIZE) return null;

  const groups: ChunkGroup[] = [];
  for (let i = 0; i < chunks.length; i += GROUP_SIZE) {
    const groupChunks = chunks.slice(i, Math.min(i + GROUP_SIZE, chunks.length));
    const startIdx = i;
    const endIdx = Math.min(i + GROUP_SIZE - 1, chunks.length - 1);
    groups.push({
      id: `${parentId}_group_${startIdx}_${endIdx}`,
      name: `chunks ${startIdx}–${endIdx}`,
      type: 'chunkGroup',
      rowRange: [groupChunks[0].rowRange[0], groupChunks[groupChunks.length - 1].rowRange[1]],
      chunks: groupChunks,
      _isGroup: true,
    });
  }
  return groups;
}

/**
 * Format bytes to human readable string
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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
  return `${(n / 1000).toFixed(1)}k`;
}

/**
 * Check if a node has expandable children
 */
export function hasExpandableChildren(
  node: LayoutNode | ChunkNode | ZoneNode | ChunkGroup,
): boolean {
  if ('chunks' in node && node.chunks) return true;
  if ('zones' in node && node.zones) return true;
  if ('children' in node && node.children) return true;
  if ('child' in node && node.child) return true;
  if ('_isGroup' in node && node._isGroup) return true;
  return false;
}
