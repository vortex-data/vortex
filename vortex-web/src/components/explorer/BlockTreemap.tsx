// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { hierarchy, treemap, treemapSquarify } from 'd3-hierarchy';
import type { HierarchyRectangularNode } from 'd3-hierarchy';
import type { LayoutTreeNode, SegmentMapEntry } from '../swimlane/types';
import {
  getNodeDisplayName,
  getDtypeCategory,
  shortEncoding,
  collectSubtreeSegments,
  findNodeById,
  findPathToNode,
  isFlatLayout,
  formatBytes,
  DTYPE_COLORS,
} from '../swimlane/utils';
import { useTheme } from '../../contexts/ThemeContext';
import {
  nodePhysicalStats,
  arraySubtreeBytes,
  formatPercent,
  formatBytesPerRow,
} from './physicalStats';

interface BlockTreemapProps {
  /** The full layout tree the map can explore. */
  root: LayoutTreeNode;
  segments: SegmentMapEntry[];
  fileSize: number;
  /** Currently selected node id — highlighted in the map. */
  selectedNodeId: string | null;
  /** Currently hovered node id (from selection context). */
  hoveredNodeId: string | null;
  /** Single-click selects a block (highlights it and scrolls the tree); does not
   *  re-root. Double-click zooms in. */
  onSelectNode: (id: string | null) => void;
  onHoverNode: (id: string | null) => void;
  /** Called when a flat layout should reveal its array-encoding children. */
  onExpand?: (id: string) => void;
}

interface TreeNode {
  name: string;
  nodeId: string;
  color: string;
  bytes: number;
  layoutNode: LayoutTreeNode;
  children?: TreeNode[];
}

type RectNode = HierarchyRectangularNode<TreeNode>;

// Header band reserved at the top of each container layout for its own name.
const HEADER = 16;
const PAD = 2;

/**
 * Total encoded *data* bytes for a node's whole subtree (no metadata) — this is
 * what tile areas are proportional to. Note the tooltip's "% of file" uses
 * total bytes (data + metadata), so a tile's area and its % can differ slightly.
 */
function subtreeBytes(node: LayoutTreeNode, segmentMap: Map<number, SegmentMapEntry>): number {
  if (node.isArrayNode) return arraySubtreeBytes(node);
  return collectSubtreeSegments(node).reduce(
    (sum, id) => sum + (segmentMap.get(id)?.byteLength ?? 0),
    0,
  );
}

/** Build the full nested treemap for the file so every physical block is
 *  visible down to the leaves. Flat / array layouts expose their array-encoding
 *  children; other layouts hide array children until expanded. */
function buildTree(node: LayoutTreeNode, segmentMap: Map<number, SegmentMapEntry>): TreeNode {
  const isArray = node.isArrayNode ?? false;
  const isFlatOrArray = isArray || node.encoding === 'vortex.flat';
  const children = isFlatOrArray ? node.children : node.children.filter((c) => !c.isArrayNode);
  const base = {
    name: getNodeDisplayName(node),
    nodeId: node.id,
    color: DTYPE_COLORS[getDtypeCategory(node.dtype)],
    bytes: Math.max(subtreeBytes(node, segmentMap), 1),
    layoutNode: node,
  };
  if (children.length === 0) return base;
  return { ...base, children: children.map((c) => buildTree(c, segmentMap)) };
}

interface ThemeColors {
  fg: string;
  dim: string;
  border: string;
  highlight: string;
}

function resolveThemeColors(choice: 'light' | 'dark' | 'system'): ThemeColors {
  const isDark =
    choice === 'system'
      ? window.matchMedia('(prefers-color-scheme: dark)').matches
      : choice === 'dark';
  return isDark
    ? { fg: '#e4e4e8', dim: '#71717a', border: 'rgba(255,255,255,0.12)', highlight: '#2CB9D1' }
    : { fg: '#18181b', dim: '#71717a', border: 'rgba(0,0,0,0.1)', highlight: '#2CB9D1' };
}

interface Tooltip {
  node: LayoutTreeNode;
  x: number;
  y: number;
}

export function BlockTreemap({
  root,
  segments,
  fileSize,
  selectedNodeId,
  hoveredNodeId,
  onSelectNode,
  onHoverNode,
  onExpand,
}: BlockTreemapProps) {
  const boxRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [tooltip, setTooltip] = useState<Tooltip | null>(null);
  const [localHover, setLocalHover] = useState<string | null>(null);
  // The layout the map is zoomed into (double-click drills in). Independent of
  // selection, so single-click can select a block without re-rooting.
  const [drillId, setDrillId] = useState(root.id);
  // Tracks selections the map itself made, so they don't trigger a re-zoom — only
  // external selections (e.g. clicking the tree panel) drive the zoom.
  const selfSelected = useRef<string | null>(null);

  const { theme: themeChoice } = useTheme();
  const theme = useMemo(() => resolveThemeColors(themeChoice), [themeChoice]);
  const segmentMap = useMemo(() => new Map(segments.map((s) => [s.index, s])), [segments]);
  const drillNode = useMemo(() => findNodeById(root, drillId) ?? root, [root, drillId]);
  const tree = useMemo(() => buildTree(drillNode, segmentMap), [drillNode, segmentMap]);
  const drillPath = useMemo(() => findPathToNode(root, drillNode.id), [root, drillNode.id]);
  const drillParent = drillPath.length >= 2 ? drillPath[drillPath.length - 2] : null;

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Reset the zoom when the file (root) changes.
  useEffect(() => setDrillId(root.id), [root.id]);

  // An external selection (e.g. clicking the tree panel) drives the zoom: re-root
  // the map at the selected node. Selections the map made itself are ignored via
  // a one-shot marker that is cleared as soon as it is consumed — otherwise a
  // stale marker would suppress a later external re-selection of the same node.
  useEffect(() => {
    if (!selectedNodeId) return;
    if (selectedNodeId === selfSelected.current) {
      selfSelected.current = null;
      return;
    }
    selfSelected.current = null;
    setDrillId(selectedNodeId);
    const node = findNodeById(root, selectedNodeId);
    if (node && isFlatLayout(node) && !node.children.some((c) => c.isArrayNode)) {
      onExpand?.(node.id);
    }
  }, [selectedNodeId, root, onExpand]);

  const nodes = useMemo<RectNode[]>(() => {
    if (!size) return [];
    const r = hierarchy(tree)
      .sum((d) => (d.children ? 0 : d.bytes))
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
    treemap<TreeNode>()
      // Lay out HEADER taller and shift up so the root's own header band sits
      // off-screen — top-level fields start flush with the top.
      .size([size.w, size.h + HEADER])
      .tile(treemapSquarify)
      .paddingInner(PAD)
      .paddingOuter(PAD)
      // paddingTop MUST be set after paddingOuter (which also sets the top pad),
      // or the header band collapses and container names overlap their children.
      .paddingTop(HEADER)
      .round(true)(r);
    const desc = r.descendants() as RectNode[];
    for (const n of desc) {
      n.y0 -= HEADER;
      n.y1 -= HEADER;
    }
    return desc;
  }, [tree, size]);

  const byId = useMemo(() => {
    const m = new Map<string, RectNode>();
    for (const n of nodes) m.set(n.data.nodeId, n);
    return m;
  }, [nodes]);

  // Node ids in the selected subtree — rendered with a solid tint.
  const selectedSubtreeIds = useMemo<Set<string>>(() => {
    const ids = new Set<string>();
    const sel = selectedNodeId ? byId.get(selectedNodeId) : undefined;
    if (!sel) return ids;
    (function walk(n: RectNode) {
      ids.add(n.data.nodeId);
      if (n.children) for (const c of n.children) walk(c);
    })(sel);
    return ids;
  }, [selectedNodeId, byId]);

  const activeHover = localHover ?? hoveredNodeId;

  /** Deepest tile (below the root) containing a point. A container's header band
   *  is not covered by children, so clicking there selects the container. */
  const hitTest = useCallback(
    (px: number, py: number): RectNode | null => {
      let best: RectNode | null = null;
      for (const n of nodes) {
        if (n.depth >= 1 && px >= n.x0 && px < n.x1 && py >= n.y0 && py < n.y1) {
          if (!best || n.depth > best.depth) best = n;
        }
      }
      return best;
    },
    [nodes],
  );

  const localPoint = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    return { px: clientX - rect.left, py: clientY - rect.top };
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const p = localPoint(e.clientX, e.clientY);
      if (!p) return;
      const hit = hitTest(p.px, p.py);
      const id = hit ? hit.data.nodeId : null;
      setLocalHover(id);
      onHoverNode(id);
      setTooltip(hit ? { node: hit.data.layoutNode, x: e.clientX, y: e.clientY } : null);
    },
    [hitTest, localPoint, onHoverNode],
  );

  const handleMouseLeave = useCallback(() => {
    setLocalHover(null);
    setTooltip(null);
    onHoverNode(null);
  }, [onHoverNode]);

  /** Reveal a flat layout's array buffers in place so they render to leaves. */
  const expandIfFlat = useCallback(
    (node: LayoutTreeNode) => {
      if (isFlatLayout(node) && !node.children.some((c) => c.isArrayNode)) onExpand?.(node.id);
    },
    [onExpand],
  );

  // Single click selects the block under the cursor — highlighting it and
  // scrolling the tree panel to it — without changing the zoom.
  const handleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const p = localPoint(e.clientX, e.clientY);
      if (!p) return;
      const hit = hitTest(p.px, p.py);
      if (!hit) return;
      selfSelected.current = hit.data.nodeId;
      onSelectNode(hit.data.nodeId);
      expandIfFlat(hit.data.layoutNode);
    },
    [hitTest, localPoint, onSelectNode, expandIfFlat],
  );

  // Double click zooms in: re-root the map at the block (and select it).
  const handleDoubleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const p = localPoint(e.clientX, e.clientY);
      if (!p) return;
      const hit = hitTest(p.px, p.py);
      if (!hit) return;
      e.stopPropagation();
      selfSelected.current = hit.data.nodeId;
      setDrillId(hit.data.nodeId);
      onSelectNode(hit.data.nodeId);
      expandIfFlat(hit.data.layoutNode);
    },
    [hitTest, localPoint, onSelectNode, expandIfFlat],
  );

  return (
    <div ref={boxRef} className="relative w-full h-full overflow-hidden">
      {/* Zoom-out control, shown when drilled into a child layout. */}
      {drillParent && (
        <button
          onClick={() => {
            selfSelected.current = drillParent.id;
            setDrillId(drillParent.id);
            onSelectNode(drillParent.id);
          }}
          title={`Up to ${getNodeDisplayName(drillParent)}`}
          className="absolute top-1.5 left-1.5 z-10 flex items-center gap-1 rounded border border-vortex-grey-light/60 dark:border-white/[0.12] bg-vortex-white/90 dark:bg-vortex-black/80 px-1.5 py-0.5 text-[10px] font-mono text-vortex-grey-dark hover:text-vortex-light-blue shadow-sm backdrop-blur-sm cursor-pointer"
        >
          ↑ {getNodeDisplayName(drillParent)}
        </button>
      )}
      {size && (
        <svg
          ref={svgRef}
          width={size.w}
          height={size.h}
          className="block select-none cursor-pointer"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          onDoubleClick={handleDoubleClick}
        >
          {nodes.map((n) => {
            // The drill root's frame is shifted off-screen — skip it, unless the
            // root is itself a leaf (then render it as the single block).
            if (n.depth === 0 && n.children && n.children.length > 0) return null;
            const w = n.x1 - n.x0;
            const h = n.y1 - n.y0;
            if (w < 1 || h < 1) return null;

            const d = n.data;
            const isLeaf = !n.children || n.children.length === 0;
            const isHovered = d.nodeId === activeHover;
            const isSelected = selectedSubtreeIds.has(d.nodeId);
            const maxChars = Math.floor((w - 6) / 6);
            const label = maxChars < 2 ? '' : truncate(d.name, maxChars);

            return (
              <g key={d.nodeId} pointerEvents="none">
                <rect
                  x={n.x0}
                  y={n.y0}
                  width={w}
                  height={h}
                  fill={d.color}
                  fillOpacity={isSelected ? 0.4 : isLeaf ? 0.18 : 0.06}
                  stroke={isHovered ? theme.highlight : theme.border}
                  strokeWidth={isHovered ? 2 : isLeaf ? 0.5 : 1}
                />
                {/* Name sits locally on the block: in a leaf's body, or in a
                    container's header band (which children never occupy, so it
                    cannot collide with them). */}
                {label && h > 11 && (
                  <text
                    x={n.x0 + 4}
                    y={n.y0 + 11}
                    fill={theme.fg}
                    fontSize={10}
                    fontWeight={isLeaf ? 400 : 600}
                    fontFamily="'Geist Mono', monospace"
                  >
                    {label}
                  </text>
                )}
                {isLeaf && w > 50 && h > 26 && (
                  <text
                    x={n.x0 + 4}
                    y={n.y0 + 22}
                    fill={theme.dim}
                    fontSize={9}
                    fontFamily="'Geist Mono', monospace"
                  >
                    {formatBytes(d.bytes)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      )}

      {tooltip && <TileTooltip tooltip={tooltip} segmentMap={segmentMap} fileSize={fileSize} />}
    </div>
  );
}

function truncate(s: string, maxChars: number): string {
  if (maxChars < 2) return '';
  return s.length > maxChars ? s.slice(0, maxChars - 1) + '…' : s;
}

function TileTooltip({
  tooltip,
  segmentMap,
  fileSize,
}: {
  tooltip: Tooltip;
  segmentMap: Map<number, SegmentMapEntry>;
  fileSize: number;
}) {
  const { node, x, y } = tooltip;
  // Keyed on the node so it isn't recomputed for every pixel of mouse movement.
  const stats = useMemo(
    () => nodePhysicalStats(node, segmentMap, fileSize),
    [node, segmentMap, fileSize],
  );
  const dtypeCat = getDtypeCategory(node.dtype);
  const dtypeColor = DTYPE_COLORS[dtypeCat];
  return (
    <div
      className="fixed z-[1000] pointer-events-none max-w-[240px] rounded-lg border border-vortex-grey-light/60 dark:border-white/[0.1] bg-vortex-white dark:bg-[#252528] shadow-lg p-2 text-xs"
      style={{ left: x + 12, top: y - 10 }}
    >
      <div className="mb-1 flex items-center gap-1.5">
        <span className="font-medium text-vortex-fg-light dark:text-vortex-fg">
          {getNodeDisplayName(node)}
        </span>
        <span
          className="rounded px-1.5 py-0.5 text-[9px]"
          style={{ color: dtypeColor, backgroundColor: `${dtypeColor}20` }}
        >
          {dtypeCat}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-vortex-grey-dark">
        <span>rows</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">
          {stats.rowCount.toLocaleString()}
        </span>
        <span>encoding</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg" title={node.encoding}>
          {shortEncoding(node.encoding)}
        </span>
        <span>data</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">
          {formatBytes(stats.dataBytes)}
        </span>
        <span>% of file</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">
          {formatPercent(stats.fractionOfFile)}
        </span>
        <span>density</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">
          {formatBytesPerRow(stats.bytesPerRow)}
        </span>
      </div>
    </div>
  );
}
