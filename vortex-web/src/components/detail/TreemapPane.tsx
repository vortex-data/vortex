// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useRef, useEffect, useCallback, useState } from 'react';
import { hierarchy, treemap, treemapSquarify } from 'd3-hierarchy';
import type { HierarchyRectangularNode } from 'd3-hierarchy';
import type { LayoutTreeNode, SegmentMapEntry } from '../swimlane/types';
import {
  getNodeDisplayName,
  getDtypeCategory,
  collectSubtreeSegments,
  DTYPE_COLORS,
  formatBytes,
} from '../swimlane/utils';
import { useTheme } from '../../contexts/ThemeContext';

interface TreemapPaneProps {
  node: LayoutTreeNode;
  segments: SegmentMapEntry[];
  onSelectNode: (nodeId: string) => void;
  onHoverNode: (nodeId: string | null) => void;
}

interface TreeNode {
  name: string;
  nodeId: string;
  color: string;
  bytes: number;
  children?: TreeNode[];
}

type RectNode = HierarchyRectangularNode<TreeNode>;

/** Total buffer bytes for an array node subtree. */
function arraySubtreeBytes(node: LayoutTreeNode): number {
  const own = (node.bufferLengths ?? []).reduce((s, b) => s + b, 0);
  const childBytes = node.children
    .filter((c) => c.isArrayNode)
    .reduce((s, c) => s + arraySubtreeBytes(c), 0);
  return own + childBytes;
}

function buildTree(node: LayoutTreeNode, segmentMap: Map<number, SegmentMapEntry>): TreeNode {
  const color = DTYPE_COLORS[getDtypeCategory(node.dtype)];
  const name = getNodeDisplayName(node);

  // For array nodes, size by buffer bytes; for layout nodes, by segment bytes.
  // Layout-level treemaps skip array children to avoid eager expansion.
  const isArray = node.isArrayNode ?? false;
  const bytes = isArray
    ? arraySubtreeBytes(node)
    : collectSubtreeSegments(node).reduce(
        (sum, id) => sum + (segmentMap.get(id)?.byteLength ?? 0),
        0,
      );

  // For layout nodes, skip array children of NON-flat layouts to avoid eager expansion.
  // Flat layouts and array nodes show all their children (array tree is already fetched).
  const isFlatOrArray = isArray || node.encoding === 'vortex.flat';
  const childrenToShow = isFlatOrArray
    ? node.children
    : node.children.filter((c) => !c.isArrayNode);

  if (childrenToShow.length === 0) {
    return { name, nodeId: node.id, color, bytes: Math.max(bytes, 1) };
  }

  return {
    name,
    nodeId: node.id,
    color,
    bytes: Math.max(bytes, 1),
    children: childrenToShow.map((c) => buildTree(c, segmentMap)),
  };
}

function resolveThemeColors(choice: 'light' | 'dark' | 'system') {
  let isDark: boolean;
  if (choice === 'system') {
    isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  } else {
    isDark = choice === 'dark';
  }
  return isDark
    ? { fg: '#e4e4e8', dim: '#71717a', border: 'rgba(255,255,255,0.12)', highlight: '#2CB9D1' }
    : { fg: '#18181b', dim: '#71717a', border: 'rgba(0,0,0,0.1)', highlight: '#2CB9D1' };
}

/** Find the deepest node containing point (px, py), preferring depth >= 1. */
function hitTest(nodes: RectNode[], px: number, py: number): RectNode | null {
  let best: RectNode | null = null;
  for (const n of nodes) {
    if (n.depth >= 1 && px >= n.x0 && px < n.x1 && py >= n.y0 && py < n.y1) {
      if (!best || n.depth > best.depth) best = n;
    }
  }
  return best;
}

/** Collect all nodeIds in a RectNode subtree. */
function collectRectIds(n: RectNode): Set<string> {
  const ids = new Set<string>();
  function walk(node: RectNode) {
    ids.add(node.data.nodeId);
    if (node.children) {
      for (const c of node.children) walk(c);
    }
  }
  walk(n);
  return ids;
}

export function TreemapPane({ node, segments, onSelectNode, onHoverNode }: TreemapPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const segmentMap = useMemo(() => new Map(segments.map((s) => [s.index, s])), [segments]);

  const tree = useMemo(() => buildTree(node, segmentMap), [node, segmentMap]);
  const { theme: themeChoice } = useTheme();
  const theme = useMemo(() => resolveThemeColors(themeChoice), [themeChoice]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Reset local selection when the treemap root node changes.
  useEffect(() => {
    setSelectedNodeId(null);
  }, [node.id]);

  const nodes = useMemo<RectNode[]>(() => {
    if (!size) return [];
    const root = hierarchy(tree)
      .sum((d) => (d.children ? 0 : d.bytes))
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
    treemap<TreeNode>()
      .size([size.w, size.h])
      .tile(treemapSquarify)
      .paddingTop(18)
      .paddingInner(2)
      .paddingOuter(1)
      .round(true)(root);
    return root.descendants() as RectNode[];
  }, [tree, size]);

  // Set of nodeIds in the selected depth-1 subtree (for solid highlight).
  const selectedSubtreeIds = useMemo<Set<string>>(() => {
    if (!selectedNodeId) return new Set();
    const selected = nodes.find((n) => n.data.nodeId === selectedNodeId);
    return selected ? collectRectIds(selected) : new Set();
  }, [selectedNodeId, nodes]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const hit = hitTest(nodes, px, py);
      const nodeId = hit ? hit.data.nodeId : null;
      setHoveredNodeId(nodeId);
      onHoverNode(nodeId);
    },
    [nodes, onHoverNode],
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredNodeId(null);
    onHoverNode(null);
  }, [onHoverNode]);

  const handleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const hit = hitTest(nodes, px, py);
      if (hit) {
        e.stopPropagation();
        setSelectedNodeId(hit.data.nodeId);
      } else {
        setSelectedNodeId(null);
      }
    },
    [nodes],
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const hit = hitTest(nodes, px, py);
      if (hit) {
        e.stopPropagation();
        onSelectNode(hit.data.nodeId);
      }
    },
    [nodes, onSelectNode],
  );

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden">
      {size && (
        <svg
          ref={svgRef}
          width={size.w}
          height={size.h}
          className="block cursor-pointer"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          onDoubleClick={handleDoubleClick}
        >
          {nodes.map((n) => {
            const w = n.x1 - n.x0;
            const h = n.y1 - n.y0;
            if (w < 1 || h < 1) return null;

            const isLeaf = !n.children || n.children.length === 0;
            const d = n.data;
            const isHovered = d.nodeId === hoveredNodeId;
            const isSelected = selectedSubtreeIds.has(d.nodeId);
            const maxChars = Math.floor(w / 6);
            const label =
              maxChars < 2
                ? ''
                : d.name.length > maxChars
                  ? d.name.slice(0, maxChars - 1) + '\u2026'
                  : d.name;

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
                {n.depth === 1 && label && h > 14 && (
                  <text
                    x={n.x0 + 4}
                    y={n.y0 + 12}
                    fill={theme.fg}
                    fontSize={10}
                    fontFamily="'Geist Mono', monospace"
                  >
                    {label}
                  </text>
                )}
                {n.depth === 1 && w > 50 && h > 28 && (
                  <text
                    x={n.x0 + 4}
                    y={n.y0 + 23}
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
    </div>
  );
}
