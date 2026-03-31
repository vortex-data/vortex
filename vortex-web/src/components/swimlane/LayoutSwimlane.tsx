// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import type { LayoutTreeNode, FlattenedRow } from './types';
import { flattenTree, filterTreeBySearch } from './utils';
import { ROW_HEIGHT, TREE_WIDTH, DEFAULT_SWIMLANE_MIN_WIDTH } from './styles';
import { TreeRow } from './TreeRow';
import { TreeSearch } from './TreeSearch';
import { SwimlaneBar } from './SwimlaneBar';
import { AxisBar } from './AxisBar';
import { Tooltip } from './Tooltip';

export interface LayoutSwimlaneProps {
  /** The root layout tree node to visualize */
  layout: LayoutTreeNode;
  /** Total number of rows in the dataset */
  totalRows: number;
  /** Initially expanded node IDs */
  defaultExpanded?: string[];
  /** Currently selected node ID (controlled) */
  selectedNodeId?: string | null;
  /** Callback when a tree/bar node is clicked */
  onNodeSelect?: (nodeId: string | null) => void;
  /** Display mode: 'schema' shows logical columns, 'layout' shows full layout tree */
  mode?: 'schema' | 'layout';
  /** Minimum width of the swimlane panel */
  swimlaneMinWidth?: number;
  /** Height of the scrollable area */
  height?: number;
}

export function LayoutSwimlane({
  layout,
  totalRows,
  defaultExpanded = [],
  selectedNodeId = null,
  onNodeSelect,
  mode = 'schema',
  swimlaneMinWidth = DEFAULT_SWIMLANE_MIN_WIDTH,
  height,
}: LayoutSwimlaneProps) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(defaultExpanded));
  const [searchQuery, setSearchQuery] = useState('');
  const [tooltip, setTooltip] = useState<{
    node: LayoutTreeNode;
    position: { x: number; y: number };
  } | null>(null);
  const [rulerPosition, setRulerPosition] = useState<{ x: number; row: number } | null>(null);

  const treeScrollRef = useRef<HTMLDivElement>(null);
  const swimlaneScrollRef = useRef<HTMLDivElement>(null);
  const swimlanePanelRef = useRef<HTMLDivElement>(null);
  const axisRef = useRef<HTMLDivElement>(null);

  // Flatten the tree
  const allRows = useMemo(
    () => flattenTree(layout, expanded, null, mode),
    [layout, expanded, mode],
  );

  const visibleRows = useMemo(
    () => filterTreeBySearch(allRows, searchQuery, layout),
    [allRows, searchQuery, layout],
  );

  const toggleExpanded = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleNodeClick = useCallback(
    (nodeId: string) => {
      onNodeSelect?.(selectedNodeId === nodeId ? null : nodeId);
    },
    [selectedNodeId, onNodeSelect],
  );

  const handleTooltip = useCallback(
    (node: LayoutTreeNode | null, position: { x: number; y: number }) => {
      setTooltip(node ? { node, position } : null);
    },
    [],
  );

  // Sync vertical scroll between tree and swimlane
  useEffect(() => {
    const tree = treeScrollRef.current;
    const swimlane = swimlaneScrollRef.current;
    if (!tree || !swimlane) return;

    let syncing = false;
    const syncScroll = (source: HTMLDivElement, target: HTMLDivElement) => () => {
      if (syncing) return;
      syncing = true;
      target.scrollTop = source.scrollTop;
      syncing = false;
    };

    const treeHandler = syncScroll(tree, swimlane);
    const swimlaneHandler = syncScroll(swimlane, tree);
    tree.addEventListener('scroll', treeHandler);
    swimlane.addEventListener('scroll', swimlaneHandler);
    return () => {
      tree.removeEventListener('scroll', treeHandler);
      swimlane.removeEventListener('scroll', swimlaneHandler);
    };
  }, []);

  // Sync horizontal scroll between swimlane and axis
  useEffect(() => {
    const swimlane = swimlaneScrollRef.current;
    const axis = axisRef.current;
    if (!swimlane || !axis) return;

    const handleScroll = () => {
      axis.style.transform = `translateX(-${swimlane.scrollLeft}px)`;
    };
    swimlane.addEventListener('scroll', handleScroll);
    return () => swimlane.removeEventListener('scroll', handleScroll);
  }, []);

  // Ruler mouse tracking
  const handleSwimlaneMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const panel = swimlanePanelRef.current;
      if (!panel) return;
      const rect = panel.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const panelWidth = panel.offsetWidth;
      if (x >= 0 && x <= panelWidth) {
        const rowNum = (x / panelWidth) * totalRows;
        setRulerPosition({ x, row: Math.max(0, Math.min(totalRows, rowNum)) });
      }
    },
    [totalRows],
  );

  const handleSwimlaneMouseLeave = useCallback(() => setRulerPosition(null), []);

  const contentHeight = visibleRows.length * ROW_HEIGHT;

  return (
    <div className="flex flex-col" style={height ? { height } : { flex: 1, minHeight: 0 }}>
      {/* Tree + Swimlane */}
      <div className="flex flex-1 min-h-0">
        {/* Tree panel */}
        <div className="flex-shrink-0 flex flex-col min-h-0" style={{ width: TREE_WIDTH }}>
          <TreeSearch onSearch={setSearchQuery} />
          <div ref={treeScrollRef} className="flex-1 overflow-y-auto overflow-x-hidden">
            <div style={{ height: contentHeight }}>
              {visibleRows.map((row) => (
                <TreeRow
                  key={row.node.id}
                  row={row}
                  isExpanded={expanded.has(row.node.id)}
                  isSelected={selectedNodeId === row.node.id}
                  mode={mode}
                  onToggle={() => toggleExpanded(row.node.id)}
                  onSelect={() => handleNodeClick(row.node.id)}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Swimlane panel */}
        <div
          ref={swimlaneScrollRef}
          className="flex-1 overflow-auto relative"
          onMouseMove={handleSwimlaneMouseMove}
          onMouseLeave={handleSwimlaneMouseLeave}
        >
          <div ref={swimlanePanelRef} style={{ minWidth: swimlaneMinWidth, height: contentHeight }}>
            {visibleRows.map((row) => (
              <SwimlaneRow
                key={row.node.id}
                row={row}
                totalRows={totalRows}
                onHover={handleTooltip}
              />
            ))}

            {rulerPosition && (
              <div
                className="absolute top-0 bottom-0 w-px bg-vortex-black dark:bg-vortex-white opacity-40 pointer-events-none z-[100]"
                style={{ left: rulerPosition.x }}
              />
            )}
          </div>
        </div>
      </div>

      {/* Axis */}
      <div className="flex flex-shrink-0">
        <div className="flex-shrink-0" style={{ width: TREE_WIDTH }} />
        <AxisBar
          totalRows={totalRows}
          swimlaneMinWidth={swimlaneMinWidth}
          rulerPosition={rulerPosition}
          scrollLeft={swimlaneScrollRef.current?.scrollLeft ?? 0}
          containerWidth={swimlaneScrollRef.current?.offsetWidth ?? 0}
          axisRef={axisRef}
        />
      </div>

      {tooltip && <Tooltip node={tooltip.node} position={tooltip.position} />}
    </div>
  );
}

/** A single swimlane row — just height + bar positioning, no decoration */
function SwimlaneRow({
  row,
  totalRows,
  onHover,
}: {
  row: FlattenedRow;
  totalRows: number;
  onHover: (node: LayoutTreeNode | null, position: { x: number; y: number }) => void;
}) {
  return (
    <div className="relative" style={{ height: ROW_HEIGHT }}>
      {row.displayKind !== 'hiddenIndicator' && (
        <SwimlaneBar row={row} totalRows={totalRows} onHover={onHover} />
      )}
    </div>
  );
}

export default LayoutSwimlane;
