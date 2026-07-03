// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { useSelection } from '../../contexts/SelectionContextCore';
import type { LayoutTreeNode } from './types';
import {
  buildSegmentIndex,
  flattenTree,
  findNodeById,
  getNodeDisplayName,
  getDtypeCategory,
  hasExpandableChildren,
  isFlatLayout,
  formatBytes,
  DTYPE_COLORS,
} from './utils';
import { ROW_HEIGHT } from './styles';
import { SwimlaneBar } from './SwimlaneBar';
import { Tooltip } from './Tooltip';

const LABEL_WIDTH = 150;

/**
 * Standalone whole-file column byte-map. Each schema column is a row whose bars
 * show where that column's bytes physically live in the file; clicking a bar (or
 * its label) focuses the column in the shared selection, which drives the detail
 * view. Rows expand/collapse via a chevron in the label gutter — the same model
 * as the tree — lazily attaching a flat column's array-encoding children the
 * first time it is expanded.
 */
export function SwimlaneOverview() {
  const file = useVortexFile();
  const { state: selection, selectNode, hoverNode } = useSelection();
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(['root']));
  const [tooltip, setTooltip] = useState<{
    node: LayoutTreeNode;
    position: { x: number; y: number };
  } | null>(null);

  const totalBytes = file.fileStructure.fileSize;

  const rows = useMemo(
    () =>
      flattenTree(file.layoutTree, expanded, null, 'schema').filter(
        (r) => r.displayKind !== 'hiddenIndicator',
      ),
    [file.layoutTree, expanded],
  );

  const segmentIndex = useMemo(
    () => buildSegmentIndex(file.layoutTree, file.segments),
    [file.layoutTree, file.segments],
  );

  const toggleExpanded = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Expanding a flat layout node lazily attaches its array-encoding children.
  // The tree is hidden in this view, so the swimlane drives the load itself.
  const arrayRequests = useRef(new Set<string>());
  useEffect(() => {
    for (const id of expanded) {
      if (arrayRequests.current.has(id)) continue;
      const node = findNodeById(file.layoutTree, id);
      if (node && isFlatLayout(node) && !node.children.some((c) => c.isArrayNode)) {
        arrayRequests.current.add(id);
        file.expandArrayTree(id);
      }
    }
  }, [expanded, file]);

  const toggleSelect = useCallback(
    (nodeId: string) => selectNode(selection.selectedNodeId === nodeId ? null : nodeId),
    [selection.selectedNodeId, selectNode],
  );

  const handleHover = useCallback(
    (node: LayoutTreeNode | null, position: { x: number; y: number }) => {
      setTooltip(node ? { node, position } : null);
      hoverNode(node ? node.id : null);
    },
    [hoverNode],
  );

  const axisTicks = useMemo(() => {
    if (totalBytes <= 0) return [];
    return Array.from({ length: 6 }, (_, i) => Math.round((i * totalBytes) / 5));
  }, [totalBytes]);

  return (
    <div className="flex flex-col h-full bg-vortex-white dark:bg-vortex-black">
      {/* Column rows. scrollbar-gutter reserves space for the vertical scrollbar so
          the bars stay aligned with the axis below whether or not it's present. */}
      <div
        className="flex-1 overflow-y-auto overflow-x-hidden"
        style={{ scrollbarGutter: 'stable' }}
      >
        {rows.map((row) => {
          const { node, depth } = row;
          const expandable = hasExpandableChildren(node);
          const isExpanded = expanded.has(node.id);
          const isSelected = selection.selectedNodeId === node.id;
          const isHovered = selection.hoveredNodeId === node.id;
          const rowBg = isSelected
            ? 'bg-vortex-light-blue/10'
            : isHovered
              ? 'bg-vortex-light-blue/[0.06]'
              : '';
          return (
            <div
              key={node.id}
              className={`flex items-stretch ${rowBg}`}
              style={{ height: ROW_HEIGHT }}
              onMouseEnter={() => hoverNode(node.id)}
              onMouseLeave={() => hoverNode(null)}
            >
              <div
                className="flex-shrink-0 flex items-center gap-1 pr-2 text-[11px] overflow-hidden whitespace-nowrap cursor-pointer text-vortex-fg-light dark:text-vortex-fg hover:text-vortex-light-blue"
                style={{ width: LABEL_WIDTH, paddingLeft: 6 + depth * 10 }}
                onClick={() => toggleSelect(node.id)}
                title={node.dtype}
              >
                <span
                  className={`text-[8px] w-3 flex-shrink-0 text-vortex-grey-dark ${expandable ? 'cursor-pointer' : ''}`}
                  style={{ opacity: expandable ? 1 : 0 }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (expandable) toggleExpanded(node.id);
                  }}
                >
                  {isExpanded ? '▼' : '▶'}
                </span>
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: DTYPE_COLORS[getDtypeCategory(node.dtype)] }}
                />
                <span className="truncate">{getNodeDisplayName(node)}</span>
              </div>

              <div className="relative flex-1 min-w-0">
                <SwimlaneBar
                  row={row}
                  totalBytes={totalBytes}
                  segmentIndex={segmentIndex}
                  onHover={handleHover}
                  onSelect={() => toggleSelect(node.id)}
                  isSelected={isSelected}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Byte axis — aligned to the bars column via the same label-width offset and
          scrollbar gutter. */}
      <div className="flex flex-shrink-0 h-[22px] border-t border-vortex-grey-light/40 dark:border-white/[0.06]">
        <div className="flex-shrink-0" style={{ width: LABEL_WIDTH }} />
        <div className="relative flex-1 overflow-hidden" style={{ scrollbarGutter: 'stable' }}>
          {axisTicks.map((tick) => (
            <div
              key={tick}
              className="absolute top-1.5 text-[9px] text-vortex-grey-dark"
              style={{ left: `${(tick / totalBytes) * 100}%`, transform: 'translateX(-50%)' }}
            >
              {formatBytes(tick)}
            </div>
          ))}
        </div>
      </div>

      {tooltip && <Tooltip node={tooltip.node} position={tooltip.position} />}
    </div>
  );
}

export default SwimlaneOverview;
