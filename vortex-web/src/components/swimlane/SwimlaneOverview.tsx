// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { useSelection } from '../../contexts/SelectionContextCore';
import type { LayoutTreeNode } from './types';
import {
  buildSegmentIndex,
  flattenTree,
  findNodeById,
  getNodeDisplayName,
  dtypeClass,
  hasExpandableChildren,
  isFlatLayout,
  formatBytes,
} from './utils';
import { Icon } from '../Icon';
import { SwimlaneBar } from './SwimlaneBar';
import { Tooltip } from './Tooltip';

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

  // Ticks on a round step (1, 2, or 5 times a power of ten) near a fifth of the file, then the
  // file's size at the end. A round tick too close to the end gives way to the size.
  const axisTicks = useMemo(() => {
    if (totalBytes <= 0) return [];
    const raw = totalBytes / 5;
    const power = 10 ** Math.floor(Math.log10(raw));
    const m = raw / power;
    const step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * power;
    const ticks: number[] = [];
    for (let t = 0; t <= totalBytes * 0.85; t += step) ticks.push(t);
    return [...ticks, totalBytes];
  }, [totalBytes]);

  return (
    <section className="vx-card vx-pane x-view" aria-label="Swimlane">
      <div className="x-lanes">
        {/* The rows reserve the scrollbar's gutter, as the axis does, so the bars and the
            ticks line up whether or not the rows scroll. */}
        <ul className="x-lanes-rows" aria-label="Columns by place in the file">
          {rows.map((row) => {
            const { node, depth } = row;
            const expandable = hasExpandableChildren(node);
            const isExpanded = expanded.has(node.id);
            const isSelected = selection.selectedNodeId === node.id;
            const isHovered = selection.hoveredNodeId === node.id;
            return (
              <li
                key={node.id}
                className={`x-lane${isSelected ? ' is-selected' : ''}${isHovered ? ' is-hovered' : ''}`}
                onPointerEnter={() => hoverNode(node.id)}
                onPointerLeave={() => hoverNode(null)}
              >
                <span className="x-lane-name" style={{ '--vx-depth': depth } as CSSProperties}>
                  <button
                    type="button"
                    className={`x-lane-toggle${expandable ? '' : ' is-leaf'}`}
                    aria-expanded={expandable ? isExpanded : undefined}
                    aria-label={`${isExpanded ? 'Close' : 'Open'} ${getNodeDisplayName(node)}`}
                    tabIndex={expandable ? 0 : -1}
                    disabled={!expandable}
                    onClick={() => toggleExpanded(node.id)}
                  >
                    <Icon name="chevron_right" className="vx-tree-toggle" />
                  </button>
                  <button
                    type="button"
                    className="x-lane-select"
                    aria-pressed={isSelected}
                    title={node.dtype}
                    onClick={() => toggleSelect(node.id)}
                  >
                    <span className={`vx-tree-dot ${dtypeClass(node.dtype)}`} />
                    <span>{getNodeDisplayName(node)}</span>
                  </button>
                </span>
                <span className="x-lane-bar">
                  <SwimlaneBar
                    row={row}
                    totalBytes={totalBytes}
                    segmentIndex={segmentIndex}
                    onHover={handleHover}
                    onSelect={() => toggleSelect(node.id)}
                  />
                </span>
              </li>
            );
          })}
        </ul>

        <div className="x-axis vx-label-sm" aria-hidden="true">
          <span />
          <span className="x-axis-ticks">
            {axisTicks.map((tick) => (
              <span key={tick} style={{ left: `${(tick / totalBytes) * 100}%` }}>
                {formatBytes(tick)}
              </span>
            ))}
          </span>
        </div>
      </div>

      {tooltip && <Tooltip node={tooltip.node} position={tooltip.position} />}
    </section>
  );
}

export default SwimlaneOverview;
