// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useState, useCallback, useMemo, useEffect, useRef, type KeyboardEvent } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { useSelection } from '../../contexts/SelectionContextCore';
import {
  flattenTree,
  filterTreeBySearch,
  findPathToNode,
  isFlatLayout,
  findNodeById,
  formatBytes,
  collectSubtreeSegments,
  hasExpandableChildren,
} from '../swimlane/utils';
import { arraySubtreeBytes } from './physicalStats';
import { Switch } from '../Switch';
import { TreeRow } from '../swimlane/TreeRow';
import { TreeSearch } from '../swimlane/TreeSearch';

type TreeMode = 'schema' | 'layout';

const MODES = [
  { value: 'schema', label: 'Schema' },
  { value: 'layout', label: 'Layout' },
] as const;

export function TreePanel() {
  const file = useVortexFile();
  const { state: selection, selectNode, hoverNode } = useSelection();
  const [mode, setMode] = useState<TreeMode>('schema');
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(['root']));
  const [searchQuery, setSearchQuery] = useState('');

  // Auto-expand ancestors so the selected node is visible, including synthetic group nodes.
  useEffect(() => {
    if (!selection.selectedNodeId) return;
    const path = findPathToNode(file.layoutTree, selection.selectedNodeId);
    if (path.length === 0) return;

    setExpanded((prev) => {
      let next = new Set(prev);
      for (const node of path) next.add(node.id);

      // Iteratively expand group nodes that contain the target until it's visible.
      for (let attempt = 0; attempt < 5; attempt++) {
        const rows = flattenTree(file.layoutTree, next, null, mode);
        if (rows.some((r) => r.node.id === selection.selectedNodeId)) break;

        // Find group rows whose grouped children include the target.
        let changed = false;
        for (const row of rows) {
          if (row.displayKind === 'group' && row.groupedChildren) {
            if (row.groupedChildren.some((c) => c.id === selection.selectedNodeId)) {
              next = new Set(next);
              next.add(row.node.id);
              changed = true;
            }
          }
        }
        if (!changed) break;
      }

      return next;
    });
  }, [selection.selectedNodeId, file.layoutTree, mode]);

  // Scroll the selected node into view only when the selection changes.
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const lastScrolledTo = useRef<string | null>(null);
  useEffect(() => {
    if (!selection.selectedNodeId || selection.selectedNodeId === lastScrolledTo.current) return;
    lastScrolledTo.current = selection.selectedNodeId;
    // Defer to let the DOM update after expansion.
    requestAnimationFrame(() => {
      const container = scrollContainerRef.current;
      if (!container) return;
      const el = container.querySelector(
        `[data-node-id="${CSS.escape(selection.selectedNodeId!)}"]`,
      );
      el?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    });
  }, [selection.selectedNodeId, expanded]);

  const allRows = useMemo(
    () => flattenTree(file.layoutTree, expanded, null, mode),
    [file.layoutTree, expanded, mode],
  );

  const visibleRows = useMemo(
    () => filterTreeBySearch(allRows, searchQuery, file.layoutTree),
    [allRows, searchQuery, file.layoutTree],
  );

  // Compute the set of ancestor node IDs for the hovered node (for dim highlighting in tree)
  const hoveredAncestorIds = useMemo(() => {
    if (!selection.hoveredNodeId) return new Set<string>();
    const path = findPathToNode(file.layoutTree, selection.hoveredNodeId);
    // Exclude the hovered node itself — it gets full highlight
    const ids = new Set(path.map((n) => n.id));
    ids.delete(selection.hoveredNodeId);
    return ids;
  }, [selection.hoveredNodeId, file.layoutTree]);

  // When a flat layout node is expanded, lazily attach array encoding children.
  // Track which nodes we've already requested to avoid re-triggering on tree updates.
  const expandedArrayRequests = useRef(new Set<string>());
  useEffect(() => {
    for (const id of expanded) {
      if (expandedArrayRequests.current.has(id)) continue;
      const node = findNodeById(file.layoutTree, id);
      if (node && isFlatLayout(node) && !node.children.some((c) => c.isArrayNode)) {
        expandedArrayRequests.current.add(id);
        file.expandArrayTree(id);
      }
    }
  }, [expanded, file]);

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
      selectNode(selection.selectedNodeId === nodeId ? null : nodeId);
    },
    [selection.selectedNodeId, selectNode],
  );

  // Each node's data size, for the rows that show.
  const segmentBytes = useMemo(
    () => new Map(file.segments.map((s) => [s.index, s.byteLength])),
    [file.segments],
  );
  const sizes = useMemo(() => {
    const out = new Map<string, string>();
    for (const { node, displayKind } of visibleRows) {
      if (displayKind === 'hiddenIndicator') continue;
      const bytes = node.isArrayNode
        ? arraySubtreeBytes(node)
        : [...new Set(collectSubtreeSegments(node))].reduce(
            (sum, id) => sum + (segmentBytes.get(id) ?? 0),
            0,
          );
      out.set(node.id, formatBytes(bytes));
    }
    return out;
  }, [visibleRows, segmentBytes]);

  // The tree is one tab stop: the selected node, else the first.
  const focusId =
    visibleRows.find((r) => r.node.id === selection.selectedNodeId)?.node.id ??
    visibleRows[0]?.node.id;

  const focusNode = useCallback((id: string) => {
    requestAnimationFrame(() => {
      scrollContainerRef.current
        ?.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(id)}"]`)
        ?.focus();
    });
  }, []);

  // The keys of the WAI-ARIA tree pattern, as the design system's treeKey gives them. Every row
  // drawn is visible, so Up and Down step through the rows.
  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      const el = (e.target as Element).closest<HTMLElement>('[data-node-id]');
      const at = visibleRows.findIndex((r) => r.node.id === el?.dataset.nodeId);
      const row = visibleRows[at];
      if (!row) return;
      const go = (i: number) => {
        const target = visibleRows[Math.max(0, Math.min(visibleRows.length - 1, i))];
        if (!target) return;
        // A group row ("···") and the hidden-rows line are not nodes of the file: focus them only.
        if (target.displayKind === 'normal') selectNode(target.node.id);
        focusNode(target.node.id);
      };
      const parent = row.displayKind !== 'hiddenIndicator' && hasExpandableChildren(row.node);
      const open = expanded.has(row.node.id);
      switch (e.key) {
        case 'ArrowDown':
          go(at + 1);
          break;
        case 'ArrowUp':
          go(at - 1);
          break;
        case 'Home':
          go(0);
          break;
        case 'End':
          go(visibleRows.length - 1);
          break;
        case 'ArrowRight':
          if (!parent) return;
          if (!open) toggleExpanded(row.node.id);
          else if ((visibleRows[at + 1]?.depth ?? -1) > row.depth) go(at + 1);
          break;
        case 'ArrowLeft': {
          if (parent && open) {
            toggleExpanded(row.node.id);
            break;
          }
          let j = at - 1;
          while (j >= 0 && (visibleRows[j]?.depth ?? 0) >= row.depth) j--;
          if (j < 0) return;
          go(j);
          break;
        }
        case 'Enter':
        case ' ':
          if (!parent) return;
          toggleExpanded(row.node.id);
          break;
        default:
          return;
      }
      e.preventDefault();
    },
    [visibleRows, expanded, selectNode, focusNode, toggleExpanded],
  );

  return (
    <section
      className="vx-card vx-pane x-tree-pane"
      aria-label={mode === 'schema' ? 'Schema' : 'Layout'}
    >
      <div className="vx-pane-head">
        <Switch label="Tree view" value={mode} options={MODES} onChange={setMode} />
        <TreeSearch onSearch={setSearchQuery} />
      </div>
      <div ref={scrollContainerRef} className="vx-pane-body">
        <div
          className="vx-tree"
          role="tree"
          aria-label={`${mode === 'schema' ? 'Schema' : 'Layout'} of ${file.fileName}`}
          onKeyDown={onKeyDown}
        >
          {visibleRows.map((row) => (
            <TreeRow
              key={row.node.id}
              row={row}
              isExpanded={expanded.has(row.node.id)}
              isSelected={selection.selectedNodeId === row.node.id}
              isHovered={selection.hoveredNodeId === row.node.id}
              isHoveredAncestor={hoveredAncestorIds.has(row.node.id)}
              mode={mode}
              tabIndex={row.node.id === focusId ? 0 : -1}
              onToggle={() => toggleExpanded(row.node.id)}
              onSelect={() =>
                row.displayKind === 'group'
                  ? toggleExpanded(row.node.id)
                  : handleNodeClick(row.node.id)
              }
              onHover={hoverNode}
              {...(sizes.has(row.node.id) ? { size: sizes.get(row.node.id)! } : {})}
            />
          ))}
        </div>
      </div>
      <div className="vx-pane-foot vx-label-sm">
        {visibleRows.length} {visibleRows.length === 1 ? 'node' : 'nodes'} shown
        {searchQuery ? ` for “${searchQuery}”` : ''}
      </div>
    </section>
  );
}
