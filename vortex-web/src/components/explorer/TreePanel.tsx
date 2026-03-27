// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useState, useCallback, useMemo } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';
import { flattenTree, filterTreeBySearch } from '../swimlane/utils';
import { TreeRow } from '../swimlane/TreeRow';
import { TreeSearch } from '../swimlane/TreeSearch';

type TreeMode = 'schema' | 'layout';

export function TreePanel() {
  const file = useVortexFile();
  const { state: selection, selectNode, hoverNode } = useSelection();
  const [mode, setMode] = useState<TreeMode>('schema');
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(['root']));
  const [searchQuery, setSearchQuery] = useState('');

  const allRows = useMemo(
    () => flattenTree(file.layoutTree, expanded, null, mode),
    [file.layoutTree, expanded, mode],
  );

  const visibleRows = useMemo(
    () => filterTreeBySearch(allRows, searchQuery, file.layoutTree),
    [allRows, searchQuery, file.layoutTree],
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
      selectNode(selection.selectedNodeId === nodeId ? null : nodeId);
    },
    [selection.selectedNodeId, selectNode],
  );

  return (
    <div className="flex flex-col h-full border-r border-vortex-grey-light/60 dark:border-white/[0.08]">
      {/* Header: mode toggle + search */}
      <div className="flex items-center gap-2 px-2 py-1.5 flex-shrink-0 border-b border-vortex-grey-light/40 dark:border-white/[0.06]">
        <ModeToggle mode={mode} onChange={setMode} />
        <div className="flex-1">
          <TreeSearch onSearch={setSearchQuery} />
        </div>
      </div>

      {/* Tree rows */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {visibleRows.map((row) => (
          <TreeRow
            key={row.node.id}
            row={row}
            isExpanded={expanded.has(row.node.id)}
            isSelected={selection.selectedNodeId === row.node.id}
            mode={mode}
            onToggle={() => toggleExpanded(row.node.id)}
            onSelect={() => handleNodeClick(row.node.id)}
            onHover={hoverNode}
          />
        ))}
      </div>
    </div>
  );
}

/** Subtle segmented toggle for Schema / Layout mode */
function ModeToggle({ mode, onChange }: { mode: TreeMode; onChange: (m: TreeMode) => void }) {
  return (
    <div className="flex rounded-md bg-vortex-grey-lightest dark:bg-white/[0.06] p-0.5 flex-shrink-0">
      {(['schema', 'layout'] as const).map((m) => (
        <button
          key={m}
          className={`px-2 py-0.5 text-[10px] rounded-[3px] transition-colors ${
            mode === m
              ? 'bg-white dark:bg-white/[0.1] text-vortex-fg-light dark:text-vortex-fg shadow-sm'
              : 'text-vortex-grey-dark hover:text-vortex-fg-light dark:hover:text-vortex-fg'
          }`}
          onClick={() => onChange(m)}
        >
          {m === 'schema' ? 'Schema' : 'Layout'}
        </button>
      ))}
    </div>
  );
}
