// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useState } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';
import { LayoutSwimlane } from '../swimlane/LayoutSwimlane';

export function TimelineTab() {
  const file = useVortexFile();
  const { state: selection, selectNode } = useSelection();
  const [mode, setMode] = useState<'schema' | 'layout'>('schema');

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Mode toggle */}
      <div className="flex items-center gap-2 px-4 py-2 flex-shrink-0">
        <button
          className={`px-3 py-1 text-xs rounded ${
            mode === 'schema'
              ? 'bg-vortex-light-blue/15 text-vortex-light-blue font-medium'
              : 'text-vortex-grey-dark hover:text-vortex-fg-light dark:hover:text-vortex-fg'
          }`}
          onClick={() => setMode('schema')}
        >
          Schema
        </button>
        <button
          className={`px-3 py-1 text-xs rounded ${
            mode === 'layout'
              ? 'bg-vortex-light-blue/15 text-vortex-light-blue font-medium'
              : 'text-vortex-grey-dark hover:text-vortex-fg-light dark:hover:text-vortex-fg'
          }`}
          onClick={() => setMode('layout')}
        >
          Layout
        </button>
      </div>

      <LayoutSwimlane
        layout={file.layoutTree}
        totalRows={file.rowCount}
        mode={mode}
        selectedNodeId={selection.selectedNodeId}
        onNodeSelect={selectNode}
        defaultExpanded={['root']}
      />
    </div>
  );
}
