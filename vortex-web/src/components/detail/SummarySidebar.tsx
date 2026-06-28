// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { useSelection } from '../../contexts/SelectionContextCore';
import { SummaryPane } from './SummaryPane';
import { ArraySummaryPane } from './ArraySummaryPane';

/**
 * Right-hand summary sidebar shared by the details and swimlane views. Shows the
 * selected node's summary (array-aware), falling back to the whole-file summary
 * when nothing is selected.
 */
export function SummarySidebar() {
  const file = useVortexFile();
  const { state: selection } = useSelection();
  const isArrayNode = selection.selectedNode?.isArrayNode ?? false;

  return (
    <div className="w-[180px] flex-shrink-0 overflow-y-auto border-l border-vortex-grey-light/40 dark:border-white/[0.06] p-2.5">
      {isArrayNode && selection.selectedNode ? (
        <ArraySummaryPane node={selection.selectedNode} />
      ) : (
        <SummaryPane node={selection.selectedNode} file={file} />
      )}
    </div>
  );
}
