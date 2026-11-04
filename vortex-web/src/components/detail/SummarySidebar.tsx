// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { useSelection } from '../../contexts/SelectionContextCore';
import { SummaryPane } from './SummaryPane';
import { ArraySummaryPane } from './ArraySummaryPane';

/**
 * The summary pane beside the view, in both views: the selected node's facts (array-aware), or
 * the whole file's when nothing is selected.
 */
export function SummarySidebar() {
  const file = useVortexFile();
  const { state: selection } = useSelection();
  const node = selection.selectedNode;

  return (
    <section className="vx-card vx-pane x-side" aria-label="Summary">
      {node?.isArrayNode ? (
        <ArraySummaryPane node={node} />
      ) : (
        <SummaryPane node={node} file={file} />
      )}
    </section>
  );
}
