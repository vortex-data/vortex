// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { TreePanel } from './TreePanel';
import { FileMap } from './FileMap';
import { DataPreview } from './DataPreview';
import { DetailPanel } from '../detail/DetailPanel';

/**
 * Main explorer area: tree panel (left) | detail + filemap + preview (right).
 *
 * Layout constraints:
 *  - This component is flex-1 inside a h-screen column, so it fills remaining height.
 *  - Left panel (tree): fixed width, full height, scrolls internally.
 *  - Right panel: flex column filling remaining width.
 *    - DetailPanel: flex-1 (takes remaining vertical space), scrolls internally.
 *    - FileMap: fixed height strip.
 *    - DataPreview: fixed height bottom section.
 */
export function MainArea() {
  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      {/* Left: tree panel — full height, fixed width */}
      <div className="w-[260px] flex-shrink-0 h-full overflow-hidden">
        <TreePanel />
      </div>

      {/* Right: detail pane, file map, data preview — stacked vertically */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Detail pane — fills available vertical space, scrolls internally */}
        <DetailPanel />

        {/* File map strip */}
        <div className="flex-shrink-0">
          <FileMap />
        </div>

        {/* Data preview — fixed height bottom section */}
        <div className="h-[200px] flex-shrink-0 overflow-hidden border-t border-vortex-grey-lightest dark:border-vortex-grey-dark/30">
          <DataPreview />
        </div>
      </div>
    </div>
  );
}
