// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useSelection } from '../../contexts/SelectionContext';
import { getNodeDisplayName } from '../swimlane/utils';

export function DataPreview() {
  const { state: selection } = useSelection();

  return (
    <div className="h-full flex flex-col bg-vortex-white dark:bg-vortex-black">
      <div className="flex items-center px-3 py-1 border-b border-vortex-grey-lightest dark:border-vortex-grey-dark/30 flex-shrink-0">
        <span className="text-[10px] font-medium text-vortex-grey-dark uppercase tracking-wider">
          Preview
        </span>
        {selection.selectedNode && (
          <span className="ml-2 text-[10px] text-vortex-light-blue">
            {getNodeDisplayName(selection.selectedNode)}
          </span>
        )}
      </div>
      <div className="flex-1 flex items-center justify-center text-[11px] text-vortex-grey-dark">
        {selection.selectedNode
          ? 'Data preview requires WASM integration'
          : 'Select a column to preview data'}
      </div>
    </div>
  );
}
