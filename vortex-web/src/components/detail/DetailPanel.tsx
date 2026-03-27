// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useState } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';
import { getNodeDisplayName, findPathToNode } from '../swimlane/utils';
import { SummaryPane } from './SummaryPane';
import { EncodingPane } from './EncodingPane';
import { SegmentsPane } from './SegmentsPane';

type TabId = 'summary' | 'encoding' | 'segments';

interface TabDef {
  id: TabId;
  label: string;
}

export function DetailPanel() {
  const file = useVortexFile();
  const { state: selection, selectNode } = useSelection();
  const [activeTab, setActiveTab] = useState<TabId>('summary');

  const tabs = useMemo<TabDef[]>(() => {
    const result: TabDef[] = [{ id: 'summary', label: 'Summary' }];
    if (selection.selectedNode) {
      result.push({ id: 'segments', label: 'Segments' });
      if (selection.selectedNode.children.length === 0) {
        result.push({ id: 'encoding', label: 'Encoding' });
      }
    }
    return result;
  }, [selection.selectedNode]);

  const breadcrumb = useMemo(() => {
    if (!selection.selectedNodeId) return [];
    return findPathToNode(file.layoutTree, selection.selectedNodeId);
  }, [file.layoutTree, selection.selectedNodeId]);

  const currentTab = tabs.find((t) => t.id === activeTab) ? activeTab : 'summary';

  return (
    <div className="flex flex-col flex-1 min-h-0 h-full bg-vortex-white dark:bg-vortex-black">
      {/* Breadcrumb + tab bar */}
      <div className="flex items-center border-b border-vortex-grey-lightest dark:border-vortex-grey-dark/30 px-2 flex-shrink-0">
        {/* Tabs */}
        <div className="flex">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`px-2.5 py-1 text-[10px] font-medium border-b-2 transition-colors ${
                currentTab === tab.id
                  ? 'border-vortex-light-blue text-vortex-light-blue'
                  : 'border-transparent text-vortex-grey-dark hover:text-vortex-black dark:hover:text-vortex-white'
              }`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Breadcrumb — right of tabs */}
        {breadcrumb.length > 1 && (
          <div className="flex items-center gap-0.5 ml-auto text-[10px] text-vortex-grey-dark overflow-hidden">
            {breadcrumb.map((node, i) => {
              const isLast = i === breadcrumb.length - 1;
              return (
                <span key={node.id} className="flex items-center gap-0.5 min-w-0">
                  {i > 0 && <span className="opacity-40 flex-shrink-0">/</span>}
                  {isLast ? (
                    <span className="text-vortex-black dark:text-vortex-white truncate">
                      {getNodeDisplayName(node)}
                    </span>
                  ) : (
                    <button
                      className="hover:text-vortex-light-blue truncate"
                      onClick={() => selectNode(node.id)}
                    >
                      {getNodeDisplayName(node)}
                    </button>
                  )}
                </span>
              );
            })}
          </div>
        )}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto p-2.5">
        {currentTab === 'summary' && <SummaryPane node={selection.selectedNode} file={file} />}
        {currentTab === 'encoding' && selection.selectedNode && (
          <EncodingPane node={selection.selectedNode} />
        )}
        {currentTab === 'segments' && selection.selectedNode && (
          <SegmentsPane node={selection.selectedNode} segments={file.segments} />
        )}
      </div>
    </div>
  );
}
