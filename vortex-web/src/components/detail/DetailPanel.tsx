// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useState } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';
import { getNodeDisplayName, findPathToNode } from '../swimlane/utils';
import { SummaryPane } from './SummaryPane';
import { EncodingPane } from './EncodingPane';
import { SegmentsPane } from './SegmentsPane';
import { TreemapPane } from './TreemapPane';

type TabId = 'summary' | 'encoding' | 'segments' | 'treemap';

interface TabDef {
  id: TabId;
  label: string;
}

export function DetailPanel() {
  const file = useVortexFile();
  const { state: selection, selectNode, hoverNode } = useSelection();
  const [activeTab, setActiveTab] = useState<TabId>('summary');

  const tabs = useMemo<TabDef[]>(() => {
    const result: TabDef[] = [{ id: 'summary', label: 'Summary' }];
    if (selection.selectedNode) {
      result.push({ id: 'segments', label: 'Segments' });
      result.push({ id: 'treemap', label: 'Treemap' });
      if (selection.selectedNode.children.length === 0) {
        result.push({ id: 'encoding', label: 'Encoding' });
      }
    }
    return result;
  }, [selection.selectedNode]);

  const selectedPath = useMemo(() => {
    if (!selection.selectedNodeId) return [];
    return findPathToNode(file.layoutTree, selection.selectedNodeId);
  }, [file.layoutTree, selection.selectedNodeId]);

  const hoveredPath = useMemo(() => {
    if (!selection.hoveredNodeId) return [];
    return findPathToNode(file.layoutTree, selection.hoveredNodeId);
  }, [file.layoutTree, selection.hoveredNodeId]);

  // Show hovered path when hovering, otherwise the selected path.
  const breadcrumb = hoveredPath.length > 0 ? hoveredPath : selectedPath;

  // When hovering, the portion of the path that overlaps with the selected path stays bright.
  const selectedIdSet = useMemo(
    () => new Set(selectedPath.map((n) => n.id)),
    [selectedPath],
  );
  const isHoverBreadcrumb = hoveredPath.length > 0;

  const currentTab = tabs.find((t) => t.id === activeTab) ? activeTab : 'summary';

  return (
    <div className="flex flex-col flex-1 min-h-0 h-full bg-vortex-white dark:bg-vortex-black">
      {/* Breadcrumb + tab bar */}
      <div className="flex items-center border-b border-vortex-grey-light/40 dark:border-white/[0.06] px-2 flex-shrink-0">
        {/* Tabs */}
        <div className="flex">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`px-2.5 py-1 text-[10px] font-medium border-b-2 transition-colors ${
                currentTab === tab.id
                  ? 'border-vortex-light-blue text-vortex-light-blue'
                  : 'border-transparent text-vortex-grey-dark hover:text-vortex-fg-light dark:hover:text-vortex-fg'
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
              // When showing hover path: nodes shared with selected path are bright, others are dim.
              // When showing selected path: last node is bright, others are clickable.
              const isShared = isHoverBreadcrumb && selectedIdSet.has(node.id);
              const dimClass = isHoverBreadcrumb && !isShared ? 'opacity-50' : '';
              return (
                <span key={node.id} className={`flex items-center gap-0.5 min-w-0 ${dimClass}`}>
                  {i > 0 && <span className="opacity-40 flex-shrink-0">/</span>}
                  {isLast && !isHoverBreadcrumb ? (
                    <span className="text-vortex-fg-light dark:text-vortex-fg truncate">
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
      <div className="flex-1 overflow-auto">
        {currentTab === 'summary' && (
          <div className="p-2.5">
            <SummaryPane node={selection.selectedNode} file={file} />
          </div>
        )}
        {currentTab === 'encoding' && selection.selectedNode && (
          <div className="p-2.5">
            <EncodingPane node={selection.selectedNode} />
          </div>
        )}
        {currentTab === 'segments' && selection.selectedNode && (
          <SegmentsPane node={selection.selectedNode} segments={file.segments} />
        )}
        {currentTab === 'treemap' && selection.selectedNode && (
          <TreemapPane node={selection.selectedNode} segments={file.segments} onSelectNode={selectNode} onHoverNode={hoverNode} />
        )}
      </div>
    </div>
  );
}
