// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useState } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';
import {
  getNodeDisplayName,
  findPathToNode,
  getDtypeCategory,
  shortEncoding,
  DTYPE_COLORS,
} from '../swimlane/utils';
import { SummaryPane } from './SummaryPane';
import { ArraySummaryPane } from './ArraySummaryPane';
import { EncodingPane } from './EncodingPane';
import { SegmentsPane } from './SegmentsPane';
import { TreemapPane } from './TreemapPane';
import { BuffersPane } from './BuffersPane';

type TabId = 'encoding' | 'segments' | 'treemap' | 'buffers';

interface TabDef {
  id: TabId;
  label: string;
}

export function DetailPanel() {
  const file = useVortexFile();
  const { state: selection, selectNode, hoverNode } = useSelection();
  const [activeTab, setActiveTab] = useState<TabId>('segments');

  const isArrayNode = selection.selectedNode?.isArrayNode ?? false;

  const tabs = useMemo<TabDef[]>(() => {
    const result: TabDef[] = [];
    if (selection.selectedNode) {
      if (isArrayNode) {
        result.push({ id: 'treemap', label: 'Treemap' });
        if ((selection.selectedNode.bufferLengths ?? []).length > 0) {
          result.push({ id: 'buffers', label: 'Buffers' });
        }
      } else {
        result.push({ id: 'treemap', label: 'Treemap' });
        result.push({ id: 'segments', label: 'Segments' });
        if (selection.selectedNode.children.length === 0) {
          result.push({ id: 'encoding', label: 'Encoding' });
        }
      }
    }
    return result;
  }, [selection.selectedNode, isArrayNode]);

  const selectedPath = useMemo(() => {
    if (!selection.selectedNodeId) return [];
    return findPathToNode(file.layoutTree, selection.selectedNodeId);
  }, [file.layoutTree, selection.selectedNodeId]);

  const hoveredPath = useMemo(() => {
    if (!selection.hoveredNodeId) return [];
    return findPathToNode(file.layoutTree, selection.hoveredNodeId);
  }, [file.layoutTree, selection.hoveredNodeId]);

  const breadcrumb = hoveredPath.length > 0 ? hoveredPath : selectedPath;

  const selectedIdSet = useMemo(() => new Set(selectedPath.map((n) => n.id)), [selectedPath]);
  const isHoverBreadcrumb = hoveredPath.length > 0;

  const currentTab = tabs.find((t) => t.id === activeTab) ? activeTab : tabs[0]?.id;

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
              const isShared = isHoverBreadcrumb && selectedIdSet.has(node.id);
              const dimClass = isHoverBreadcrumb && !isShared ? 'opacity-50' : '';
              const prevNode = i > 0 ? breadcrumb[i - 1] : null;
              const isArrayBoundary = node.isArrayNode && prevNode && !prevNode.isArrayNode;
              return (
                <span key={node.id} className={`flex items-center gap-0.5 min-w-0 ${dimClass}`}>
                  {isArrayBoundary && (
                    <span
                      className="opacity-40 flex-shrink-0 mx-0.5"
                      title={`array: ${shortEncoding(prevNode.encoding)}`}
                    >
                      ›
                    </span>
                  )}
                  {i > 0 && !isArrayBoundary && <span className="opacity-40 flex-shrink-0">/</span>}
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
            {(() => {
              const tip = breadcrumb[breadcrumb.length - 1];
              if (!tip) return null;
              const cat = getDtypeCategory(tip.dtype);
              return (
                <span
                  className="ml-1.5 px-1 py-0 rounded text-[9px] font-medium text-white flex-shrink-0"
                  style={{ backgroundColor: DTYPE_COLORS[cat] }}
                  title={tip.dtype}
                >
                  {cat}
                </span>
              );
            })()}
          </div>
        )}
      </div>

      {/* Main content: tab content (left) + summary sidebar (right) */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Tab content */}
        <div className="flex-1 overflow-auto min-w-0">
          {currentTab === 'encoding' && selection.selectedNode && (
            <div className="p-2.5">
              <EncodingPane node={selection.selectedNode} />
            </div>
          )}
          {currentTab === 'segments' && selection.selectedNode && (
            <SegmentsPane node={selection.selectedNode} segments={file.segments} />
          )}
          {currentTab === 'treemap' && selection.selectedNode && (
            <TreemapPane
              node={selection.selectedNode}
              segments={file.segments}
              onSelectNode={selectNode}
              onHoverNode={hoverNode}
            />
          )}
          {currentTab === 'buffers' && selection.selectedNode && (
            <BuffersPane node={selection.selectedNode} />
          )}
          {!currentTab && !selection.selectedNode && (
            <div className="p-2.5 text-xs text-vortex-grey-dark">
              Select a node to view details.
            </div>
          )}
        </div>

        {/* Summary sidebar — always visible */}
        <div className="w-[180px] flex-shrink-0 overflow-y-auto border-l border-vortex-grey-light/40 dark:border-white/[0.06] p-2.5">
          {isArrayNode && selection.selectedNode ? (
            <ArraySummaryPane node={selection.selectedNode} />
          ) : (
            <SummaryPane node={selection.selectedNode} file={file} />
          )}
        </div>
      </div>
    </div>
  );
}
