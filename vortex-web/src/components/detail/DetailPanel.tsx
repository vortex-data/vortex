// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useId, useMemo, useState, type KeyboardEvent } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { useSelection } from '../../contexts/SelectionContextCore';
import {
  getNodeDisplayName,
  findPathToNode,
  getDtypeCategory,
  shortEncoding,
  dtypeClass,
} from '../swimlane/utils';
import { EncodingPane } from './EncodingPane';
import { SegmentsPane } from './SegmentsPane';
import { BlockTreemap } from '../explorer/BlockTreemap';
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

  // Prefer the active tab; otherwise fall back to the first available tab.
  const currentTab = tabs.find((t) => t.id === activeTab)?.id ?? tabs[0]?.id;

  const ids = useId();
  const tabId = (id: TabId) => `${ids}-tab-${id}`;
  const panelId = `${ids}-panel`;

  // Left and Right move between the tabs, Home and End to the ends: the WAI-ARIA tabs pattern.
  const onTabKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const at = tabs.findIndex((t) => t.id === currentTab);
    const next = {
      ArrowRight: at + 1,
      ArrowLeft: at - 1,
      Home: 0,
      End: tabs.length - 1,
    }[e.key];
    if (next === undefined || tabs.length === 0) return;
    e.preventDefault();
    const tab = tabs[(next + tabs.length) % tabs.length]!;
    setActiveTab(tab.id);
    document.getElementById(tabId(tab.id))?.focus();
  };

  const tip = breadcrumb[breadcrumb.length - 1];

  return (
    <section className="vx-card vx-pane x-view" aria-label="Selected node">
      <div className="x-view-head">
        {tabs.length > 0 && (
          <div className="vx-tabs-list" role="tablist" aria-label="Views" onKeyDown={onTabKey}>
            {tabs.map((tab) => (
              <button
                type="button"
                role="tab"
                key={tab.id}
                id={tabId(tab.id)}
                aria-selected={currentTab === tab.id}
                aria-controls={panelId}
                tabIndex={currentTab === tab.id ? 0 : -1}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        )}

        {breadcrumb.length > 1 && (
          <nav
            className={`vx-label-sm x-crumbs${isHoverBreadcrumb ? ' is-preview' : ''}`}
            aria-label="Path"
          >
            <ol>
              {breadcrumb.map((node, i) => {
                const isLast = i === breadcrumb.length - 1;
                const prevNode = i > 0 ? breadcrumb[i - 1] : null;
                const isArrayBoundary = node.isArrayNode && prevNode && !prevNode.isArrayNode;
                const isShared = isHoverBreadcrumb && selectedIdSet.has(node.id);
                return (
                  <li
                    key={node.id}
                    className={isArrayBoundary ? 'is-array' : undefined}
                    {...(isArrayBoundary && prevNode
                      ? { title: `array: ${shortEncoding(prevNode.encoding)}` }
                      : {})}
                  >
                    {isLast && !isHoverBreadcrumb ? (
                      <span aria-current="location">{getNodeDisplayName(node)}</span>
                    ) : (
                      <button
                        type="button"
                        onClick={() => selectNode(node.id)}
                        {...(isShared ? { 'aria-current': 'location' as const } : {})}
                      >
                        {getNodeDisplayName(node)}
                      </button>
                    )}
                  </li>
                );
              })}
            </ol>
            {tip && (
              <span className={`x-dtype ${dtypeClass(tip.dtype)}`} title={tip.dtype}>
                {getDtypeCategory(tip.dtype)}
              </span>
            )}
          </nav>
        )}
      </div>

      <div
        className="vx-pane-body"
        {...(currentTab
          ? { role: 'tabpanel', id: panelId, 'aria-labelledby': tabId(currentTab) }
          : {})}
      >
        {currentTab === 'encoding' && selection.selectedNode && (
          <EncodingPane node={selection.selectedNode} />
        )}
        {currentTab === 'segments' && selection.selectedNode && (
          <SegmentsPane node={selection.selectedNode} segments={file.segments} />
        )}
        {currentTab === 'treemap' && selection.selectedNode && (
          <BlockTreemap
            root={file.layoutTree}
            segments={file.segments}
            fileSize={file.fileSize}
            selectedNodeId={selection.selectedNodeId}
            hoveredNodeId={selection.hoveredNodeId}
            onSelectNode={selectNode}
            onHoverNode={hoverNode}
            onExpand={file.expandArrayTree}
          />
        )}
        {currentTab === 'buffers' && selection.selectedNode && (
          <BuffersPane node={selection.selectedNode} />
        )}
        {!selection.selectedNode && (
          <p className="vx-label x-empty">
            Select a column or a layout in the tree to see its treemap, segments, and encoding.
          </p>
        )}
      </div>
    </section>
  );
}
