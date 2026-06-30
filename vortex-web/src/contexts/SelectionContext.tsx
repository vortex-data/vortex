// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useReducer, useCallback, type ReactNode } from 'react';
import type { LayoutTreeNode } from '../components/swimlane/types';
import { findNodeById } from '../components/swimlane/utils';
import { SelectionContext } from './SelectionContextCore';

export interface SelectionState {
  selectedNodeId: string | null;
  selectedNode: LayoutTreeNode | null;
  hoveredNodeId: string | null;
  hoveredNode: LayoutTreeNode | null;
  hoveredSegmentIndex: number | null;
  selectedSegmentIndex: number | null;
}

type SelectionAction =
  | { type: 'selectNode'; nodeId: string | null; tree: LayoutTreeNode }
  | { type: 'hoverNode'; nodeId: string | null; tree: LayoutTreeNode }
  | { type: 'hoverSegment'; index: number | null }
  | { type: 'selectSegment'; index: number | null }
  | { type: 'clearSelection' };

function selectionReducer(state: SelectionState, action: SelectionAction): SelectionState {
  switch (action.type) {
    case 'selectNode': {
      const nodeId = action.nodeId;
      const node = nodeId ? findNodeById(action.tree, nodeId) : null;
      return { ...state, selectedNodeId: nodeId, selectedNode: node, selectedSegmentIndex: null };
    }
    case 'hoverNode': {
      const nodeId = action.nodeId;
      const node = nodeId ? findNodeById(action.tree, nodeId) : null;
      return { ...state, hoveredNodeId: nodeId, hoveredNode: node, hoveredSegmentIndex: null };
    }
    case 'hoverSegment':
      return { ...state, hoveredSegmentIndex: action.index };
    case 'selectSegment':
      return { ...state, selectedSegmentIndex: action.index };
    case 'clearSelection':
      return {
        selectedNodeId: null,
        selectedNode: null,
        hoveredNodeId: null,
        hoveredNode: null,
        hoveredSegmentIndex: null,
        selectedSegmentIndex: null,
      };
  }
}

const initialState: SelectionState = {
  selectedNodeId: null,
  selectedNode: null,
  hoveredNodeId: null,
  hoveredNode: null,
  hoveredSegmentIndex: null,
  selectedSegmentIndex: null,
};

export interface SelectionContextValue {
  state: SelectionState;
  selectNode: (nodeId: string | null) => void;
  hoverNode: (nodeId: string | null) => void;
  hoverSegment: (index: number | null) => void;
  selectSegment: (index: number | null) => void;
  clearSelection: () => void;
}

export function SelectionProvider({
  tree,
  children,
}: {
  tree: LayoutTreeNode;
  children: ReactNode;
}) {
  const [state, dispatch] = useReducer(selectionReducer, initialState);

  const selectNode = useCallback(
    (nodeId: string | null) => dispatch({ type: 'selectNode', nodeId, tree }),
    [tree],
  );

  const hoverNode = useCallback(
    (nodeId: string | null) => dispatch({ type: 'hoverNode', nodeId, tree }),
    [tree],
  );

  const hoverSegment = useCallback(
    (index: number | null) => dispatch({ type: 'hoverSegment', index }),
    [],
  );

  const selectSegment = useCallback(
    (index: number | null) => dispatch({ type: 'selectSegment', index }),
    [],
  );

  const clearSelection = useCallback(() => dispatch({ type: 'clearSelection' }), []);

  return (
    <SelectionContext.Provider
      value={{ state, selectNode, hoverNode, hoverSegment, selectSegment, clearSelection }}
    >
      {children}
    </SelectionContext.Provider>
  );
}
