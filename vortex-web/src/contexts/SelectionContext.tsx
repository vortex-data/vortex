// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { createContext, useContext, useReducer, useCallback, type ReactNode } from 'react';
import type { LayoutTreeNode } from '../components/swimlane/types';
import { findNodeById } from '../components/swimlane/utils';

export interface SelectionState {
  selectedNodeId: string | null;
  selectedNode: LayoutTreeNode | null;
  hoveredNodeId: string | null;
}

type SelectionAction =
  | { type: 'selectNode'; nodeId: string | null; tree: LayoutTreeNode }
  | { type: 'hoverNode'; nodeId: string | null }
  | { type: 'clearSelection' };

function selectionReducer(state: SelectionState, action: SelectionAction): SelectionState {
  switch (action.type) {
    case 'selectNode': {
      const nodeId = action.nodeId;
      const node = nodeId ? findNodeById(action.tree, nodeId) : null;
      return { ...state, selectedNodeId: nodeId, selectedNode: node };
    }
    case 'hoverNode':
      return { ...state, hoveredNodeId: action.nodeId };
    case 'clearSelection':
      return { selectedNodeId: null, selectedNode: null, hoveredNodeId: null };
  }
}

const initialState: SelectionState = {
  selectedNodeId: null,
  selectedNode: null,
  hoveredNodeId: null,
};

interface SelectionContextValue {
  state: SelectionState;
  selectNode: (nodeId: string | null) => void;
  hoverNode: (nodeId: string | null) => void;
  clearSelection: () => void;
}

const SelectionContext = createContext<SelectionContextValue | null>(null);

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
    (nodeId: string | null) => dispatch({ type: 'hoverNode', nodeId }),
    [],
  );

  const clearSelection = useCallback(() => dispatch({ type: 'clearSelection' }), []);

  return (
    <SelectionContext.Provider value={{ state, selectNode, hoverNode, clearSelection }}>
      {children}
    </SelectionContext.Provider>
  );
}

export function useSelection(): SelectionContextValue {
  const ctx = useContext(SelectionContext);
  if (!ctx) throw new Error('useSelection must be used within SelectionProvider');
  return ctx;
}

export { SelectionContext };
