// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Decorator } from '@storybook/react-vite';
import type { VortexFileState } from '../contexts/VortexFileContext';
import type { LayoutTreeNode } from '../components/swimlane/types';
import { VortexFileProvider } from '../contexts/VortexFileContext';
import { SelectionProvider } from '../contexts/SelectionContext';

const EMPTY_TREE: LayoutTreeNode = {
  id: 'root',
  encoding: 'vortex.struct',
  dtype: 'struct',
  rowCount: 0,
  rowOffset: 0,
  metadataBytes: 0,
  segmentIds: [],
  childType: { kind: 'root' },
  children: [],
};

/**
 * Storybook decorator that wraps a story in VortexFileContext.Provider
 */
export function withMockFileContext(state: VortexFileState): Decorator {
  return (Story) => (
    <VortexFileProvider value={state}>
      <Story />
    </VortexFileProvider>
  );
}

/**
 * Storybook decorator that wraps a story in both VortexFileContext and SelectionContext
 */
export function withMockSelection(tree?: LayoutTreeNode): Decorator {
  return (Story) => (
    <SelectionProvider tree={tree ?? EMPTY_TREE}>
      <Story />
    </SelectionProvider>
  );
}
