// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Meta, StoryObj } from '@storybook/react-vite';
import { BlockTreemap } from './BlockTreemap';
import { withMockFileContext, withMockSelection } from '../../storybook/decorators';
import { ordersMock, heavyChunksMock } from '../../mocks/layouts';
import { generateSegments } from '../../mocks/segments';
import { generateFileStructure } from '../../mocks/fileStructure';
import type { LayoutTreeNode } from '../swimlane/types';
import type { VortexFileState } from '../../contexts/VortexFileContext';
import { findNodeById } from '../swimlane/utils';

function mockState(
  fileName: string,
  layout: LayoutTreeNode,
  fileSize: number,
  rowCount: number,
): VortexFileState {
  const segments = generateSegments(layout, fileSize);
  return {
    fileName,
    fileSize,
    rowCount,
    version: 1,
    dtype: layout.dtype,
    layoutTree: layout,
    segments,
    fileStructure: generateFileStructure(segments, fileSize),
  };
}

const orders = ordersMock();
const ordersState = mockState('orders.vortex', orders, 12_400_000, 100_000);

const meta: Meta<typeof BlockTreemap> = {
  component: BlockTreemap,
  parameters: { layout: 'fullscreen' },
  globals: { theme: 'dark' },
  decorators: [
    (Story) => (
      <div style={{ width: '100%', height: '600px' }}>
        <Story />
      </div>
    ),
  ],
};
export default meta;

type Story = StoryObj<typeof BlockTreemap>;

const noop = {
  onSelectNode: (id: string | null) => console.log('select (zoom in)', id),
  onHoverNode: (id: string | null) => console.log('hover', id),
};

/** The whole file — every physical block down to the leaves, with one block
 *  selected (highlighted). Single-click selects, double-click zooms in. */
export const Root: Story = {
  decorators: [withMockFileContext(ordersState), withMockSelection(orders)],
  args: {
    root: orders,
    segments: ordersState.segments,
    fileSize: 12_400_000,
    selectedNodeId: 'root.amount',
    hoveredNodeId: null,
    ...noop,
  },
};

/** Zoomed in: rooted at a single field (what a double-click produces). */
export const Field: Story = {
  decorators: [withMockFileContext(ordersState), withMockSelection(orders)],
  args: {
    root: findNodeById(orders, 'root.customer')!,
    segments: ordersState.segments,
    fileSize: 12_400_000,
    selectedNodeId: null,
    hoveredNodeId: null,
    ...noop,
  },
};

const heavy = heavyChunksMock();
const heavyState = mockState('heavy.vortex', heavy, 80_000_000, 1_000_000);

/** A column with 500 chunks, all rendered as their own blocks. */
export const HeavyChunks: Story = {
  decorators: [withMockFileContext(heavyState), withMockSelection(heavy)],
  args: {
    root: heavy,
    segments: heavyState.segments,
    fileSize: 80_000_000,
    selectedNodeId: null,
    hoveredNodeId: null,
    ...noop,
  },
};
