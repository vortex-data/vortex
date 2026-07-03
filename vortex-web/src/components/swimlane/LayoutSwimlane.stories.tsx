// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { LayoutSwimlane } from './LayoutSwimlane';
import type { LayoutTreeNode } from './types';
import {
  ordersMock,
  simpleMock,
  wideMock,
  deepMock,
  heavyChunksMock,
  gappedMock,
  generateSegments,
} from '../../mocks';

/** Bundle a layout with a generated physical segment map for the byte-based axis. */
function physical(layout: LayoutTreeNode, fileSize: number) {
  return { layout, segments: generateSegments(layout, fileSize), totalBytes: fileSize };
}

const meta: Meta<typeof LayoutSwimlane> = {
  component: LayoutSwimlane,
  parameters: {
    layout: 'fullscreen',
  },
  decorators: [
    (Story) => (
      <div className="p-4 max-w-6xl mx-auto" style={{ height: 500 }}>
        <Story />
      </div>
    ),
  ],
};
export default meta;

type Story = StoryObj<typeof LayoutSwimlane>;

const ordersLayout = ordersMock();

export const Orders: Story = {
  args: {
    ...physical(ordersLayout, 12_400_000),
    defaultExpanded: ['root', 'root.customer', 'root.customer.id', 'root.status'],
    height: 400,
  },
};

export const SchemaMode: Story = {
  args: {
    ...physical(ordersLayout, 12_400_000),
    mode: 'schema',
    defaultExpanded: ['root', 'root.customer'],
    height: 400,
  },
};

export const LayoutMode: Story = {
  args: {
    ...physical(ordersLayout, 12_400_000),
    mode: 'layout',
    defaultExpanded: ['root', 'root.customer', 'root.customer.id', 'root.status'],
    height: 400,
  },
};

export const SingleFlat: Story = {
  args: {
    ...physical(simpleMock(), 800_000),
    height: 100,
  },
};

export const WideSchema: Story = {
  args: {
    ...physical(wideMock(), 6_000_000),
    mode: 'schema',
    defaultExpanded: ['root'],
    height: 400,
  },
};

export const DeepNesting: Story = {
  args: {
    ...physical(deepMock(), 2_500_000),
    mode: 'schema',
    defaultExpanded: ['root', 'root.user', 'root.user.profile', 'root.user.profile.address'],
    height: 400,
  },
};

export const HeavyChunks: Story = {
  args: {
    ...physical(heavyChunksMock(), 60_000_000),
    mode: 'layout',
    defaultExpanded: ['root', 'root.values'],
    height: 400,
  },
};

/** Two columns interleaved on disk: each column's chunks are split by the other's,
    so plotting by physical offset reveals the gaps in each column's storage. */
export const Gapped: Story = {
  args: {
    ...gappedMock(),
    mode: 'schema',
    defaultExpanded: ['root', 'root.temp', 'root.count'],
    height: 240,
  },
};

/** Interactive story demonstrating controlled selection */
export const WithSelection: StoryObj = {
  render: () => {
    const [selectedId, setSelectedId] = useState<string | null>(null);
    return (
      <div>
        <div className="text-xs text-vortex-grey-dark mb-2">Selected: {selectedId ?? 'none'}</div>
        <LayoutSwimlane
          {...physical(ordersLayout, 12_400_000)}
          defaultExpanded={['root', 'root.customer']}
          selectedNodeId={selectedId}
          onNodeSelect={setSelectedId}
          height={350}
        />
      </div>
    );
  },
};
