// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { LayoutSwimlane } from './LayoutSwimlane';
import { ordersMock, simpleMock, wideMock, deepMock, heavyChunksMock } from '../../mocks';

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
    layout: ordersLayout,
    totalRows: 100000,
    defaultExpanded: ['root', 'root.customer', 'root.customer.id', 'root.status'],
    height: 400,
  },
};

export const SchemaMode: Story = {
  args: {
    layout: ordersLayout,
    totalRows: 100000,
    mode: 'schema',
    defaultExpanded: ['root', 'root.customer'],
    height: 400,
  },
};

export const LayoutMode: Story = {
  args: {
    layout: ordersLayout,
    totalRows: 100000,
    mode: 'layout',
    defaultExpanded: ['root', 'root.customer', 'root.customer.id', 'root.status'],
    height: 400,
  },
};

export const SingleFlat: Story = {
  args: {
    layout: simpleMock(),
    totalRows: 10000,
    height: 100,
  },
};

export const WideSchema: Story = {
  args: {
    layout: wideMock(),
    totalRows: 50000,
    mode: 'schema',
    defaultExpanded: ['root'],
    height: 400,
  },
};

export const DeepNesting: Story = {
  args: {
    layout: deepMock(),
    totalRows: 25000,
    mode: 'schema',
    defaultExpanded: ['root', 'root.user', 'root.user.profile', 'root.user.profile.address'],
    height: 400,
  },
};

export const HeavyChunks: Story = {
  args: {
    layout: heavyChunksMock(),
    totalRows: 1000000,
    mode: 'layout',
    defaultExpanded: ['root', 'root.values'],
    height: 400,
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
          layout={ordersLayout}
          totalRows={100000}
          defaultExpanded={['root', 'root.customer']}
          selectedNodeId={selectedId}
          onNodeSelect={setSelectedId}
          height={350}
        />
      </div>
    );
  },
};
