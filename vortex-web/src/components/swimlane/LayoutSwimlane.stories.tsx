// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Meta, StoryObj } from '@storybook/react-vite';
import { LayoutSwimlane } from './LayoutSwimlane';
import type { LayoutNode } from './types';

function generateChunks(id: string, count: number, totalRange: [number, number]) {
  const [start, end] = totalRange;
  const rowsPerChunk = (end - start) / count;

  return Array.from({ length: count }, (_, i) => ({
    id: `${id}_c${i}`,
    name: `chunk ${i}`,
    rowRange: [
      Math.round(start + i * rowsPerChunk),
      Math.round(start + (i + 1) * rowsPerChunk),
    ] as [number, number],
    child: {
      id: `${id}_c${i}_data`,
      name: 'data',
      type: 'flat' as const,
      rowRange: [
        Math.round(start + i * rowsPerChunk),
        Math.round(start + (i + 1) * rowsPerChunk),
      ] as [number, number],
      meta: {
        dtype: 'int64',
        bytes: Math.round(rowsPerChunk * 8),
        min: Math.round(start + i * rowsPerChunk),
        max: Math.round(start + (i + 1) * rowsPerChunk),
      },
    },
  }));
}

const ordersLayout: LayoutNode = {
  id: 'root',
  name: 'orders',
  type: 'struct',
  rowRange: [0, 100000],
  children: [
    {
      id: 'order_id',
      name: 'order_id',
      type: 'flat',
      rowRange: [0, 100000],
      meta: { dtype: 'int64', bytes: 800000, min: 1, max: 100000 },
    },
    {
      id: 'is_active',
      name: 'is_active',
      type: 'flat',
      rowRange: [0, 100000],
      meta: { dtype: 'bool', bytes: 100000, min: false, max: true },
    },
    {
      id: 'customer',
      name: 'customer',
      type: 'struct',
      rowRange: [0, 100000],
      children: [
        {
          id: 'customer_id',
          name: 'id',
          type: 'chunked',
          rowRange: [0, 100000],
          chunkCount: 50,
          chunks: generateChunks('cid', 50, [0, 100000]),
        },
        {
          id: 'customer_name',
          name: 'name',
          type: 'chunked',
          rowRange: [0, 100000],
          chunkCount: 8,
          chunks: Array.from({ length: 8 }, (_, i) => ({
            id: `cname_c${i}`,
            name: `chunk ${i}`,
            rowRange: [i * 12500, (i + 1) * 12500] as [number, number],
            child: {
              id: `cname_c${i}_dict`,
              name: 'dict',
              type: 'dict' as const,
              rowRange: [i * 12500, (i + 1) * 12500] as [number, number],
              children: [
                {
                  id: `cname_c${i}_codes`,
                  name: 'codes',
                  type: 'flat' as const,
                  rowRange: [i * 12500, (i + 1) * 12500] as [number, number],
                  meta: { dtype: 'uint16', bytes: 25000, min: 0, max: 4200 },
                },
                {
                  id: `cname_c${i}_vals`,
                  name: 'values',
                  type: 'flat' as const,
                  rowRange: [i * 12500, (i + 1) * 12500] as [number, number],
                  meta: { dtype: 'utf8', bytes: 45000, min: 'Aaron', max: 'Zoe' },
                },
              ],
            },
          })),
        },
      ],
    },
    {
      id: 'items',
      name: 'items',
      type: 'flat',
      rowRange: [0, 100000],
      meta: { dtype: 'list<struct>', bytes: 2400000, min: '[]', max: '[...]' },
    },
    {
      id: 'amount',
      name: 'amount',
      type: 'zonemap',
      rowRange: [0, 100000],
      zoneCount: 5,
      zones: Array.from({ length: 5 }, (_, i) => ({
        id: `amt_z${i}`,
        name: `zone ${i}`,
        rowRange: [i * 20000, (i + 1) * 20000] as [number, number],
        meta: { min: i * 100, max: (i + 1) * 150 },
        child: {
          id: `amt_z${i}_data`,
          name: 'data',
          type: 'flat' as const,
          rowRange: [i * 20000, (i + 1) * 20000] as [number, number],
          meta: {
            dtype: 'float64',
            bytes: 160000,
            min: i * 100,
            max: (i + 1) * 150,
          },
        },
      })),
    },
    {
      id: 'metadata',
      name: 'metadata',
      type: 'flat',
      rowRange: [0, 100000],
      meta: { dtype: 'struct', bytes: 500000, min: '{}', max: '{...}' },
    },
    {
      id: 'status',
      name: 'status',
      type: 'dict',
      rowRange: [0, 100000],
      children: [
        {
          id: 'status_codes',
          name: 'codes',
          type: 'flat',
          rowRange: [0, 100000],
          meta: { dtype: 'uint8', bytes: 100000, min: 0, max: 4 },
        },
        {
          id: 'status_vals',
          name: 'values',
          type: 'flat',
          rowRange: [0, 100000],
          meta: { dtype: 'utf8', bytes: 38, min: 'cancelled', max: 'shipped' },
        },
      ],
    },
  ],
};

const meta: Meta<typeof LayoutSwimlane> = {
  component: LayoutSwimlane,
  parameters: {
    layout: 'fullscreen',
  },
  decorators: [
    (Story) => (
      <div className="p-4 max-w-6xl mx-auto">
        <Story />
      </div>
    ),
  ],
};
export default meta;

type Story = StoryObj<typeof LayoutSwimlane>;

export const Orders: Story = {
  args: {
    layout: ordersLayout,
    totalRows: 100000,
    fileName: 'orders.vortex',
    defaultExpanded: ['root', 'customer', 'customer_id', 'status'],
  },
};

export const Collapsed: Story = {
  args: {
    layout: ordersLayout,
    totalRows: 100000,
    fileName: 'orders.vortex',
    defaultExpanded: [],
  },
};

export const FullyExpanded: Story = {
  args: {
    layout: ordersLayout,
    totalRows: 100000,
    fileName: 'orders.vortex',
    defaultExpanded: [
      'root',
      'customer',
      'customer_id',
      'customer_name',
      'amount',
      'status',
      'items',
      'metadata',
    ],
  },
};

const simpleLayout: LayoutNode = {
  id: 'root',
  name: 'data',
  type: 'flat',
  rowRange: [0, 10000],
  meta: { dtype: 'int64', bytes: 80000, min: 0, max: 9999 },
};

export const SingleFlat: Story = {
  args: {
    layout: simpleLayout,
    totalRows: 10000,
    fileName: 'simple.vortex',
  },
};
