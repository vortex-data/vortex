// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Meta, StoryObj } from '@storybook/react-vite';
import { FileMap } from './FileMap';
import { withMockFileContext, withMockSelection } from '../../storybook/decorators';
import { ordersMock } from '../../mocks/layouts';
import { generateSegments } from '../../mocks/segments';
import { generateFileStructure } from '../../mocks/fileStructure';
import type { VortexFileState } from '../../contexts/VortexFileContext';

const layout = ordersMock();
const segments = generateSegments(layout, 12_400_000);
const fileStructure = generateFileStructure(segments, 12_400_000);

const mockFileState: VortexFileState = {
  fileName: 'orders.vortex',
  fileSize: 12_400_000,
  rowCount: 100_000,
  version: 1,
  dtype: '{order_id=i64, ...}',
  layoutTree: layout,
  segments,
  fileStructure,
};

const meta: Meta<typeof FileMap> = {
  component: FileMap,
  decorators: [withMockFileContext(mockFileState), withMockSelection(layout)],
};
export default meta;

type Story = StoryObj<typeof FileMap>;

export const Default: Story = {
  args: { height: 40 },
};

export const Tall: Story = {
  args: { height: 80 },
};
