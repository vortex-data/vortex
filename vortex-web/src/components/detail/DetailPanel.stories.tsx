// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Meta, StoryObj } from '@storybook/react-vite';
import { DetailPanel } from './DetailPanel';
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

const meta: Meta<typeof DetailPanel> = {
  component: DetailPanel,
  decorators: [withMockFileContext(mockFileState), withMockSelection(layout)],
  parameters: {
    layout: 'padded',
  },
};
export default meta;

type Story = StoryObj<typeof DetailPanel>;

export const NoSelection: Story = {};
