// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Meta, StoryObj } from '@storybook/react-vite';
import { SwimlaneOverview } from './SwimlaneOverview';
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
  dtype:
    '{order_id=i64, is_active=bool, customer={id=i64, name=utf8}, items=list<struct>, amount=f64, metadata=struct, status=utf8}',
  layoutTree: layout,
  segments,
  fileStructure,
};

const meta: Meta<typeof SwimlaneOverview> = {
  component: SwimlaneOverview,
  parameters: { layout: 'padded' },
  decorators: [
    withMockFileContext(mockFileState),
    withMockSelection(layout),
    // The overview fills its parent's height; give it a bounded box like the panel
    // it occupies in the app.
    (Story) => (
      <div style={{ height: 240 }}>
        <Story />
      </div>
    ),
  ],
};
export default meta;

type Story = StoryObj<typeof SwimlaneOverview>;

export const Default: Story = {};
