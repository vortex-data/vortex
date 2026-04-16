// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Meta, StoryObj } from '@storybook/react-vite';
import { FileHeader } from './FileHeader';
import { MainArea } from './MainArea';
import { StatusBar } from './StatusBar';
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

/** Full page layout — mirrors App.tsx when a file is loaded */
function ExplorerPage() {
  return (
    <div className="flex flex-col h-screen bg-vortex-white dark:bg-vortex-black">
      <FileHeader onClose={() => {}} />
      <MainArea />
      <StatusBar />
    </div>
  );
}

const meta: Meta<typeof ExplorerPage> = {
  component: ExplorerPage,
  parameters: { layout: 'fullscreen' },
  decorators: [withMockFileContext(mockFileState), withMockSelection(layout)],
};
export default meta;

type Story = StoryObj<typeof ExplorerPage>;

export const Default: Story = {};
