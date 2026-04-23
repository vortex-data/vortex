// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Meta, StoryObj } from '@storybook/react-vite';
import { VideoExplorer } from './VideoExplorer';
import { withMockFileContext, withMockSelection } from '../../storybook/decorators';
import { simpleMock } from '../../mocks/layouts';
import { generateSegments } from '../../mocks/segments';
import { generateFileStructure } from '../../mocks/fileStructure';
import { makeVideoIndexMock } from '../../mocks/videoIndex';
import type { VortexFileState } from '../../contexts/VortexFileContext';

const layout = simpleMock();
const segments = generateSegments(layout, 34_000);
const fileStructure = generateFileStructure(segments, 34_000);
const videoIndex = makeVideoIndexMock();

const mockFileState: VortexFileState = {
  kind: 'videoIndex',
  fileName: 'mock-video.vortex',
  fileSize: 34_000,
  rowCount: 1,
  version: 1,
  dtype: '{source_uri=utf8, codec=utf8, gops=list<struct>, frames_by_video=list<struct>}',
  layoutTree: layout,
  segments,
  fileStructure,
  videoIndex,
  pairedMedia: {
    fileName: null,
    objectUrl: null,
    hasMedia: false,
  },
};

function StoryPage() {
  return (
    <div className="h-screen bg-vortex-white dark:bg-vortex-black">
      <VideoExplorer onAttachMedia={() => {}} />
    </div>
  );
}

const meta: Meta<typeof StoryPage> = {
  component: StoryPage,
  parameters: { layout: 'fullscreen' },
  decorators: [withMockFileContext(mockFileState), withMockSelection(layout)],
};
export default meta;

type Story = StoryObj<typeof StoryPage>;

export const Default: Story = {};
