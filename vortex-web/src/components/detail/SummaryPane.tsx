// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode } from '../swimlane/types';
import type { VortexFileState } from '../../contexts/VortexFileContext';
import {
  formatBytes,
  formatRowCount,
  getNodeDisplayName,
  collectSubtreeSegments,
  shortEncoding,
} from '../swimlane/utils';
import { Facts } from './Facts';

interface SummaryPaneProps {
  node: LayoutTreeNode | null;
  file: VortexFileState;
}

const count = (n: number) => n.toLocaleString('en-US');

export function SummaryPane({ node, file }: SummaryPaneProps) {
  if (!node) {
    return (
      <>
        <div className="vx-pane-head">
          <h2 className="vx-label x-side-title">File</h2>
        </div>
        <div className="vx-pane-body">
          <Facts
            items={[
              ['Name', file.fileName],
              ['Size', formatBytes(file.fileSize)],
              ['Rows', formatRowCount(file.rowCount)],
              ['Segments', count(file.segments.length)],
              ['Format version', `v${file.version}`],
            ]}
          />
        </div>
      </>
    );
  }

  const segIds = new Set(collectSubtreeSegments(node));
  const reachable = file.segments.filter((s) => segIds.has(s.index));
  const totalBytes = reachable.reduce((sum, s) => sum + s.byteLength, 0);

  return (
    <>
      <div className="vx-pane-head">
        <h2 className="vx-label x-side-title">{getNodeDisplayName(node)}</h2>
      </div>
      <div className="vx-pane-body">
        <Facts
          items={[
            ['Encoding', shortEncoding(node.encoding), node.encoding],
            ['Rows', count(node.rowCount)],
            ['Row offset', count(node.rowOffset)],
            ['Metadata', formatBytes(node.metadataBytes)],
            ['Data size', formatBytes(totalBytes)],
            ['Segments', count(reachable.length)],
            ['Children', count(node.children.length)],
          ]}
        />
      </div>
    </>
  );
}
