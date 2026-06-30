// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { ReactNode } from 'react';
import type {
  LayoutTreeNode,
  SegmentMapEntry,
  FileStructureInfo,
  ArrayEncodingNode,
} from '../components/swimlane/types';
import { VortexFileContext } from './VortexFileContextCore';

export interface VortexFileState {
  fileName: string;
  fileSize: number;
  rowCount: number;
  version: number;
  dtype: string;
  layoutTree: LayoutTreeNode;
  segments: SegmentMapEntry[];
  fileStructure: FileStructureInfo;
}

export interface VortexFileContextValue extends VortexFileState {
  fetchEncodingTree: (nodeId: string) => Promise<ArrayEncodingNode>;
  previewData: (nodeId: string, rowLimit: number) => Promise<Uint8Array>;
  /** Fetch and attach array encoding tree children to a flat layout node. */
  expandArrayTree: (nodeId: string) => Promise<void>;
  /** Fetch a buffer from a decoded array node. */
  fetchArrayBuffer: (
    layoutNodeId: string,
    arrayPath: string[],
    bufferIndex: number,
  ) => Promise<Uint8Array>;
  /** Preview data from a specific array node, returning Arrow IPC bytes. */
  previewArrayData: (
    layoutNodeId: string,
    arrayPath: string[],
    rowLimit: number,
  ) => Promise<Uint8Array>;
}

export function VortexFileProvider({
  value,
  children,
}: {
  value: VortexFileContextValue;
  children: ReactNode;
}) {
  return <VortexFileContext.Provider value={value}>{children}</VortexFileContext.Provider>;
}
