// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { createContext, useContext } from 'react';
import type {
  LayoutTreeNode,
  SegmentMapEntry,
  FileStructureInfo,
} from '../components/swimlane/types';

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

const VortexFileContext = createContext<VortexFileState | null>(null);

export function VortexFileProvider({
  value,
  children,
}: {
  value: VortexFileState;
  children: React.ReactNode;
}) {
  return <VortexFileContext.Provider value={value}>{children}</VortexFileContext.Provider>;
}

export function useVortexFile(): VortexFileState {
  const ctx = useContext(VortexFileContext);
  if (!ctx) throw new Error('useVortexFile must be used within VortexFileProvider');
  return ctx;
}

export { VortexFileContext };
