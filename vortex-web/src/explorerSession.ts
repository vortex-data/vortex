// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { VortexFileState } from './contexts/VortexFileContext';
import type { ArrayEncodingNode, LayoutTreeNode } from './components/swimlane/types';
import { arrayTreeToLayoutChildren, findNodeById } from './components/swimlane/utils';
import type { VortexWorker } from './workers/VortexWorker';

export interface ExplorerSessionSnapshot {
  fileState: VortexFileState | null;
  activeGeneration: number;
  loading: boolean;
  error: string | null;
}

export interface ExplorerFileCallbacks {
  fetchEncodingTree: (nodeId: string) => Promise<ArrayEncodingNode>;
  previewData: (nodeId: string, rowLimit: number) => Promise<Uint8Array>;
  expandArrayTree: (nodeId: string) => Promise<void>;
  fetchArrayBuffer: (
    layoutNodeId: string,
    arrayPath: string[],
    bufferIndex: number,
  ) => Promise<Uint8Array>;
  previewArrayData: (
    layoutNodeId: string,
    arrayPath: string[],
    rowLimit: number,
  ) => Promise<Uint8Array>;
}

/** Clone a tree, replacing the node at targetId with a modified version. */
function cloneTreeWithUpdate(
  root: LayoutTreeNode,
  targetId: string,
  update: (node: LayoutTreeNode) => LayoutTreeNode,
): LayoutTreeNode {
  if (root.id === targetId) return update(root);
  const newChildren = root.children.map((child) => cloneTreeWithUpdate(child, targetId, update));
  if (newChildren === root.children) return root;
  return { ...root, children: newChildren };
}

/**
 * Coordinates file open, close, and per-file worker operations using a monotonic
 * generation counter so stale asynchronous replies cannot overwrite a newer file.
 */
export class ExplorerSession {
  private generation = 0;
  private snapshot: ExplorerSessionSnapshot = {
    fileState: null,
    activeGeneration: 0,
    loading: false,
    error: null,
  };
  private listeners = new Set<() => void>();
  private getWorker: () => VortexWorker | null;

  constructor(getWorker: () => VortexWorker | null) {
    this.getWorker = getWorker;
  }

  getFileCallbacks(generation: number): ExplorerFileCallbacks {
    return {
      fetchEncodingTree: (nodeId) => this.getWorker()!.fetchEncodingTree(nodeId, generation),
      previewData: (nodeId, rowLimit) =>
        this.getWorker()!.previewData(nodeId, rowLimit, generation),
      expandArrayTree: (nodeId) => this.expandArrayTree(nodeId, generation),
      fetchArrayBuffer: (layoutNodeId, arrayPath, bufferIndex) =>
        this.getWorker()!.fetchArrayBuffer(layoutNodeId, arrayPath, bufferIndex, generation),
      previewArrayData: (layoutNodeId, arrayPath, rowLimit) =>
        this.getWorker()!.previewArrayData(layoutNodeId, arrayPath, rowLimit, generation),
    };
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): ExplorerSessionSnapshot => this.snapshot;

  private update(next: ExplorerSessionSnapshot): void {
    this.snapshot = next;
    for (const listener of this.listeners) {
      listener();
    }
  }

  async openFile(file: File): Promise<void> {
    const gen = ++this.generation;
    this.update({ ...this.snapshot, error: null, loading: true });
    try {
      const result = await this.getWorker()!.openFile(file, gen);
      if (this.generation !== gen) return;
      this.update({
        fileState: {
          fileName: file.name,
          fileSize: file.size,
          rowCount: result.rowCount,
          version: result.fileStructure.version,
          dtype: result.dtype,
          layoutTree: result.layoutTree,
          segments: result.segments,
          fileStructure: result.fileStructure,
        },
        activeGeneration: gen,
        loading: false,
        error: null,
      });
    } catch (e) {
      if (this.generation !== gen) return;
      this.getWorker()?.closeFile(gen);
      this.update({
        fileState: null,
        activeGeneration: 0,
        loading: false,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  closeFile(): void {
    const gen = ++this.generation;
    this.getWorker()?.closeFile(gen);
    this.update({
      fileState: null,
      activeGeneration: 0,
      loading: false,
      error: null,
    });
  }

  async expandArrayTree(
    nodeId: string,
    generation = this.snapshot.activeGeneration,
  ): Promise<void> {
    if (generation === 0 || this.snapshot.activeGeneration !== generation) {
      return;
    }

    let arrayTree: ArrayEncodingNode;
    try {
      arrayTree = await this.getWorker()!.fetchEncodingTree(nodeId, generation);
    } catch (e) {
      if (this.snapshot.activeGeneration !== generation) {
        return;
      }
      throw e;
    }
    if (!arrayTree) return;

    const current = this.snapshot;
    if (current.activeGeneration !== generation || !current.fileState) {
      return;
    }

    const node = findNodeById(current.fileState.layoutTree, nodeId);
    if (node?.encoding !== 'vortex.flat') return;
    if (node.children.some((c) => c.isArrayNode)) return;

    const arrayChildren = arrayTreeToLayoutChildren(arrayTree, node);
    const newTree = cloneTreeWithUpdate(current.fileState.layoutTree, nodeId, (n) => ({
      ...n,
      arrayEncodingTree: arrayTree,
      children: [...n.children, ...arrayChildren],
    }));

    this.update({
      ...current,
      fileState: { ...current.fileState, layoutTree: newTree },
    });
  }
}
