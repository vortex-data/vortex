// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { VortexFileState } from './contexts/VortexFileContext';
import type { ArrayEncodingNode, LayoutTreeNode } from './components/swimlane/types';
import { arrayTreeToLayoutChildren, findNodeById } from './components/swimlane/utils';
import type { OpenFileResult } from './workers/VortexWorker';

export interface ExplorerWorkerClient {
  openFile(file: File, generation: number): Promise<OpenFileResult>;
  closeFile(generation: number): void;
  fetchEncodingTree(nodeId: string, generation: number): Promise<ArrayEncodingNode>;
  fetchArrayBuffer(
    layoutNodeId: string,
    arrayPath: string[],
    bufferIndex: number,
    generation: number,
  ): Promise<Uint8Array>;
  previewArrayData(
    layoutNodeId: string,
    arrayPath: string[],
    rowLimit: number,
    generation: number,
  ): Promise<Uint8Array>;
  previewData(nodeId: string, rowLimit: number, generation: number): Promise<Uint8Array>;
}

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
  private worker: ExplorerWorkerClient;
  private cachedCallbacksGen = -1;
  private cachedCallbacks: ExplorerFileCallbacks | null = null;

  constructor(worker: ExplorerWorkerClient) {
    this.worker = worker;
  }

  /**
   * Returns stable context callbacks bound to a single file generation so
   * in-file state updates (like tree expansion) do not recreate callbacks
   * and trigger consumer effects such as BuffersPane auto-selection.
   */
  getFileCallbacks(generation: number): ExplorerFileCallbacks {
    if (this.cachedCallbacks && this.cachedCallbacksGen === generation) {
      return this.cachedCallbacks;
    }
    this.cachedCallbacksGen = generation;
    this.cachedCallbacks = {
      fetchEncodingTree: (nodeId) => this.fetchEncodingTree(nodeId, generation),
      previewData: (nodeId, rowLimit) => this.previewData(nodeId, rowLimit, generation),
      expandArrayTree: (nodeId) => this.expandArrayTree(nodeId, generation),
      fetchArrayBuffer: (layoutNodeId, arrayPath, bufferIndex) =>
        this.fetchArrayBuffer(layoutNodeId, arrayPath, bufferIndex, generation),
      previewArrayData: (layoutNodeId, arrayPath, rowLimit) =>
        this.previewArrayData(layoutNodeId, arrayPath, rowLimit, generation),
    };
    return this.cachedCallbacks;
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
      const result = await this.worker.openFile(file, gen);
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
      this.worker.closeFile(gen);
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
    this.worker.closeFile(gen);
    this.update({
      fileState: null,
      activeGeneration: 0,
      loading: false,
      error: null,
    });
  }

  fetchEncodingTree(nodeId: string, generation = this.snapshot.activeGeneration) {
    return this.worker.fetchEncodingTree(nodeId, generation);
  }

  previewData(nodeId: string, rowLimit: number, generation = this.snapshot.activeGeneration) {
    return this.worker.previewData(nodeId, rowLimit, generation);
  }

  fetchArrayBuffer(
    layoutNodeId: string,
    arrayPath: string[],
    bufferIndex: number,
    generation = this.snapshot.activeGeneration,
  ) {
    return this.worker.fetchArrayBuffer(layoutNodeId, arrayPath, bufferIndex, generation);
  }

  previewArrayData(
    layoutNodeId: string,
    arrayPath: string[],
    rowLimit: number,
    generation = this.snapshot.activeGeneration,
  ) {
    return this.worker.previewArrayData(layoutNodeId, arrayPath, rowLimit, generation);
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
      arrayTree = await this.worker.fetchEncodingTree(nodeId, generation);
    } catch {
      return;
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
