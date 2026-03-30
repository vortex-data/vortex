// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type {
  LayoutTreeNode,
  SegmentMapEntry,
  FileStructureInfo,
  ArrayEncodingNode,
} from '../components/swimlane/types';

export interface OpenFileResult {
  rowCount: number;
  dtype: string;
  layoutTree: LayoutTreeNode;
  segments: SegmentMapEntry[];
  fileStructure: FileStructureInfo;
}

interface PendingRequest<T = unknown> {
  resolve: (value: T) => void;
  reject: (reason: Error) => void;
}

/** Typed async wrapper around the Vortex WASM Web Worker. */
export class VortexWorker {
  private worker: Worker;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private pending = new Map<number, PendingRequest<any>>();
  private nextId = 0;

  constructor() {
    this.worker = new Worker(new URL('./vortex.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent) => {
      const { type, id, data, error } = e.data;
      const req = this.pending.get(id);
      if (!req) return;
      this.pending.delete(id);
      if (type === 'result') {
        req.resolve(data);
      } else {
        req.reject(new Error(error ?? 'Unknown worker error'));
      }
    };
  }

  /** Open a Vortex file in the worker. The File is structured-cloneable. */
  openFile(file: File): Promise<OpenFileResult> {
    const id = this.nextId++;
    return new Promise<OpenFileResult>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'open', id, file });
    });
  }

  /** Fetch the array encoding tree for a flat layout node by its ID. */
  fetchEncodingTree(nodeId: string): Promise<ArrayEncodingNode> {
    const id = this.nextId++;
    return new Promise<ArrayEncodingNode>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'fetchEncodingTree', id, nodeId });
    });
  }

  /** Fetch a buffer from a decoded array node. */
  fetchArrayBuffer(
    layoutNodeId: string,
    arrayPath: string[],
    bufferIndex: number,
  ): Promise<Uint8Array> {
    const id = this.nextId++;
    return new Promise<Uint8Array>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({
        type: 'fetchArrayBuffer',
        id,
        layoutNodeId,
        arrayPath,
        bufferIndex,
      });
    });
  }

  /** Preview data from a specific array node within a flat layout. */
  previewArrayData(
    layoutNodeId: string,
    arrayPath: string[],
    rowLimit: number,
  ): Promise<Uint8Array> {
    const id = this.nextId++;
    return new Promise<Uint8Array>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'previewArrayData', id, layoutNodeId, arrayPath, rowLimit });
    });
  }

  /** Preview data from a specific layout node, returning Arrow IPC bytes. */
  previewData(nodeId: string, rowLimit: number): Promise<Uint8Array> {
    const id = this.nextId++;
    return new Promise<Uint8Array>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'previewData', id, nodeId, rowLimit });
    });
  }

  terminate(): void {
    this.worker.terminate();
  }
}
