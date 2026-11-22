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

/** Minimal Worker surface needed by VortexWorker (allows test injection). */
export interface WorkerLike {
  onmessage: ((e: MessageEvent) => void) | null;
  onerror: ((e: ErrorEvent) => void) | null;
  postMessage(message: unknown): void;
  terminate(): void;
}

/** Typed async wrapper around the Vortex WASM Web Worker. */
export class VortexWorker {
  private worker: WorkerLike;
  // biome-ignore lint/suspicious/noExplicitAny: requests with different result types share one map.
  private pending = new Map<number, PendingRequest<any>>();
  private nextId = 0;
  private closedError: Error | null = null;

  constructor(worker?: WorkerLike) {
    this.worker =
      worker ?? new Worker(new URL('./vortex.worker.ts', import.meta.url), { type: 'module' });
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
    this.worker.onerror = (e: ErrorEvent) => {
      const detail = e.message || 'Worker crashed';
      this.failAllPending(new Error(`Worker error: ${detail}`));
    };
  }

  private failAllPending(err: Error): void {
    if (this.closedError) return;
    this.closedError = err;
    const inflight = Array.from(this.pending.values());
    this.pending.clear();
    this.worker.terminate();
    for (const req of inflight) {
      req.reject(err);
    }
  }

  private sendRequest<T>(payload: Record<string, unknown>): Promise<T> {
    if (this.closedError) {
      return Promise.reject(this.closedError);
    }
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id });
    });
  }

  /** Open a Vortex file in the worker. The File is structured-cloneable. */
  openFile(file: File): Promise<OpenFileResult> {
    return this.sendRequest<OpenFileResult>({ type: 'open', file });
  }

  /** Fetch the array encoding tree for a flat layout node by its ID. */
  fetchEncodingTree(nodeId: string): Promise<ArrayEncodingNode> {
    return this.sendRequest<ArrayEncodingNode>({ type: 'fetchEncodingTree', nodeId });
  }

  /** Fetch a buffer from a decoded array node. */
  fetchArrayBuffer(
    layoutNodeId: string,
    arrayPath: string[],
    bufferIndex: number,
  ): Promise<Uint8Array> {
    return this.sendRequest<Uint8Array>({
      type: 'fetchArrayBuffer',
      layoutNodeId,
      arrayPath,
      bufferIndex,
    });
  }

  /** Preview data from a specific array node within a flat layout. */
  previewArrayData(
    layoutNodeId: string,
    arrayPath: string[],
    rowLimit: number,
  ): Promise<Uint8Array> {
    return this.sendRequest<Uint8Array>({
      type: 'previewArrayData',
      layoutNodeId,
      arrayPath,
      rowLimit,
    });
  }

  /** Preview data from a specific layout node, returning Arrow IPC bytes. */
  previewData(nodeId: string, rowLimit: number): Promise<Uint8Array> {
    return this.sendRequest<Uint8Array>({ type: 'previewData', nodeId, rowLimit });
  }

  closeFile(): void {
    this.worker.postMessage({ type: 'close' });
  }

  terminate(): void {
    this.failAllPending(new Error('Worker terminated'));
  }
}
