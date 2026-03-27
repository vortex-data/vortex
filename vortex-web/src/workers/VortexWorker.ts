// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type {
  LayoutTreeNode,
  SegmentMapEntry,
  FileStructureInfo,
} from '../components/swimlane/types';

export interface OpenFileResult {
  rowCount: number;
  dtype: string;
  layoutTree: LayoutTreeNode;
  segments: SegmentMapEntry[];
  fileStructure: FileStructureInfo;
}

interface PendingRequest {
  resolve: (value: OpenFileResult) => void;
  reject: (reason: Error) => void;
}

/** Typed async wrapper around the Vortex WASM Web Worker. */
export class VortexWorker {
  private worker: Worker;
  private pending = new Map<number, PendingRequest>();
  private nextId = 0;

  constructor() {
    this.worker = new Worker(
      new URL('./vortex.worker.ts', import.meta.url),
      { type: 'module' },
    );
    this.worker.onmessage = (e: MessageEvent) => {
      const { type, id, data, error } = e.data;
      const req = this.pending.get(id);
      if (!req) return;
      this.pending.delete(id);
      if (type === 'result') {
        req.resolve(data as OpenFileResult);
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

  terminate(): void {
    this.worker.terminate();
  }
}
