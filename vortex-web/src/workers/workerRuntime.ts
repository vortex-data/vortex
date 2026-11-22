// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { OpenFileResult } from './VortexWorker';

export interface VortexWorkerFileHandle {
  readonly row_count: bigint | number;
  readonly dtype: string;
  layout_tree(): string;
  segment_map(): string;
  file_structure(): string;
  fetch_encoding_tree(nodeId: string): Promise<string>;
  fetch_array_buffer(
    layoutNodeId: string,
    arrayPath: string[],
    bufferIndex: number,
  ): Promise<Uint8Array>;
  preview_array_data(
    layoutNodeId: string,
    arrayPath: string[],
    rowLimit: number,
  ): Promise<Uint8Array>;
  preview_data(nodeId: string, rowLimit: number): Promise<Uint8Array>;
  free(): void;
}

export type WorkerRequest =
  | { type: 'open'; id: number; generation: number; file: File }
  | { type: 'close'; generation: number }
  | { type: 'fetchEncodingTree'; id: number; generation: number; nodeId: string }
  | {
      type: 'fetchArrayBuffer';
      id: number;
      generation: number;
      layoutNodeId: string;
      arrayPath: string[];
      bufferIndex: number;
    }
  | {
      type: 'previewArrayData';
      id: number;
      generation: number;
      layoutNodeId: string;
      arrayPath: string[];
      rowLimit: number;
    }
  | { type: 'previewData'; id: number; generation: number; nodeId: string; rowLimit: number };

export type WorkerResponse =
  | { type: 'result'; id: number; data: unknown }
  | { type: 'error'; id: number; error: string };

export interface WorkerRuntimeDeps {
  init: () => Promise<unknown>;
  openVortexFile: (file: File) => Promise<VortexWorkerFileHandle>;
  postMessage: (msg: WorkerResponse, transfer?: Transferable[]) => void;
}

interface HandleEntry {
  generation: number;
  handle: VortexWorkerFileHandle;
  inFlight: number;
  retired: boolean;
  disposed: boolean;
}

const STALE_GENERATION_ERROR = 'Stale file generation';

export function createVortexWorkerHandler(deps: WorkerRuntimeDeps) {
  let initialized = false;
  let initPromise: Promise<void> | null = null;
  let currentGeneration = 0;
  let activeHandle: HandleEntry | null = null;

  function ensureInit(): Promise<void> {
    if (!initPromise) {
      initPromise = deps
        .init()
        .then(() => {
          initialized = true;
        })
        .catch((err) => {
          initPromise = null;
          throw err;
        });
    }
    return initPromise;
  }

  function disposeIfReady(entry: HandleEntry): void {
    if (entry.retired && entry.inFlight === 0 && !entry.disposed) {
      entry.disposed = true;
      entry.handle.free();
    }
  }

  function retireActiveHandle(): void {
    if (!activeHandle) return;
    const entry = activeHandle;
    activeHandle = null;
    entry.retired = true;
    disposeIfReady(entry);
  }

  async function withActiveHandle<T>(
    generation: number,
    fn: (handle: VortexWorkerFileHandle) => Promise<T>,
  ): Promise<T> {
    const entry = activeHandle;
    if (!entry) {
      throw new Error(generation < currentGeneration ? STALE_GENERATION_ERROR : 'No file open');
    }
    if (entry.generation !== generation) {
      throw new Error(STALE_GENERATION_ERROR);
    }
    entry.inFlight++;
    try {
      const result = await fn(entry.handle);
      if (entry.retired) {
        throw new Error(STALE_GENERATION_ERROR);
      }
      return result;
    } finally {
      entry.inFlight--;
      disposeIfReady(entry);
    }
  }

  return async function handleMessage(msg: WorkerRequest): Promise<void> {
    if (msg.type === 'close') {
      if (msg.generation >= currentGeneration) {
        currentGeneration = msg.generation;
        retireActiveHandle();
      }
      return;
    }

    const { id, generation } = msg;
    try {
      if (msg.type === 'open') {
        if (generation <= currentGeneration) {
          deps.postMessage({ type: 'error', id, error: STALE_GENERATION_ERROR });
          return;
        }
        currentGeneration = generation;

        if (!initialized) {
          await ensureInit();
          if (generation !== currentGeneration) {
            deps.postMessage({ type: 'error', id, error: STALE_GENERATION_ERROR });
            return;
          }
        }

        const handle = await deps.openVortexFile(msg.file);
        if (generation !== currentGeneration) {
          handle.free();
          deps.postMessage({ type: 'error', id, error: STALE_GENERATION_ERROR });
          return;
        }

        let result: OpenFileResult;
        try {
          result = {
            rowCount: Number(handle.row_count),
            dtype: handle.dtype,
            layoutTree: JSON.parse(handle.layout_tree()),
            segments: JSON.parse(handle.segment_map()),
            fileStructure: JSON.parse(handle.file_structure()),
          };
        } catch (err) {
          handle.free();
          throw err;
        }

        retireActiveHandle();
        activeHandle = {
          generation,
          handle,
          inFlight: 0,
          retired: false,
          disposed: false,
        };
        deps.postMessage({ type: 'result', id, data: result });
        return;
      }

      if (!initialized) {
        await ensureInit();
      }

      if (msg.type === 'fetchEncodingTree') {
        const json = await withActiveHandle(generation, (h) => h.fetch_encoding_tree(msg.nodeId));
        deps.postMessage({ type: 'result', id, data: JSON.parse(json) });
      } else if (msg.type === 'fetchArrayBuffer') {
        const buf = await withActiveHandle(generation, (h) =>
          h.fetch_array_buffer(msg.layoutNodeId, msg.arrayPath, msg.bufferIndex),
        );
        deps.postMessage({ type: 'result', id, data: buf }, [buf.buffer]);
      } else if (msg.type === 'previewArrayData') {
        const ipcBytes = await withActiveHandle(generation, (h) =>
          h.preview_array_data(msg.layoutNodeId, msg.arrayPath, msg.rowLimit),
        );
        deps.postMessage({ type: 'result', id, data: ipcBytes }, [ipcBytes.buffer]);
      } else if (msg.type === 'previewData') {
        const ipcBytes = await withActiveHandle(generation, (h) =>
          h.preview_data(msg.nodeId, msg.rowLimit),
        );
        deps.postMessage({ type: 'result', id, data: ipcBytes }, [ipcBytes.buffer]);
      }
    } catch (err) {
      deps.postMessage({
        type: 'error',
        id,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  };
}
