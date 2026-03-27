// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// <reference lib="webworker" />

import init, { open_vortex_file } from '../wasm/pkg/vortex_web_wasm.js';

let initialized = false;

self.onmessage = async (e: MessageEvent) => {
  const { type, id, file } = e.data;
  if (type === 'open') {
    try {
      if (!initialized) {
        await init();
        initialized = true;
      }
      const handle = await open_vortex_file(file);
      try {
        const result = {
          rowCount: Number(handle.row_count),
          dtype: handle.dtype,
          layoutTree: JSON.parse(handle.layout_tree()),
          segments: JSON.parse(handle.segment_map()),
          fileStructure: JSON.parse(handle.file_structure()),
        };
        self.postMessage({ type: 'result', id, data: result });
      } finally {
        handle.free();
      }
    } catch (err) {
      self.postMessage({ type: 'error', id, error: String(err) });
    }
  }
};
