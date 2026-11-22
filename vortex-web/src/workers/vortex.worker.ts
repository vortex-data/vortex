// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// <reference lib="webworker" />

import init, { open_vortex_file } from '../wasm/pkg/vortex_web_wasm.js';
import { createVortexWorkerHandler, type WorkerRequest } from './workerRuntime';

const handleMessage = createVortexWorkerHandler({
  init,
  openVortexFile: open_vortex_file,
  postMessage: (msg, transfer) => {
    if (transfer) {
      self.postMessage(msg, transfer);
    } else {
      self.postMessage(msg);
    }
  },
});

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  void handleMessage(e.data);
};
