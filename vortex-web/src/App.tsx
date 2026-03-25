// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import type { InitOutput } from "./wasm/pkg/vortex_web_wasm.d.ts";

interface FileInfo {
  name: string;
  rowCount: bigint;
  dtype: string;
}

function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const wasmRef = useRef<InitOutput | null>(null);

  useEffect(() => {
    import("./wasm/pkg/vortex_web_wasm.js").then(async (wasm) => {
      wasmRef.current = await wasm.default();
    });
  }, []);

  const openFile = useCallback(
    async (file: File) => {
      setError(null);
      setLoading(true);
      try {
        const wasm = await import("./wasm/pkg/vortex_web_wasm.js");
        if (!wasmRef.current) {
          wasmRef.current = await wasm.default();
        }
        const bytes = new Uint8Array(await file.arrayBuffer());
        const handle = wasm.open_vortex_file(bytes);
        setFileInfo({
          name: file.name,
          rowCount: handle.row_count,
          dtype: handle.dtype,
        });
        handle.free();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setFileInfo(null);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) {
        openFile(file);
      }
    },
    [openFile],
  );

  const handleClick = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".vortex,.vtx";
    input.onchange = () => {
      const file = input.files?.[0];
      if (file) {
        openFile(file);
      }
    };
    input.click();
  }, [openFile]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center text-white">
      <h1 className="mb-8 font-funnel text-4xl font-light tracking-tight md:text-6xl">
        Vortex Explorer
      </h1>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`dashed-top dashed-bottom flex h-64 w-full max-w-lg cursor-pointer flex-col items-center justify-center transition-colors ${
          isDragging ? "bg-vortex-light-blue/10" : "hover:bg-white/5"
        }`}
      >
        {loading ? (
          <p className="font-mono text-lg text-grey">Loading...</p>
        ) : fileInfo ? (
          <div className="text-center">
            <p className="font-mono text-lg text-white">{fileInfo.name}</p>
            <p className="mt-4 font-mono text-sm text-grey">
              <span className="text-vortex-light-blue">
                {fileInfo.rowCount.toLocaleString()}
              </span>{" "}
              rows
            </p>
            <pre className="mt-4 max-w-md overflow-x-auto rounded bg-white/5 px-4 py-2 text-left font-mono text-xs text-vortex-grey-light">
              {fileInfo.dtype}
            </pre>
          </div>
        ) : (
          <div className="text-center">
            <p className="font-mono text-lg text-grey">
              Drop a{" "}
              <code className="rounded bg-white/10 px-1.5 py-0.5 text-vortex-light-blue">
                .vortex
              </code>{" "}
              file here
            </p>
            <p className="mt-2 font-mono text-sm text-vortex-grey-dark">
              or click to browse
            </p>
          </div>
        )}
      </div>

      {error && (
        <p className="mt-4 max-w-lg font-mono text-sm text-vortex-red">
          {error}
        </p>
      )}
    </div>
  );
}

export default App;
