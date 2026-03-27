// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useState, useCallback, type DragEvent } from 'react';

interface FileDropScreenProps {
  onFileLoaded: (file: File) => void;
  loading: boolean;
  error: string | null;
}

export function FileDropScreen({ onFileLoaded, loading, error }: FileDropScreenProps) {
  const [isDragging, setIsDragging] = useState(false);

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
      if (file) onFileLoaded(file);
    },
    [onFileLoaded],
  );

  const handleClick = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.vortex,.vtx';
    input.onchange = () => {
      const file = input.files?.[0];
      if (file) onFileLoaded(file);
    };
    input.click();
  }, [onFileLoaded]);

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
          isDragging ? 'bg-vortex-light-blue/10' : 'hover:bg-white/5'
        }`}
      >
        {loading ? (
          <p className="font-mono text-lg text-grey">Loading...</p>
        ) : (
          <div className="text-center">
            <p className="font-mono text-lg text-grey">
              Drop a{' '}
              <code className="rounded bg-white/10 px-1.5 py-0.5 text-vortex-light-blue">
                .vortex
              </code>{' '}
              file here
            </p>
            <p className="mt-2 font-mono text-sm text-vortex-grey-dark">or click to browse</p>
          </div>
        )}
      </div>

      {error && <p className="mt-4 max-w-lg font-mono text-sm text-vortex-red">{error}</p>}
    </div>
  );
}
