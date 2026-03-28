// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useState, useCallback, type DragEvent, type FormEvent } from 'react';
import { ThemePicker } from '../ThemePicker';

interface FileDropScreenProps {
  onFileLoaded: (file: File) => void;
  loading: boolean;
  error: string | null;
}

export function FileDropScreen({ onFileLoaded, loading, error }: FileDropScreenProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [url, setUrl] = useState('');
  const [fetchingUrl, setFetchingUrl] = useState(false);
  const [urlError, setUrlError] = useState<string | null>(null);

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

  const handleUrlSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const trimmed = url.trim();
      if (!trimmed) return;

      setFetchingUrl(true);
      setUrlError(null);
      try {
        const resp = await fetch(trimmed);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
        const blob = await resp.blob();
        const name = trimmed.split('/').pop() ?? 'remote.vortex';
        const file = new File([blob], name, { type: blob.type });
        onFileLoaded(file);
      } catch (err) {
        setUrlError(err instanceof Error ? err.message : String(err));
      } finally {
        setFetchingUrl(false);
      }
    },
    [url, onFileLoaded],
  );

  const busy = loading || fetchingUrl;

  return (
    <div className="flex min-h-screen flex-col items-center justify-center text-vortex-fg-light dark:text-vortex-fg relative">
      <div className="absolute top-3 right-3">
        <ThemePicker />
      </div>
      <h1 className="mb-8 font-funnel text-4xl font-light tracking-tight md:text-6xl">
        Vortex Explorer
      </h1>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`dashed-top dashed-bottom flex h-64 w-full max-w-lg cursor-pointer flex-col items-center justify-center transition-colors ${
          isDragging
            ? 'bg-vortex-light-blue/10'
            : 'hover:bg-vortex-black/[0.03] dark:hover:bg-white/[0.03]'
        }`}
      >
        {busy ? (
          <p className="font-mono text-lg text-vortex-grey-dark">
            {fetchingUrl ? 'Fetching…' : 'Loading…'}
          </p>
        ) : (
          <div className="text-center">
            <p className="font-mono text-lg text-vortex-grey-dark">
              Drop a{' '}
              <code className="rounded bg-vortex-black/[0.06] dark:bg-white/[0.08] px-1.5 py-0.5 text-vortex-light-blue">
                .vortex
              </code>{' '}
              file here
            </p>
            <p className="mt-2 font-mono text-sm text-vortex-grey-dark/60">or click to browse</p>
          </div>
        )}
      </div>

      {/* URL input */}
      <form
        onSubmit={handleUrlSubmit}
        className="mt-6 flex w-full max-w-lg gap-2"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/file.vortex"
          disabled={busy}
          className="flex-1 rounded border border-vortex-grey-light/40 dark:border-white/[0.08] bg-transparent px-3 py-1.5 font-mono text-sm text-vortex-fg-light dark:text-vortex-fg placeholder:text-vortex-grey-dark/40 focus:border-vortex-light-blue focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={busy || !url.trim()}
          className="rounded bg-vortex-light-blue px-4 py-1.5 font-mono text-sm text-white hover:opacity-90 disabled:opacity-40 transition-opacity"
        >
          Open
        </button>
      </form>

      {(error || urlError) && (
        <p className="mt-4 max-w-lg font-mono text-sm text-vortex-red">{urlError || error}</p>
      )}
    </div>
  );
}
