// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useId, useRef, useState, type DragEvent, type FormEvent } from 'react';
import { Footer } from './Footer';
import { Header } from './Header';

interface FileDropScreenProps {
  onFileLoaded: (file: File) => void;
  loading: boolean;
  error: string | null;
}

/** The first screen: open a Vortex file from the disk, by drop or by choosing it, or from a URL. */
export function FileDropScreen({ onFileLoaded, loading, error }: FileDropScreenProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [url, setUrl] = useState('');
  const [fetchingUrl, setFetchingUrl] = useState(false);
  const [urlError, setUrlError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const urlId = useId();
  const errorId = useId();

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

  const handleChosen = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFileLoaded(file);
      e.target.value = '';
    },
    [onFileLoaded],
  );

  const handleUrlSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const trimmed = url.trim();
      if (!trimmed) return;

      setFetchingUrl(true);
      setUrlError(null);
      try {
        let resp: Response;
        try {
          resp = await fetch(trimmed);
        } catch (err) {
          // A network fault and a CORS refusal look the same from here.
          console.error('Explorer: the fetch failed', err);
          setUrlError(
            'The server did not let this page read the file. Download the file, then open it here.',
          );
          return;
        }
        if (!resp.ok) {
          setUrlError(
            `The file at this address did not load (HTTP ${resp.status}). Check the address.`,
          );
          return;
        }
        const blob = await resp.blob();
        const name = new URL(trimmed).pathname.split('/').pop() || 'remote.vortex';
        onFileLoaded(new File([blob], name, { type: blob.type }));
      } catch (err) {
        console.error('Explorer: the URL did not open', err);
        setUrlError('This address is not valid. Check it and try again.');
      } finally {
        setFetchingUrl(false);
      }
    },
    [url, onFileLoaded],
  );

  const busy = loading || fetchingUrl;

  return (
    <div className="x-app is-page">
      <a className="vx-skip-link" href="#main">
        Skip to content
      </a>
      <Header />
      <main id="main" tabIndex={-1} className="x-open has-dot-field">
        <div className="vx-dot-field" aria-hidden="true" />
        <div className="x-open-body">
          <div className="x-open-head">
            <h1 className="vx-display-mono">Open a Vortex file.</h1>
            <p className="vx-body">
              See its layout, encodings, and rows. The file stays on your computer: the Explorer
              reads it in this tab.
            </p>
          </div>

          {/* The drop target. Its button is the keyboard and pointer path; the drop is extra. */}
          {/* biome-ignore lint/a11y/noStaticElementInteractions: a drop target only; the button inside opens the file. */}
          <div
            className={`vx-card x-drop${isDragging ? ' is-over' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <p className="vx-mono" aria-live="polite">
              {busy
                ? fetchingUrl
                  ? 'Fetching the file…'
                  : 'Reading the file…'
                : 'Drop a .vortex file here'}
            </p>
            <button
              type="button"
              className="vx-btn vx-btn-primary"
              disabled={busy}
              onClick={() => inputRef.current?.click()}
            >
              Choose a file
            </button>
            <input
              ref={inputRef}
              className="vx-sr-only"
              type="file"
              accept=".vortex,.vtx"
              tabIndex={-1}
              aria-hidden="true"
              onChange={handleChosen}
            />
            <p className="vx-label-sm">.vortex or .vtx</p>
            {error && (
              <div className="x-file-error" role="alert">
                <p className="vx-small x-error">
                  The file did not open. Open a .vortex or .vtx file.
                </p>
                <p className="vx-label-sm">
                  Reason · <span className="vx-ref">{error}</span>
                </p>
              </div>
            )}
          </div>

          <form className="x-url" onSubmit={handleUrlSubmit}>
            <label className="vx-label" htmlFor={urlId}>
              Or open a file from a URL
            </label>
            <div className="vx-field-row">
              <input
                id={urlId}
                className="vx-field"
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/orders.vortex"
                autoComplete="off"
                disabled={busy}
                {...(urlError ? { 'aria-invalid': true, 'aria-describedby': errorId } : {})}
              />
              <button
                type="submit"
                className="vx-btn vx-btn-secondary"
                disabled={busy || !url.trim()}
              >
                Open
              </button>
            </div>
            {urlError && (
              <p className="vx-small vx-field-error" id={errorId} role="alert">
                {urlError}
              </p>
            )}
          </form>
        </div>
      </main>
      <Footer />
    </div>
  );
}
