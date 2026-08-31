// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useState } from 'react';
import type { VortexFileState } from '../../contexts/VortexFileContext';
import { diffLayoutTrees, flattenDiff, type DiffStatus } from './diff';

function bytes(value: number): string {
  if (Math.abs(value) < 1024) return `${value} B`;
  if (Math.abs(value) < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / 1024 ** 2).toFixed(2)} MiB`;
}

function delta(before: number, after: number): string {
  const difference = after - before;
  const percent = before === 0 ? null : (difference / before) * 100;
  return `${difference > 0 ? '+' : ''}${bytes(difference)}${percent === null ? '' : ` (${percent > 0 ? '+' : ''}${percent.toFixed(1)}%)`}`;
}

const STATUS_CLASS: Record<DiffStatus, string> = {
  unchanged: 'text-vortex-grey-dark',
  changed: 'text-vortex-orange',
  added: 'text-vortex-green',
  removed: 'text-vortex-red',
};

function Metric({ label, before, after }: { label: string; before: number; after: number }) {
  const improved = after <= before;
  return (
    <div className="rounded-md border border-vortex-grey-light/50 dark:border-white/[0.08] p-3">
      <div className="text-[10px] uppercase tracking-wide text-vortex-grey-dark">{label}</div>
      <div className="mt-1 font-mono text-lg text-vortex-fg-light dark:text-vortex-fg">
        {bytes(after)}
      </div>
      <div
        className={`mt-0.5 font-mono text-xs ${improved ? 'text-vortex-green' : 'text-vortex-red'}`}
      >
        {delta(before, after)}
      </div>
      <div className="mt-1 text-[10px] text-vortex-grey-dark">was {bytes(before)}</div>
    </div>
  );
}

export function CompareView({
  baseline,
  candidate,
  onViewBaseline,
  onViewCandidate,
  onReplaceBaseline,
  onReplaceCandidate,
}: {
  baseline: VortexFileState;
  candidate: VortexFileState;
  onViewBaseline?: () => void;
  onViewCandidate?: () => void;
  onReplaceBaseline?: (file: File) => void;
  onReplaceCandidate?: (file: File) => void;
}) {
  const [changesOnly, setChangesOnly] = useState(true);
  const root = useMemo(
    () => diffLayoutTrees(baseline.layoutTree, candidate.layoutTree),
    [baseline.layoutTree, candidate.layoutTree],
  );
  const rows = useMemo(() => flattenDiff(root, changesOnly), [root, changesOnly]);

  return (
    <div className="flex-1 min-h-0 overflow-auto p-4 text-vortex-fg-light dark:text-vortex-fg">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h2 className="font-funnel text-xl">Compression comparison</h2>
          <p className="mt-1 font-mono text-xs text-vortex-grey-dark">
            {baseline.fileName} → {candidate.fileName}
          </p>
        </div>
        <label className="flex items-center gap-2 text-xs text-vortex-grey-dark">
          <input
            type="checkbox"
            checked={changesOnly}
            onChange={(event) => setChangesOnly(event.target.checked)}
          />
          Changes only
        </label>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <FileControls
          label="Previous"
          fileName={baseline.fileName}
          onView={onViewBaseline}
          onReplace={onReplaceBaseline}
        />
        <FileControls
          label="New"
          fileName={candidate.fileName}
          onView={onViewCandidate}
          onReplace={onReplaceCandidate}
        />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Metric label="File size" before={baseline.fileSize} after={candidate.fileSize} />
        <Metric
          label="Data bytes"
          before={baseline.fileStructure.totalDataBytes}
          after={candidate.fileStructure.totalDataBytes}
        />
        <Metric
          label="Metadata bytes"
          before={baseline.fileStructure.totalMetadataBytes}
          after={candidate.fileStructure.totalMetadataBytes}
        />
      </div>

      {baseline.rowCount !== candidate.rowCount || baseline.dtype !== candidate.dtype ? (
        <div className="mt-4 rounded-md border border-vortex-red/40 bg-vortex-red/5 p-3 text-xs text-vortex-red">
          Inputs differ: baseline has {baseline.rowCount.toLocaleString()} rows and candidate has{' '}
          {candidate.rowCount.toLocaleString()} rows
          {baseline.dtype !== candidate.dtype ? '; their schemas also differ' : ''}. Size deltas may
          not isolate a compression change.
        </div>
      ) : null}

      <div className="mt-5 overflow-hidden rounded-md border border-vortex-grey-light/50 dark:border-white/[0.08]">
        <div className="grid grid-cols-[minmax(180px,1.2fr)_minmax(150px,1fr)_minmax(150px,1fr)_110px_110px] gap-3 border-b border-vortex-grey-light/50 dark:border-white/[0.08] bg-vortex-grey-lightest/60 dark:bg-white/[0.03] px-3 py-2 text-[10px] uppercase tracking-wide text-vortex-grey-dark">
          <span>Node</span>
          <span>Previous</span>
          <span>New</span>
          <span>Metadata Δ</span>
          <span>Buffers Δ</span>
        </div>
        {rows.map((row) => (
          <div
            key={row.key}
            className="grid grid-cols-[minmax(180px,1.2fr)_minmax(150px,1fr)_minmax(150px,1fr)_110px_110px] gap-3 border-b last:border-b-0 border-vortex-grey-light/30 dark:border-white/[0.05] px-3 py-1.5 font-mono text-xs"
          >
            <span className={STATUS_CLASS[row.status]} style={{ paddingLeft: row.depth * 14 }}>
              <span className="mr-2 inline-block w-14 text-[9px] uppercase">{row.status}</span>
              {row.label}
            </span>
            <span className="truncate text-vortex-grey-dark" title={row.beforeEncoding}>
              {row.beforeEncoding ?? '—'}
            </span>
            <span className="truncate" title={row.afterEncoding}>
              {row.afterEncoding ?? '—'}
            </span>
            <span
              className={
                row.beforeMetadataBytes === row.afterMetadataBytes ? 'text-vortex-grey-dark' : ''
              }
            >
              {delta(row.beforeMetadataBytes, row.afterMetadataBytes)}
            </span>
            <span
              className={
                row.beforeBufferBytes === row.afterBufferBytes ? 'text-vortex-grey-dark' : ''
              }
            >
              {delta(row.beforeBufferBytes, row.afterBufferBytes)}
            </span>
          </div>
        ))}
        {rows.length === 0 ? (
          <div className="p-6 text-center text-sm text-vortex-grey-dark">No changes</div>
        ) : null}
      </div>
    </div>
  );
}

function FileControls({
  label,
  fileName,
  onView,
  onReplace,
}: {
  label: string;
  fileName: string;
  onView?: () => void;
  onReplace?: (file: File) => void;
}) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-vortex-grey-light/50 dark:border-white/[0.08] p-3">
      <div className="min-w-0 flex-1">
        <div className="text-[10px] uppercase tracking-wide text-vortex-grey-dark">{label}</div>
        <div className="truncate font-mono text-xs" title={fileName}>
          {fileName}
        </div>
      </div>
      <button
        type="button"
        className="rounded px-2 py-1 text-xs text-vortex-blue hover:bg-vortex-grey-lightest dark:hover:bg-white/[0.06]"
        onClick={onView}
      >
        View
      </button>
      <label className="cursor-pointer rounded px-2 py-1 text-xs text-vortex-grey-dark hover:bg-vortex-grey-lightest dark:hover:bg-white/[0.06]">
        Replace…
        <input
          className="hidden"
          type="file"
          accept=".vortex,.vtx"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onReplace?.(file);
            event.target.value = '';
          }}
        />
      </label>
    </div>
  );
}
