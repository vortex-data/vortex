// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useEffect, useMemo, useState } from 'react';
import { tableFromIPC } from 'apache-arrow';
import { useSelection } from '../../contexts/SelectionContext';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { parseArrayNodeId } from '../swimlane/utils';
import { DataTable } from '../DataTable';

const ROW_LIMIT = 5000;

function decodeArrow(ipcBytes: Uint8Array): {
  columns: string[];
  rows: Record<string, unknown>[];
} {
  const table = tableFromIPC(ipcBytes);
  const columns = table.schema.fields.map((f) => f.name);
  const rows: Record<string, unknown>[] = [];
  for (let i = 0; i < table.numRows; i++) {
    const row: Record<string, unknown> = {};
    for (const col of columns) {
      row[col] = table.getChild(col)?.get(i) ?? null;
    }
    rows.push(row);
  }
  return { columns, rows };
}

export function DataPreview() {
  const { state: selection } = useSelection();
  const { previewData, previewArrayData } = useVortexFile();
  const [ipcBytes, setIpcBytes] = useState<Uint8Array | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedNodeId = selection.selectedNodeId;
  const isArrayNode = selection.selectedNode?.isArrayNode ?? false;

  useEffect(() => {
    if (!selectedNodeId) {
      setIpcBytes(null);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    const fetchPromise = isArrayNode
      ? (() => {
          const { layoutNodeId, arrayPath } = parseArrayNodeId(selectedNodeId);
          return previewArrayData(layoutNodeId, arrayPath, ROW_LIMIT);
        })()
      : previewData(selectedNodeId, ROW_LIMIT);

    fetchPromise
      .then((bytes) => {
        if (!cancelled) setIpcBytes(bytes);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedNodeId, isArrayNode, previewData, previewArrayData]);

  const decoded = useMemo(() => {
    if (!ipcBytes) return null;
    try {
      return decodeArrow(ipcBytes);
    } catch (err) {
      console.error('Arrow decode error:', err);
      return null;
    }
  }, [ipcBytes]);

  if (!selectedNodeId) {
    return (
      <div className="h-full flex items-center justify-center text-[11px] text-vortex-grey-dark">
        Select a node to preview data
      </div>
    );
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center text-[11px] text-vortex-grey-dark">
        Loading preview…
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center text-[11px] text-red-500 px-4 text-center">
        {error}
      </div>
    );
  }

  if (!decoded || decoded.rows.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-[11px] text-vortex-grey-dark">
        No data
      </div>
    );
  }

  return (
    <DataTable
      columns={decoded.columns}
      rows={decoded.rows}
      approximate={decoded.rows.length >= ROW_LIMIT}
    />
  );
}
