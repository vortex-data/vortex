// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useRef } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useState } from 'react';

// --- Column statistics and sparkline helpers ---

interface ColumnStats {
  kind: 'numeric' | 'string' | 'other';
  count: number;
  nullCount: number;
  // numeric
  min?: number;
  max?: number;
  mean?: number;
  histogram?: number[];
  // string
  cardinality?: number;
  topValues?: [string, number][];
}

function computeStats(values: unknown[]): ColumnStats {
  const count = values.length;
  let nullCount = 0;
  const nums: number[] = [];
  const strings: string[] = [];

  for (const v of values) {
    if (v == null) {
      nullCount++;
      continue;
    }
    if (typeof v === 'number' || typeof v === 'bigint') {
      nums.push(Number(v));
    } else if (typeof v === 'string') {
      strings.push(v);
    }
  }

  if (nums.length > 0) {
    let min = Infinity, max = -Infinity, sum = 0;
    for (const n of nums) {
      if (n < min) min = n;
      if (n > max) max = n;
      sum += n;
    }
    const mean = sum / nums.length;

    // Build histogram (16 bins).
    const bins = 16;
    const histogram = new Array<number>(bins).fill(0);
    if (max > min) {
      const range = max - min;
      for (const n of nums) {
        const idx = Math.min(bins - 1, Math.floor(((n - min) / range) * bins));
        histogram[idx]++;
      }
    } else {
      histogram[0] = nums.length;
    }

    return { kind: 'numeric', count, nullCount, min, max, mean, histogram };
  }

  if (strings.length > 0) {
    const freq = new Map<string, number>();
    for (const s of strings) {
      freq.set(s, (freq.get(s) ?? 0) + 1);
    }
    const topValues = [...freq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5) as [string, number][];
    return { kind: 'string', count, nullCount, cardinality: freq.size, topValues };
  }

  return { kind: 'other', count, nullCount };
}

function SparkHistogram({ histogram }: { histogram: number[] }) {
  const max = Math.max(...histogram);
  if (max === 0) return null;
  const barW = 3;
  const gap = 1;
  const h = 16;
  const w = histogram.length * (barW + gap) - gap;

  return (
    <svg width={w} height={h} className="inline-block align-middle ml-1">
      {histogram.map((v, i) => {
        const barH = (v / max) * h;
        return (
          <rect
            key={i}
            x={i * (barW + gap)}
            y={h - barH}
            width={barW}
            height={barH}
            className="fill-vortex-light-blue/60"
          />
        );
      })}
    </svg>
  );
}

function SparkBar({ values }: { values: [string, number][] }) {
  const total = values.reduce((s, [, c]) => s + c, 0);
  if (total === 0) return null;

  return (
    <div className="flex h-2.5 rounded-sm overflow-hidden mt-0.5 gap-px">
      {values.map(([label, count], i) => {
        const pct = (count / total) * 100;
        const colors = [
          'bg-vortex-light-blue/60',
          'bg-vortex-light-blue/40',
          'bg-vortex-light-blue/25',
          'bg-vortex-grey-dark/30',
          'bg-vortex-grey-dark/20',
        ];
        return (
          <div
            key={label}
            className={`${colors[i]} rounded-sm`}
            style={{ width: `${pct}%` }}
            title={`${label}: ${count}`}
          />
        );
      })}
    </div>
  );
}

function ColumnSummary({ stats }: { stats: ColumnStats }) {
  if (stats.kind === 'numeric') {
    const fmt = (n: number) =>
      Math.abs(n) >= 1e6 || (Math.abs(n) < 0.01 && n !== 0)
        ? n.toExponential(1)
        : n.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return (
      <div className="flex items-center gap-1.5 text-[9px] text-vortex-grey-dark font-normal whitespace-nowrap mt-0.5">
        <span title="min">{fmt(stats.min!)}</span>
        <span className="opacity-40">–</span>
        <span title="max">{fmt(stats.max!)}</span>
        <span className="opacity-40">μ</span>
        <span title="mean">{fmt(stats.mean!)}</span>
        {stats.histogram && <SparkHistogram histogram={stats.histogram} />}
      </div>
    );
  }
  if (stats.kind === 'string') {
    return (
      <div className="text-[9px] text-vortex-grey-dark font-normal mt-0.5">
        <span>{stats.cardinality} distinct</span>
        {stats.nullCount > 0 && (
          <span className="ml-1 opacity-60">({stats.nullCount} null)</span>
        )}
        {stats.topValues && stats.topValues.length > 0 && (
          <SparkBar values={stats.topValues} />
        )}
      </div>
    );
  }
  return null;
}

// --- Formatting ---

function formatCell(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  return String(value);
}

// --- Main component ---

export interface DataTableProps {
  /** Column definitions: name and optional type hint. */
  columns: string[];
  /** Row data as array of arrays (column-major access via columnData) or row objects. */
  rows: Record<string, unknown>[];
  /** Fixed row height in px. */
  rowHeight?: number;
  /** If provided, called when a row is clicked. */
  onRowClick?: (rowIndex: number) => void;
}

export function DataTable({
  columns,
  rows,
  rowHeight = 24,
  onRowClick,
}: DataTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const parentRef = useRef<HTMLDivElement>(null);

  // Compute per-column stats.
  const columnStats = useMemo(() => {
    const stats: Record<string, ColumnStats> = {};
    for (const col of columns) {
      const values = rows.map((r) => r[col]);
      stats[col] = computeStats(values);
    }
    return stats;
  }, [columns, rows]);

  const columnDefs = useMemo<ColumnDef<Record<string, unknown>>[]>(
    () => [
      {
        id: '__row_num',
        header: '#',
        size: 50,
        enableSorting: false,
        cell: (info) => (
          <span className="text-vortex-grey-dark tabular-nums">
            {info.row.index}
          </span>
        ),
      },
      ...columns.map(
        (col): ColumnDef<Record<string, unknown>> => ({
          accessorKey: col,
          header: () => (
            <div>
              <div>{col}</div>
              <ColumnSummary stats={columnStats[col]} />
            </div>
          ),
          cell: (info) => {
            const val = info.getValue();
            if (val == null) {
              return <span className="text-vortex-grey-dark italic">null</span>;
            }
            return formatCell(val);
          },
          sortingFn: 'auto',
        }),
      ),
    ],
    [columns, columnStats],
  );

  const table = useReactTable({
    data: rows,
    columns: columnDefs,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const { rows: tableRows } = table.getRowModel();

  const virtualizer = useVirtualizer({
    count: tableRows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => rowHeight,
    overscan: 20,
  });

  return (
    <div ref={parentRef} className="h-full w-full overflow-auto">
      <table className="w-full text-[10px] font-mono border-collapse">
        <thead className="sticky top-0 z-10">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id} className="bg-vortex-grey-lightest dark:bg-vortex-black">
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  className="px-2 py-1 text-left font-medium text-vortex-fg-light dark:text-vortex-fg border-b border-r border-vortex-grey-light/30 dark:border-white/[0.06] last:border-r-0 select-none"
                  style={{ width: header.getSize() }}
                  onClick={header.column.getToggleSortingHandler()}
                  role={header.column.getCanSort() ? 'button' : undefined}
                >
                  <div className="flex items-center gap-1">
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
                    {header.column.getIsSorted() === 'asc' && ' ▲'}
                    {header.column.getIsSorted() === 'desc' && ' ▼'}
                  </div>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {/* Spacer for virtual scroll offset */}
          {virtualizer.getVirtualItems().length > 0 && (
            <tr>
              <td
                colSpan={columnDefs.length}
                style={{ height: virtualizer.getVirtualItems()[0]?.start ?? 0, padding: 0 }}
              />
            </tr>
          )}
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const row = tableRows[virtualRow.index];
            return (
              <tr
                key={row.id}
                className="border-b border-vortex-grey-light/20 dark:border-white/[0.03] hover:bg-vortex-grey-lightest/50 dark:hover:bg-white/[0.02] cursor-default"
                style={{ height: rowHeight }}
                onClick={() => onRowClick?.(virtualRow.index)}
              >
                {row.getVisibleCells().map((cell) => (
                  <td
                    key={cell.id}
                    className="px-2 text-vortex-fg-light dark:text-vortex-fg/80 truncate max-w-[300px] border-r border-vortex-grey-light/30 dark:border-white/[0.06] last:border-r-0"
                    title={formatCell(cell.getValue())}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            );
          })}
          {/* Bottom spacer */}
          {virtualizer.getVirtualItems().length > 0 && (
            <tr>
              <td
                colSpan={columnDefs.length}
                style={{
                  height:
                    virtualizer.getTotalSize() -
                    (virtualizer.getVirtualItems().at(-1)?.end ?? 0),
                  padding: 0,
                }}
              />
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
