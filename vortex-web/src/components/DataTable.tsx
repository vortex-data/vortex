// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useRef, useState } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';

// --- Column statistics ---

interface ColumnStats {
  kind: 'numeric' | 'string' | 'boolean' | 'other';
  count: number;
  nullCount: number;
  min?: number;
  max?: number;
  mean?: number;
  histogram?: number[];
  cardinality?: number;
  trueCount?: number;
  falseCount?: number;
}

function computeStats(values: unknown[]): ColumnStats {
  const count = values.length;
  let nullCount = 0;
  const nums: number[] = [];
  let hasStrings = false;
  let trueCount = 0;
  let falseCount = 0;
  let hasBools = false;

  for (const v of values) {
    if (v == null) {
      nullCount++;
    } else if (typeof v === 'boolean') {
      hasBools = true;
      if (v) trueCount++;
      else falseCount++;
    } else if (typeof v === 'number' || typeof v === 'bigint') {
      nums.push(Number(v));
    } else if (typeof v === 'string') {
      hasStrings = true;
    }
  }

  if (hasBools) {
    return { kind: 'boolean', count, nullCount, trueCount, falseCount };
  }

  if (nums.length > 0) {
    let min = Infinity,
      max = -Infinity,
      sum = 0;
    for (const n of nums) {
      if (n < min) min = n;
      if (n > max) max = n;
      sum += n;
    }
    const bins = 20;
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
    return { kind: 'numeric', count, nullCount, min, max, mean: sum / nums.length, histogram };
  }

  if (hasStrings) {
    const uniq = new Set<string>();
    for (const v of values) if (typeof v === 'string') uniq.add(v);
    return { kind: 'string', count, nullCount, cardinality: uniq.size };
  }

  return { kind: 'other', count, nullCount };
}

// --- Sparkline ---

function SparkHistogram({ histogram, height = 12 }: { histogram: number[]; height?: number }) {
  const max = Math.max(...histogram);
  if (max === 0) return null;

  // Check if all values fell in a single bin (constant column).
  const nonZero = histogram.filter((v) => v > 0).length;
  if (nonZero <= 1) {
    // Render a flat line to indicate constant/single-value.
    const barW = 2;
    const gap = 0.5;
    const w = histogram.length * (barW + gap) - gap;
    return (
      <svg width={w} height={height} className="flex-shrink-0 opacity-50">
        <line
          x1={0}
          y1={height / 2}
          x2={w}
          y2={height / 2}
          stroke="currentColor"
          strokeWidth={1}
          className="text-vortex-grey-dark"
        />
      </svg>
    );
  }

  const barW = 2;
  const gap = 0.5;
  const w = histogram.length * (barW + gap) - gap;
  return (
    <svg width={w} height={height} className="flex-shrink-0 opacity-70">
      {histogram.map((v, i) => {
        const barH = Math.max(0.5, (v / max) * height);
        return (
          <rect
            key={i}
            x={i * (barW + gap)}
            y={height - barH}
            width={barW}
            height={barH}
            className="fill-vortex-light-blue"
          />
        );
      })}
    </svg>
  );
}

// --- Header summary (inline, compact) ---

function HeaderSummary({ stats }: { stats: ColumnStats }) {
  if (stats.kind === 'numeric') {
    const isConst = stats.min === stats.max && stats.nullCount === 0;
    if (isConst) {
      return (
        <span
          className="text-[8px] text-vortex-grey-dark font-normal opacity-70"
          title={`constant: ${stats.min}`}
        >
          const
        </span>
      );
    }
    if (stats.histogram) {
      return <SparkHistogram histogram={stats.histogram} />;
    }
  }
  if (stats.kind === 'boolean') {
    const total = stats.trueCount! + stats.falseCount!;
    if (total === 0) return null;
    const allTrue = stats.falseCount === 0 && stats.nullCount === 0;
    const allFalse = stats.trueCount === 0 && stats.nullCount === 0;
    if (allTrue || allFalse)
      return <span className="text-[8px] text-vortex-grey-dark font-normal opacity-70">const</span>;
    const pct = Math.round((stats.trueCount! / total) * 100);
    return (
      <div
        className="flex items-center gap-0.5 flex-shrink-0"
        title={`${stats.trueCount} true, ${stats.falseCount} false`}
      >
        <div className="flex h-2 w-8 rounded-sm overflow-hidden">
          <div className="bg-vortex-light-blue/60" style={{ width: `${pct}%` }} />
          <div className="bg-vortex-grey-dark/20" style={{ width: `${100 - pct}%` }} />
        </div>
        <span className="text-[8px] text-vortex-grey-dark font-normal opacity-70">{pct}%</span>
      </div>
    );
  }
  if (stats.kind === 'string' && stats.cardinality != null) {
    if (stats.cardinality === 1 && stats.nullCount === 0) {
      return <span className="text-[8px] text-vortex-grey-dark font-normal opacity-70">const</span>;
    }
    return (
      <span className="text-[8px] text-vortex-grey-dark font-normal opacity-70">
        {stats.cardinality}v
      </span>
    );
  }
  return null;
}

// --- Header tooltip (shown on hover) ---

function HeaderTooltip({ stats, approximate }: { stats: ColumnStats; approximate: boolean }) {
  const p = approximate ? '~' : '';
  const fmt = (n: number) => {
    const s =
      Math.abs(n) >= 1e6 || (Math.abs(n) < 0.01 && n !== 0)
        ? n.toExponential(2)
        : n.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return p + s;
  };

  return (
    <div className="text-[9px] font-normal text-vortex-fg-light dark:text-vortex-fg space-y-1 font-mono">
      <div className="text-vortex-grey-dark">
        {stats.count.toLocaleString()} rows{approximate ? ' (sampled)' : ''}
      </div>
      {stats.nullCount > 0 && (
        <div className="text-vortex-grey-dark">{stats.nullCount.toLocaleString()} nulls</div>
      )}
      {stats.kind === 'numeric' && (
        <>
          <div>min: {fmt(stats.min!)}</div>
          <div>max: {fmt(stats.max!)}</div>
          <div>mean: {fmt(stats.mean!)}</div>
          {stats.histogram && (
            <div className="pt-0.5">
              <SparkHistogram histogram={stats.histogram} height={24} />
            </div>
          )}
        </>
      )}
      {stats.kind === 'boolean' && (
        <>
          <div>true: {stats.trueCount!.toLocaleString()}</div>
          <div>false: {stats.falseCount!.toLocaleString()}</div>
        </>
      )}
      {stats.kind === 'string' && <div>{stats.cardinality!.toLocaleString()} distinct values</div>}
    </div>
  );
}

// --- Hoverable header with tooltip ---

function ColumnHeader({
  name,
  stats,
  approximate,
}: {
  name: string;
  stats: ColumnStats;
  approximate: boolean;
}) {
  const [showTip, setShowTip] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const onEnter = () => {
    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setShowTip(true), 400);
  };
  const onLeave = () => {
    clearTimeout(timeoutRef.current);
    setShowTip(false);
  };

  return (
    <div className="relative" onMouseEnter={onEnter} onMouseLeave={onLeave}>
      <div className="flex items-center gap-1.5">
        <span className="truncate">{name}</span>
        <HeaderSummary stats={stats} />
      </div>
      {showTip && (
        <div className="absolute top-full left-0 mt-1 z-50 bg-vortex-white dark:bg-vortex-black border border-vortex-grey-light/40 dark:border-white/[0.08] rounded shadow-lg px-2 py-1.5 min-w-[120px]">
          <HeaderTooltip stats={stats} approximate={approximate} />
        </div>
      )}
    </div>
  );
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

export type CellRenderer = (value: unknown, row: Record<string, unknown>) => React.ReactNode;

export interface DataTableProps {
  columns: string[];
  rows: Record<string, unknown>[];
  rowHeight?: number;
  onRowClick?: (rowIndex: number) => void;
  onRowHover?: (rowIndex: number | null) => void;
  /** Custom cell renderers keyed by column name. */
  cellRenderers?: Record<string, CellRenderer>;
  /** If true, stats are approximate (data was truncated by a row limit). */
  approximate?: boolean;
}

export function DataTable({
  columns,
  rows,
  rowHeight = 24,
  onRowClick,
  onRowHover,
  cellRenderers,
  approximate = false,
}: DataTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const parentRef = useRef<HTMLDivElement>(null);

  const columnStats = useMemo(() => {
    const stats: Record<string, ColumnStats> = {};
    for (const col of columns) {
      stats[col] = computeStats(rows.map((r) => r[col]));
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
          <span className="text-vortex-grey-dark tabular-nums">{info.row.index}</span>
        ),
      },
      ...columns.map(
        (col): ColumnDef<Record<string, unknown>> => ({
          accessorKey: col,
          header: () => (
            <ColumnHeader name={col} stats={columnStats[col]} approximate={approximate} />
          ),
          cell: (info) => {
            const renderer = cellRenderers?.[col];
            if (renderer) {
              return renderer(info.getValue(), info.row.original);
            }
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
                  className="px-2 py-1 text-left font-medium text-vortex-fg-light dark:text-vortex-fg border-b border-r border-vortex-grey-light/30 dark:border-white/[0.06] last:border-r-0 select-none cursor-pointer"
                  style={{ width: header.getSize() }}
                  onClick={header.column.getToggleSortingHandler()}
                >
                  <div className="flex items-center gap-1">
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
                    {header.column.getIsSorted() === 'asc' && (
                      <span className="text-vortex-light-blue">▲</span>
                    )}
                    {header.column.getIsSorted() === 'desc' && (
                      <span className="text-vortex-light-blue">▼</span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
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
                onMouseEnter={() => onRowHover?.(virtualRow.index)}
                onMouseLeave={() => onRowHover?.(null)}
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
          {virtualizer.getVirtualItems().length > 0 && (
            <tr>
              <td
                colSpan={columnDefs.length}
                style={{
                  height:
                    virtualizer.getTotalSize() - (virtualizer.getVirtualItems().at(-1)?.end ?? 0),
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
