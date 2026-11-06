// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useRef, useState } from 'react';
import {
  columnSizingFeature,
  createSortedRowModel,
  flexRender,
  rowSortingFeature,
  sortFn_alphanumeric,
  sortFn_datetime,
  sortFn_text,
  tableFeatures,
  useTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Facts } from './detail/Facts';

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
        histogram[idx] = (histogram[idx] ?? 0) + 1;
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

const BAR = 2;
const GAP = 0.5;

/** A small histogram of a numeric column. A column with one value draws a flat line. */
function SparkHistogram({ histogram, height = 12 }: { histogram: number[]; height?: number }) {
  const max = Math.max(...histogram);
  if (max === 0) return null;
  const w = histogram.length * (BAR + GAP) - GAP;
  const flat = histogram.filter((v) => v > 0).length <= 1;
  return (
    <svg aria-hidden="true" width={w} height={height} className="x-spark">
      {flat ? (
        <line x1={0} y1={height / 2} x2={w} y2={height / 2} strokeWidth={1} />
      ) : (
        histogram.map((v, i) => {
          const barH = Math.max(0.5, (v / max) * height);
          return <rect key={i} x={i * (BAR + GAP)} y={height - barH} width={BAR} height={barH} />;
        })
      )}
    </svg>
  );
}

// --- Header summary (inline, compact) ---

function HeaderSummary({ stats }: { stats: ColumnStats }) {
  const note = (text: string, title?: string) => (
    <span className="vx-label-sm x-col-summary" title={title}>
      {text}
    </span>
  );
  if (stats.kind === 'numeric') {
    if (stats.min === stats.max && stats.nullCount === 0)
      return note('const', `Every value is ${stats.min}`);
    if (stats.histogram) return <SparkHistogram histogram={stats.histogram} />;
  }
  if (stats.kind === 'boolean') {
    const t = stats.trueCount ?? 0;
    const f = stats.falseCount ?? 0;
    if (t + f === 0) return null;
    if ((t === 0 || f === 0) && stats.nullCount === 0) return note('const');
    const pct = Math.round((t / (t + f)) * 100);
    return (
      <span className="x-col-summary" title={`${t} true, ${f} false`}>
        <span className="x-share" aria-hidden="true">
          <span style={{ width: `${pct}%` }} />
        </span>{' '}
        <span className="vx-label-sm">{pct}%</span>
      </span>
    );
  }
  if (stats.kind === 'string' && stats.cardinality != null) {
    if (stats.cardinality === 1 && stats.nullCount === 0) return note('const');
    return note(`${stats.cardinality.toLocaleString('en-US')} distinct`);
  }
  return null;
}

// --- Header tooltip (shown on hover or focus) ---

function HeaderTooltip({ stats, approximate }: { stats: ColumnStats; approximate: boolean }) {
  const p = approximate ? '~' : '';
  const fmt = (n: number) =>
    p +
    (Math.abs(n) >= 1e6 || (Math.abs(n) < 0.01 && n !== 0)
      ? n.toExponential(2)
      : n.toLocaleString('en-US', { maximumFractionDigits: 2 }));
  const items: [string, string][] = [
    ['Rows', `${stats.count.toLocaleString('en-US')}${approximate ? ' (sample)' : ''}`],
  ];
  if (stats.nullCount > 0) items.push(['Nulls', stats.nullCount.toLocaleString('en-US')]);
  if (stats.kind === 'numeric') {
    items.push(
      ['Min', fmt(stats.min ?? 0)],
      ['Max', fmt(stats.max ?? 0)],
      ['Mean', fmt(stats.mean ?? 0)],
    );
  }
  if (stats.kind === 'boolean') {
    items.push(['True', (stats.trueCount ?? 0).toLocaleString('en-US')]);
    items.push(['False', (stats.falseCount ?? 0).toLocaleString('en-US')]);
  }
  if (stats.kind === 'string')
    items.push(['Distinct', (stats.cardinality ?? 0).toLocaleString('en-US')]);
  return (
    <div className="x-tip x-col-tip" role="tooltip">
      <Facts items={items} />
      {stats.kind === 'numeric' && stats.histogram && (
        <div className="x-pad">
          <SparkHistogram histogram={stats.histogram} height={24} />
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
    if (Number.isInteger(value)) return value.toLocaleString('en-US');
    return value.toLocaleString('en-US', { maximumFractionDigits: 4 });
  }
  return String(value);
}

// --- Main component ---

// Sorting by column header, and column widths. 'auto' sorting picks one of the
// registered sort functions from the column's values, and falls back to basic.
const features = tableFeatures({
  rowSortingFeature,
  columnSizingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns: { alphanumeric: sortFn_alphanumeric, datetime: sortFn_datetime, text: sortFn_text },
});

type Row = Record<string, unknown>;

export type CellRenderer = (value: unknown, row: Record<string, unknown>) => React.ReactNode;

export interface DataTableProps {
  /** The table's accessible name. */
  label: string;
  columns: string[];
  rows: Record<string, unknown>[];
  onRowClick?: (rowIndex: number) => void;
  onRowHover?: (rowIndex: number | null) => void;
  /** Custom cell renderers keyed by column name. */
  cellRenderers?: Record<string, CellRenderer>;
  /** Columns to set right-aligned, as numbers. The preview finds them from the values. */
  numeric?: ReadonlySet<string>;
  /** If true, stats are approximate (data was truncated by a row limit). */
  approximate?: boolean;
}

/** The row height the system sets for a data table (--app-row). */
const ROW_HEIGHT = 36;

/**
 * A data table on the system's vx-table: sticky heads that sort, virtualised rows, and a short
 * summary of each column in its head, with the column's numbers on hover or focus.
 */
export function DataTable({
  label,
  columns,
  rows,
  onRowClick,
  onRowHover,
  cellRenderers,
  numeric,
  approximate = false,
}: DataTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [tipColumn, setTipColumn] = useState<string | null>(null);
  const tipTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const parentRef = useRef<HTMLDivElement>(null);

  const columnStats = useMemo(() => {
    const stats: Record<string, ColumnStats> = {};
    for (const col of columns) stats[col] = computeStats(rows.map((r) => r[col]));
    return stats;
  }, [columns, rows]);

  const isNum = (col: string) => numeric?.has(col) ?? columnStats[col]?.kind === 'numeric';

  const columnDefs = useMemo<ColumnDef<typeof features, Row>[]>(
    () => [
      {
        id: '__row_num',
        header: '#',
        size: 56,
        enableSorting: false,
        cell: (info) => <span className="x-row-num">{info.row.index}</span>,
      },
      ...columns.map(
        (col): ColumnDef<typeof features, Row> => ({
          accessorKey: col,
          header: col,
          cell: (info) => {
            const renderer = cellRenderers?.[col];
            if (renderer) return renderer(info.getValue(), info.row.original);
            const val = info.getValue();
            if (val == null) return <span className="x-null">null</span>;
            return formatCell(val);
          },
          sortFn: 'auto',
        }),
      ),
    ],
    [cellRenderers, columns],
  );

  const table = useTable({
    features,
    data: rows,
    columns: columnDefs,
    state: { sorting },
    onSortingChange: setSorting,
  });

  const { rows: tableRows } = table.getRowModel();

  const virtualizer = useVirtualizer({
    count: tableRows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 20,
  });
  const items = virtualizer.getVirtualItems();

  const showTip = (col: string | null, delay = 400) => {
    clearTimeout(tipTimer.current);
    if (col === null) setTipColumn(null);
    else tipTimer.current = setTimeout(() => setTipColumn(col), delay);
  };

  return (
    <div ref={parentRef} className="x-table-scroll">
      <table className="vx-table is-data" aria-label={label} aria-rowcount={tableRows.length + 1}>
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => {
                const col = header.column.id;
                const sorted = header.column.getIsSorted();
                const stats = columnStats[col];
                const canSort = header.column.getCanSort();
                const title = flexRender(header.column.columnDef.header, header.getContext());
                return (
                  <th
                    key={header.id}
                    scope="col"
                    className={col !== '__row_num' && isNum(col) ? 'is-num' : undefined}
                    style={{ width: header.getSize() }}
                    aria-sort={
                      sorted === 'asc' ? 'ascending' : sorted === 'desc' ? 'descending' : undefined
                    }
                    onPointerEnter={stats ? () => showTip(col) : undefined}
                    onPointerLeave={stats ? () => showTip(null) : undefined}
                  >
                    {canSort ? (
                      <button
                        type="button"
                        onClick={header.column.getToggleSortingHandler()}
                        onFocus={stats ? () => showTip(col, 0) : undefined}
                        onBlur={stats ? () => showTip(null) : undefined}
                        title={`Sort by ${col}`}
                      >
                        {title}
                        {stats && <HeaderSummary stats={stats} />}
                        {sorted && <span aria-hidden="true">{sorted === 'asc' ? '↑' : '↓'}</span>}
                      </button>
                    ) : (
                      title
                    )}
                    {stats && tipColumn === col && (
                      <HeaderTooltip stats={stats} approximate={approximate} />
                    )}
                  </th>
                );
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {items.length > 0 && (
            <tr className="is-spacer">
              <td colSpan={columnDefs.length} style={{ height: items[0]?.start ?? 0 }} />
            </tr>
          )}
          {items.map((virtualRow) => {
            const row = tableRows[virtualRow.index];
            if (!row) return null;
            const open = onRowClick ? () => onRowClick(virtualRow.index) : undefined;
            return (
              <tr
                key={row.id}
                aria-rowindex={virtualRow.index + 2}
                className={open ? 'is-link' : undefined}
                tabIndex={open ? 0 : undefined}
                onClick={open}
                onKeyDown={
                  open
                    ? (e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          open();
                        }
                      }
                    : undefined
                }
                onPointerEnter={onRowHover ? () => onRowHover(virtualRow.index) : undefined}
                onPointerLeave={onRowHover ? () => onRowHover(null) : undefined}
              >
                {row.getAllCells().map((cell) => (
                  <td
                    key={cell.id}
                    className={
                      cell.column.id !== '__row_num' && isNum(cell.column.id) ? 'is-num' : undefined
                    }
                    title={formatCell(cell.getValue()) || undefined}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            );
          })}
          {items.length > 0 && (
            <tr className="is-spacer">
              <td
                colSpan={columnDefs.length}
                style={{ height: virtualizer.getTotalSize() - (items.at(-1)?.end ?? 0) }}
              />
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
