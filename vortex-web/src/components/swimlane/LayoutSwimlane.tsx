import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import type {
  LayoutNode,
  Split,
  FlattenedNode,
  ChunkNode,
  ZoneNode,
  FlatLayout,
} from './types';
import {
  LAYOUT_STYLES,
  DTYPE_COLORS,
  DTYPE_CATEGORIES,
  ROW_HEIGHT,
  MIN_LABEL_WIDTH,
  getDtypeCategory,
  rangesOverlap,
  createSplits,
  getSelectedRowRange,
  groupChunks,
  formatBytes,
  formatRowRange,
  formatRowCount,
  hasExpandableChildren,
} from './utils';

// ============================================================================
// Props
// ============================================================================

export interface LayoutSwimlaneProps {
  /** The root layout node to visualize */
  layout: LayoutNode;
  /** Total number of rows in the dataset */
  totalRows: number;
  /** Optional file name to display */
  fileName?: string;
  /** Initially expanded node IDs */
  defaultExpanded?: string[];
  /** Callback when splits are selected */
  onSplitSelect?: (selectedSplits: Split[]) => void;
  /** Minimum width of the swimlane panel */
  swimlaneMinWidth?: number;
  /** Height of the main scrollable area */
  height?: number;
}

// ============================================================================
// Tooltip Component
// ============================================================================

interface TooltipProps {
  node: FlatLayout;
  position: { x: number; y: number };
}

function Tooltip({ node, position }: TooltipProps) {
  const meta = node.meta;
  const rows = node.rowRange[1] - node.rowRange[0];
  const dtypeCat = getDtypeCategory(meta?.dtype);
  const dtypeColor = DTYPE_COLORS[dtypeCat];

  return (
    <div
      className="fixed z-[1000] pointer-events-none max-w-[220px] rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-lg p-2 text-xs"
      style={{ left: position.x + 12, top: position.y - 10 }}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <span className="font-medium text-gray-900 dark:text-gray-100">{node.name}</span>
        <span
          className="text-[9px] px-1.5 py-0.5 rounded"
          style={{ color: dtypeColor, backgroundColor: `${dtypeColor}20` }}
        >
          {dtypeCat}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-gray-500 dark:text-gray-400">
        <span>rows</span>
        <span className="text-gray-900 dark:text-gray-100">{rows.toLocaleString()}</span>
        {meta?.dtype && (
          <>
            <span>dtype</span>
            <span className="text-gray-900 dark:text-gray-100">{meta.dtype}</span>
          </>
        )}
        {meta?.bytes && (
          <>
            <span>size</span>
            <span className="text-gray-900 dark:text-gray-100">{formatBytes(meta.bytes)}</span>
          </>
        )}
        {meta?.min !== undefined && (
          <>
            <span>min</span>
            <span className="text-gray-900 dark:text-gray-100">{String(meta.min)}</span>
          </>
        )}
        {meta?.max !== undefined && (
          <>
            <span>max</span>
            <span className="text-gray-900 dark:text-gray-100">{String(meta.max)}</span>
          </>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Tree Row Component
// ============================================================================

interface TreeRowProps {
  node: FlattenedNode['node'];
  depth: number;
  isExpanded: boolean;
  isGroup?: boolean;
  isHint?: boolean;
  isHiddenIndicator?: boolean;
  onToggle: () => void;
}

function TreeRow({
  node,
  depth,
  isExpanded,
  isGroup,
  isHint,
  isHiddenIndicator,
  onToggle,
}: TreeRowProps) {
  if (isHint || isHiddenIndicator) {
    return (
      <div
        className="flex items-center text-[10px] text-gray-400 dark:text-gray-500 italic border-b border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900"
        style={{ height: ROW_HEIGHT, paddingLeft: 6 + depth * 10 }}
      >
        <span className="ml-3">{node.name}</span>
      </div>
    );
  }

  const style = LAYOUT_STYLES[node.type as keyof typeof LAYOUT_STYLES] || LAYOUT_STYLES.chunk;
  const expandable = hasExpandableChildren(node);
  const isFlat = node.type === 'flat';
  const isPartition = '_isPartition' in node && node._isPartition;
  const isGroupNode = isGroup || ('_isGroup' in node && node._isGroup);

  let labelText = style.label;
  let labelExtra = '';

  if ('chunkCount' in node) labelExtra = ` (${node.chunkCount})`;
  if ('zoneCount' in node) labelExtra = ` (${node.zoneCount})`;
  if (isPartition && !isGroupNode) labelText = formatRowRange(node.rowRange);
  if (isGroupNode) labelText = '···';

  const opacity = isGroupNode ? 'opacity-50' : isFlat ? 'opacity-70' : '';
  const fontStyle = isGroupNode ? 'italic' : '';

  return (
    <div
      className={`flex items-center gap-1.5 text-[11px] border-b border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900 whitespace-nowrap hover:bg-gray-50 dark:hover:bg-gray-800 cursor-default ${opacity} ${fontStyle}`}
      style={{ height: ROW_HEIGHT, paddingLeft: 6 + depth * 10, paddingRight: 8 }}
    >
      <span
        className={`text-[8px] w-3 text-gray-400 ${expandable ? 'cursor-pointer' : ''}`}
        style={{ opacity: expandable ? 1 : 0 }}
        onClick={expandable ? onToggle : undefined}
      >
        {isExpanded ? '▼' : '▶'}
      </span>
      <span className="flex-1 overflow-hidden text-ellipsis text-gray-900 dark:text-gray-100">
        {node.name}
      </span>
      <span
        className="text-[9px] px-1.5 py-0.5 rounded"
        style={{ color: style.color, backgroundColor: `${style.color}15` }}
      >
        {labelText}
        {labelExtra}
      </span>
    </div>
  );
}

// ============================================================================
// Swimlane Bar Component
// ============================================================================

interface SwimlaneBarProps {
  node: FlattenedNode['node'];
  totalRows: number;
  isGroup?: boolean;
  onHover: (node: FlatLayout | null, position: { x: number; y: number }) => void;
}

function SwimlaneBar({ node, totalRows, isGroup, onHover }: SwimlaneBarProps) {
  const isFlat = node.type === 'flat';
  const isGroupNode = isGroup || ('_isGroup' in node && node._isGroup);
  const style = LAYOUT_STYLES[node.type as keyof typeof LAYOUT_STYLES] || LAYOUT_STYLES.chunk;

  const left = (node.rowRange[0] / totalRows) * 100;
  const width = ((node.rowRange[1] - node.rowRange[0]) / totalRows) * 100;

  let barClasses = 'absolute top-[3px] bottom-[3px] rounded transition-[filter] duration-100';
  let barStyle: React.CSSProperties = {
    left: `calc(${left}% + 1px)`,
    width: `calc(${width}% - 3px)`,
  };

  if (isFlat) {
    const dtypeCat = getDtypeCategory((node as FlatLayout).meta?.dtype);
    const dtypeColor = DTYPE_COLORS[dtypeCat];
    barStyle.backgroundColor = `${dtypeColor}40`;
    barStyle.border = `1.5px solid ${dtypeColor}`;
    barClasses += ' cursor-pointer';
  } else if (isGroupNode) {
    barStyle.border = `1.5px dotted ${style.color}40`;
    barStyle.opacity = 0.4;
  } else {
    barStyle.border = `1.5px dashed ${style.color}60`;
  }

  const handleMouseEnter = (e: React.MouseEvent) => {
    if (isFlat) {
      (e.currentTarget as HTMLDivElement).style.filter = 'brightness(1.15)';
      onHover(node as FlatLayout, { x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isFlat) {
      onHover(node as FlatLayout, { x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseLeave = (e: React.MouseEvent) => {
    if (isFlat) {
      (e.currentTarget as HTMLDivElement).style.filter = '';
      onHover(null, { x: 0, y: 0 });
    }
  };

  return (
    <div
      className={barClasses}
      style={barStyle}
      onMouseEnter={handleMouseEnter}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    />
  );
}

// ============================================================================
// Split Region Component
// ============================================================================

interface SplitRegionProps {
  split: Split;
  totalRows: number;
  swimlaneWidth: number;
  isSelected: boolean;
  onClick: (e: React.MouseEvent) => void;
}

function SplitRegion({ split, totalRows, swimlaneWidth, isSelected, onClick }: SplitRegionProps) {
  const [isHovered, setIsHovered] = useState(false);

  const left = (split.rowRange[0] / totalRows) * 100;
  const width = ((split.rowRange[1] - split.rowRange[0]) / totalRows) * 100;
  const widthPx = ((split.rowRange[1] - split.rowRange[0]) / totalRows) * swimlaneWidth;
  const showLabel = widthPx >= MIN_LABEL_WIDTH || isSelected || isHovered;

  return (
    <div
      className="absolute top-0 bottom-0 cursor-pointer border-r border-gray-200 dark:border-gray-700"
      style={{
        left: `${left}%`,
        width: `${width}%`,
        backgroundColor: isSelected ? '#88C0D015' : isHovered ? 'rgba(136,192,208,0.05)' : undefined,
      }}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        className="absolute top-1 left-1/2 -translate-x-1/2 text-[9px] px-1.5 py-0.5 rounded whitespace-nowrap transition-opacity duration-150 pointer-events-none"
        style={{
          color: isSelected ? '#88C0D0' : '#9CA3AF',
          backgroundColor: isSelected ? '#88C0D025' : 'rgba(249,250,251,0.9)',
          opacity: showLabel ? 1 : 0,
          fontWeight: isSelected ? 500 : 400,
        }}
      >
        {split.id}
      </div>
    </div>
  );
}

// ============================================================================
// Selection Panel Component
// ============================================================================

interface SelectionPanelProps {
  splits: Split[];
  selectedSplits: Set<string>;
  onRemove: (id: string) => void;
}

function SelectionPanel({ splits, selectedSplits, onRemove }: SelectionPanelProps) {
  if (selectedSplits.size === 0) {
    return (
      <div className="p-5 text-center text-gray-500 dark:text-gray-400 text-xs">
        Click a split to select · Click again to deselect · Ctrl+click to multi-select
      </div>
    );
  }

  const selected = splits.filter(s => selectedSplits.has(s.id));

  return (
    <div>
      <div className="flex justify-between items-center px-3 py-2 bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-800">
        <span className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400">
          Selected splits
        </span>
        <span className="text-[10px] text-cyan-500">{selected.length} selected</span>
      </div>
      <div className="flex flex-wrap gap-1.5 p-3">
        {selected.map(s => (
          <div
            key={s.id}
            className="inline-flex gap-2 items-center bg-white dark:bg-gray-900 px-2.5 py-1.5 rounded-md text-[11px] border border-gray-100 dark:border-gray-800 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800"
            onClick={() => onRemove(s.id)}
          >
            <span className="text-cyan-500 font-medium">{s.id}</span>
            <span className="text-gray-900 dark:text-gray-100">
              {s.rowRange[0].toLocaleString()}–{s.rowRange[1].toLocaleString()}
            </span>
            <span className="text-gray-400">
              {(s.rowRange[1] - s.rowRange[0]).toLocaleString()} rows
            </span>
            <span className="text-gray-400 ml-1">✕</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Dtype Legend Component
// ============================================================================

function DtypeLegend() {
  return (
    <div className="flex gap-3.5 px-3 py-2 border-t border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900 text-[11px] text-gray-500 dark:text-gray-400">
      {DTYPE_CATEGORIES.map(cat => (
        <div key={cat} className="flex items-center gap-1">
          <div
            className="w-2.5 h-2.5 rounded"
            style={{ backgroundColor: DTYPE_COLORS[cat] }}
          />
          {cat}
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function LayoutSwimlane({
  layout,
  totalRows,
  fileName,
  defaultExpanded = [],
  onSplitSelect,
  swimlaneMinWidth = 800,
  height = 360,
}: LayoutSwimlaneProps) {
  // State
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(defaultExpanded));
  const [selectedSplits, setSelectedSplits] = useState<Set<string>>(new Set());
  const [tooltip, setTooltip] = useState<{ node: FlatLayout; position: { x: number; y: number } } | null>(null);
  const [rulerPosition, setRulerPosition] = useState<{ x: number; row: number } | null>(null);

  // Refs
  const swimlaneScrollRef = useRef<HTMLDivElement>(null);
  const swimlanePanelRef = useRef<HTMLDivElement>(null);
  const axisRef = useRef<HTMLDivElement>(null);

  // Computed values
  const splits = useMemo(() => createSplits(layout), [layout]);
  const selectedRange = useMemo(
    () => getSelectedRowRange(splits, selectedSplits),
    [splits, selectedSplits]
  );

  // Flatten tree for rendering
  const flattenedNodes = useMemo(() => {
    const nodeDataMap = new Map<string, FlattenedNode['node']>();

    function flatten(
      node: LayoutNode | ChunkNode | ZoneNode,
      depth: number,
      result: FlattenedNode[]
    ): FlattenedNode[] {
      const isChunkOrZone = 'child' in node && !('type' in node && (node as unknown as LayoutNode).type);
      const nodeType =
        'type' in node
          ? (node as LayoutNode).type
          : 'meta' in node && 'min' in (node as ZoneNode).meta
          ? 'zone'
          : 'chunk';

      const flatNode = { ...node, type: nodeType, _isPartition: isChunkOrZone } as unknown as FlattenedNode['node'];
      result.push({ node: flatNode, depth });
      nodeDataMap.set(node.id, flatNode);

      const isExpanded = expanded.has(node.id);

      if (isExpanded) {
        if ('chunks' in node && node.chunks) {
          const groups = groupChunks(node.chunks, node.id);

          if (groups) {
            const visibleGroups = selectedRange
              ? groups.filter(g => rangesOverlap(g.rowRange, selectedRange))
              : groups;

            visibleGroups.forEach(group => {
              result.push({ node: group as unknown as FlattenedNode['node'], depth: depth + 1, isGroup: true });
              nodeDataMap.set(group.id, group as unknown as FlattenedNode['node']);

              if (expanded.has(group.id)) {
                const visibleInGroup = selectedRange
                  ? group.chunks.filter(p => rangesOverlap(p.rowRange, selectedRange))
                  : group.chunks;
                visibleInGroup.forEach(p => flatten(p, depth + 2, result));

                if (selectedRange && visibleInGroup.length < group.chunks.length) {
                  const hidden = group.chunks.length - visibleInGroup.length;
                  result.push({
                    node: {
                      id: `${group.id}_hidden`,
                      type: 'hidden',
                      name: `${hidden} more in group`,
                      rowRange: group.rowRange,
                    } as unknown as FlattenedNode['node'],
                    depth: depth + 2,
                    isHiddenIndicator: true,
                  });
                }
              }
            });

            if (selectedRange && visibleGroups.length < groups.length) {
              const hiddenGroups = groups.length - visibleGroups.length;
              result.push({
                node: {
                  id: `${node.id}_hidden_groups`,
                  type: 'hidden',
                  name: `${hiddenGroups} more groups`,
                  rowRange: node.rowRange,
                } as unknown as FlattenedNode['node'],
                depth: depth + 1,
                isHiddenIndicator: true,
              });
            }
          } else {
            const visible = selectedRange
              ? node.chunks.filter(p => rangesOverlap(p.rowRange, selectedRange))
              : node.chunks;
            visible.forEach(p => flatten(p, depth + 1, result));

            if (selectedRange && visible.length < node.chunks.length) {
              result.push({
                node: {
                  id: `${node.id}_hidden`,
                  type: 'hidden',
                  name: `${node.chunks.length - visible.length} more partitions`,
                  rowRange: node.rowRange,
                } as unknown as FlattenedNode['node'],
                depth: depth + 1,
                isHiddenIndicator: true,
              });
            }
          }
        }

        if ('zones' in node && node.zones) {
          const visible = selectedRange
            ? node.zones.filter(z => rangesOverlap(z.rowRange, selectedRange))
            : node.zones;
          visible.forEach(z => flatten(z, depth + 1, result));

          if (selectedRange && visible.length < node.zones.length) {
            result.push({
              node: {
                id: `${node.id}_hidden`,
                type: 'hidden',
                name: `${node.zones.length - visible.length} more partitions`,
                rowRange: node.rowRange,
              } as unknown as FlattenedNode['node'],
              depth: depth + 1,
              isHiddenIndicator: true,
            });
          }
        }

        if ('child' in node && node.child) {
          flatten(node.child, depth + 1, result);
        }

        if ('children' in node && node.children) {
          node.children.forEach(c => flatten(c, depth + 1, result));
        }
      }

      return result;
    }

    return flatten(layout, 0, []);
  }, [layout, expanded, selectedRange]);

  // Callbacks
  const toggleExpanded = useCallback((id: string) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const handleSplitClick = useCallback(
    (splitId: string, e: React.MouseEvent) => {
      setSelectedSplits(prev => {
        const next = new Set(prev);

        if (e.ctrlKey || e.metaKey) {
          if (next.has(splitId)) {
            next.delete(splitId);
          } else {
            next.add(splitId);
          }
        } else {
          if (next.size === 1 && next.has(splitId)) {
            next.clear();
          } else {
            next.clear();
            next.add(splitId);
          }
        }

        return next;
      });
    },
    []
  );

  const handleRemoveSplit = useCallback((id: string) => {
    setSelectedSplits(prev => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }, []);

  const handleTooltip = useCallback(
    (node: FlatLayout | null, position: { x: number; y: number }) => {
      if (node) {
        setTooltip({ node, position });
      } else {
        setTooltip(null);
      }
    },
    []
  );

  // Sync horizontal scroll
  useEffect(() => {
    const swimlaneScroll = swimlaneScrollRef.current;
    const axis = axisRef.current;

    if (!swimlaneScroll || !axis) return;

    const handleScroll = () => {
      axis.style.transform = `translateX(-${swimlaneScroll.scrollLeft}px)`;
    };

    swimlaneScroll.addEventListener('scroll', handleScroll);
    return () => swimlaneScroll.removeEventListener('scroll', handleScroll);
  }, []);

  // Ruler mouse tracking
  const handleSwimlaneMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const panel = swimlanePanelRef.current;
      if (!panel) return;

      const rect = panel.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const panelWidth = panel.offsetWidth;

      if (x >= 0 && x <= panelWidth) {
        const rowNum = (x / panelWidth) * totalRows;
        setRulerPosition({ x, row: Math.max(0, Math.min(totalRows, rowNum)) });
      }
    },
    [totalRows]
  );

  const handleSwimlaneMouseLeave = useCallback(() => {
    setRulerPosition(null);
  }, []);

  // Notify parent of selection changes
  useEffect(() => {
    if (onSplitSelect) {
      const selected = splits.filter(s => selectedSplits.has(s.id));
      onSplitSelect(selected);
    }
  }, [selectedSplits, splits, onSplitSelect]);

  // Axis ticks
  const axisTicks = useMemo(() => {
    const ticks = [];
    const step = totalRows / 5;
    for (let i = 0; i <= 5; i++) {
      ticks.push(Math.round(i * step));
    }
    return ticks;
  }, [totalRows]);

  return (
    <div className="font-sans">
      {/* Header */}
      <div className="flex items-center gap-4 mb-4">
        <span className="text-lg font-medium text-gray-900 dark:text-gray-100">Layout swimlane</span>
        {fileName && (
          <span className="text-[13px] text-gray-500 dark:text-gray-400">
            {fileName} · {formatRowCount(totalRows)} rows
          </span>
        )}
      </div>

      {/* Main panel */}
      <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        {/* Tree + Swimlane */}
        <div className="flex overflow-y-auto overflow-x-hidden" style={{ height }}>
          {/* Tree panel */}
          <div className="w-[260px] flex-shrink-0 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700">
            {flattenedNodes.map(({ node, depth, isGroup, isHint, isHiddenIndicator }) => (
              <TreeRow
                key={node.id}
                node={node}
                depth={depth}
                isExpanded={expanded.has(node.id)}
                isGroup={isGroup}
                isHint={isHint}
                isHiddenIndicator={isHiddenIndicator}
                onToggle={() => toggleExpanded(node.id)}
              />
            ))}
          </div>

          {/* Swimlane panel */}
          <div
            ref={swimlaneScrollRef}
            className="flex-1 overflow-x-auto overflow-y-hidden relative"
            onMouseMove={handleSwimlaneMouseMove}
            onMouseLeave={handleSwimlaneMouseLeave}
          >
            <div
              ref={swimlanePanelRef}
              className="relative"
              style={{ minWidth: swimlaneMinWidth }}
            >
              {/* Split regions (background) */}
              <div className="absolute inset-0 z-[1]">
                {splits.map(split => (
                  <SplitRegion
                    key={split.id}
                    split={split}
                    totalRows={totalRows}
                    swimlaneWidth={swimlaneMinWidth}
                    isSelected={selectedSplits.has(split.id)}
                    onClick={e => handleSplitClick(split.id, e)}
                  />
                ))}
              </div>

              {/* Bars (foreground) */}
              <div className="relative z-[2] pointer-events-none">
                {flattenedNodes.map(({ node, isGroup, isHint, isHiddenIndicator }) => (
                  <div
                    key={node.id}
                    className="relative border-b border-gray-100 dark:border-gray-800"
                    style={{ height: ROW_HEIGHT }}
                  >
                    {!isHint && !isHiddenIndicator && (
                      <SwimlaneBar
                        node={node}
                        totalRows={totalRows}
                        isGroup={isGroup}
                        onHover={handleTooltip}
                      />
                    )}
                  </div>
                ))}
              </div>

              {/* Ruler line */}
              {rulerPosition && (
                <div
                  className="absolute top-0 bottom-0 w-px bg-gray-900 dark:bg-gray-100 opacity-40 pointer-events-none z-[100]"
                  style={{ left: rulerPosition.x }}
                />
              )}
            </div>
          </div>
        </div>

        {/* Axis */}
        <div className="flex border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 relative">
          <div className="w-[260px] flex-shrink-0 border-r border-gray-200 dark:border-gray-700" />
          <div className="flex-1 overflow-hidden relative">
            <div
              ref={axisRef}
              className="relative h-[26px]"
              style={{ minWidth: swimlaneMinWidth }}
            >
              {axisTicks.map(tick => (
                <div
                  key={tick}
                  className="absolute text-[9px] text-gray-400 top-1.5"
                  style={{
                    left: `${(tick / totalRows) * 100}%`,
                    transform: 'translateX(-50%)',
                  }}
                >
                  {(tick / 1000).toFixed(0)}k
                </div>
              ))}
            </div>

            {/* Ruler label */}
            {rulerPosition && (
              <div
                className="absolute top-1 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 px-1.5 py-0.5 rounded text-[10px] font-medium pointer-events-none z-[100] whitespace-nowrap"
                style={{
                  left: Math.max(
                    0,
                    Math.min(
                      rulerPosition.x - (swimlaneScrollRef.current?.scrollLeft || 0) - 20,
                      (swimlaneScrollRef.current?.offsetWidth || 0) - 50
                    )
                  ),
                }}
              >
                {formatRowCount(rulerPosition.row)}
              </div>
            )}
          </div>
        </div>

        {/* Dtype legend */}
        <DtypeLegend />
      </div>

      {/* Selection panel */}
      <div className="mt-4 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        <SelectionPanel
          splits={splits}
          selectedSplits={selectedSplits}
          onRemove={handleRemoveSplit}
        />
      </div>

      {/* Tooltip */}
      {tooltip && <Tooltip node={tooltip.node} position={tooltip.position} />}
    </div>
  );
}

export default LayoutSwimlane;
