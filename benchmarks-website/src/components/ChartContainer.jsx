import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import {
  ChevronLeft,
  ChevronRight,
  Expand,
  MoveHorizontal,
  SkipBack,
  SkipForward,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { fetchChartData } from '../lib/api';
import { formatDate, stringToColor } from '../lib/utils';
import { seriesMatchesFilters } from '../lib/config';

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
  CanvasRenderer,
]);

const DEFAULT_RANGE_SIZE = 100;

export default function ChartContainer({
  groupName,
  chartName,
  displayName,
  unit,
  config,
  filters,
  onFullscreen,
}) {
  const [totalCommits, setTotalCommits] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewRange, setViewRange] = useState({ last: DEFAULT_RANGE_SIZE });
  const chartRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);

      try {
        let options = {};
        if (viewRange.last) {
          options = { last: viewRange.last };
        } else if (viewRange.startIdx !== undefined && viewRange.endIdx !== undefined) {
          options = { startIdx: viewRange.startIdx, endIdx: viewRange.endIdx };
        }

        const data = await fetchChartData(groupName, chartName, options);
        if (!cancelled && data) {
          setChartData(data);
          if (data.originalLength) {
            setTotalCommits(data.originalLength);
          } else if (data.commits) {
            setTotalCommits(data.commits.length);
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadData();
    return () => { cancelled = true; };
  }, [groupName, chartName, viewRange]);

  const displayRangeInfo = useMemo(() => {
    if (!chartData) return { startIdx: 0, endIdx: 0, total: 0, rangeSize: 0 };
    const total = chartData.originalLength || chartData.commits?.length || 0;
    const rangeSize = chartData.commits?.length || 0;
    const req = chartData.requestedRange || {};
    const startIdx = req.startIndex ?? (total - rangeSize);
    const endIdx = req.endIndex ?? (total - 1);
    return { startIdx, endIdx, total, rangeSize };
  }, [chartData]);

  const getCurrentRange = useCallback(() => {
    const total = totalCommits || 0;
    if (viewRange.last) {
      const rangeSize = Math.min(viewRange.last, total);
      return {
        startIdx: Math.max(0, total - rangeSize),
        endIdx: total - 1,
        total,
        rangeSize,
      };
    }
    const startIdx = viewRange.startIdx ?? 0;
    const endIdx = viewRange.endIdx ?? (total - 1);
    return { startIdx, endIdx, total, rangeSize: endIdx - startIdx + 1 };
  }, [viewRange, totalCommits]);

  const isAtStart = displayRangeInfo.startIdx === 0;
  const isAtEnd = displayRangeInfo.endIdx >= displayRangeInfo.total - 1;
  const currentRangeSize = displayRangeInfo.rangeSize;

  const handleGoToStart = useCallback(() => {
    const range = getCurrentRange();
    if (range.startIdx === 0 || !range.total) return;
    setViewRange({ startIdx: 0, endIdx: Math.min(range.rangeSize - 1, range.total - 1) });
  }, [getCurrentRange]);

  const handleGoToEnd = useCallback(() => {
    const range = getCurrentRange();
    if (range.endIdx >= range.total - 1 || !range.total) return;
    setViewRange({ last: range.rangeSize });
  }, [getCurrentRange]);

  const handleMoveBackward = useCallback(() => {
    const range = getCurrentRange();
    if (range.startIdx === 0 || !range.total) return;
    const moveAmount = Math.max(1, Math.floor(range.rangeSize / 2));
    const newStartIdx = Math.max(0, range.startIdx - moveAmount);
    const newEndIdx = newStartIdx + range.rangeSize - 1;
    setViewRange({ startIdx: newStartIdx, endIdx: Math.min(newEndIdx, range.total - 1) });
  }, [getCurrentRange]);

  const handleMoveForward = useCallback(() => {
    const range = getCurrentRange();
    if (range.endIdx >= range.total - 1 || !range.total) return;
    const moveAmount = Math.max(1, Math.floor(range.rangeSize / 2));
    const newEndIdx = Math.min(range.total - 1, range.endIdx + moveAmount);
    const newStartIdx = Math.max(0, newEndIdx - range.rangeSize + 1);
    setViewRange({ startIdx: newStartIdx, endIdx: newEndIdx });
  }, [getCurrentRange]);

  const handleZoomIn = useCallback(() => {
    const range = getCurrentRange();
    if (!range.total || range.rangeSize <= 10) return;
    const center = Math.floor((range.startIdx + range.endIdx) / 2);
    const newRangeSize = Math.max(10, Math.floor(range.rangeSize / 2));
    const halfRange = Math.floor(newRangeSize / 2);
    let newStartIdx = center - halfRange;
    let newEndIdx = newStartIdx + newRangeSize - 1;
    if (newStartIdx < 0) { newStartIdx = 0; newEndIdx = newRangeSize - 1; }
    if (newEndIdx >= range.total) { newEndIdx = range.total - 1; newStartIdx = Math.max(0, newEndIdx - newRangeSize + 1); }
    setViewRange({ startIdx: newStartIdx, endIdx: newEndIdx });
  }, [getCurrentRange]);

  const handleZoomOut = useCallback(() => {
    const range = getCurrentRange();
    if (!range.total) return;
    const center = Math.floor((range.startIdx + range.endIdx) / 2);
    const newRangeSize = Math.min(range.total, range.rangeSize * 2);
    const halfRange = Math.floor(newRangeSize / 2);
    let newStartIdx = center - halfRange;
    let newEndIdx = newStartIdx + newRangeSize - 1;
    if (newStartIdx < 0) { newStartIdx = 0; newEndIdx = Math.min(newRangeSize - 1, range.total - 1); }
    if (newEndIdx >= range.total) { newEndIdx = range.total - 1; newStartIdx = Math.max(0, newEndIdx - newRangeSize + 1); }
    setViewRange({ startIdx: newStartIdx, endIdx: newEndIdx });
  }, [getCurrentRange]);

  const handleShowFullRange = useCallback(() => {
    const range = getCurrentRange();
    if (!range.total) return;
    setViewRange({ startIdx: 0, endIdx: range.total - 1 });
  }, [getCurrentRange]);

  const isFullRange = isAtStart && isAtEnd;

  // Process series data with filters and renaming
  const processedData = useMemo(() => {
    if (!chartData?.series || !chartData?.commits) return null;

    const { series, commits } = chartData;
    const seriesList = [];
    const labels = commits.map(c => formatDate(c.timestamp));

    Object.entries(series).forEach(([seriesName, points]) => {
      if (config.removedDatasets?.has(seriesName)) return;

      // Apply multi-dimensional filters (engine, format, arch, etc.)
      if (filters && !seriesMatchesFilters(seriesName, filters)) return;

      let displaySeriesName = seriesName;
      if (config.renamedDatasets) {
        const caseInsensitive = {};
        Object.entries(config.renamedDatasets).forEach(([k, v]) => {
          caseInsensitive[k.toLowerCase()] = v;
        });
        displaySeriesName = caseInsensitive[seriesName.toLowerCase()] || seriesName;
      }

      const hidden = config.hiddenDatasets?.has(seriesName) ||
                     config.hiddenDatasets?.has(displaySeriesName);

      seriesList.push({
        name: displaySeriesName,
        data: points,
        color: stringToColor(displaySeriesName),
        hidden,
      });
    });

    return { labels, seriesList, commits };
  }, [chartData, config, filters]);

  // Build ECharts option
  const echartsOption = useMemo(() => {
    if (!processedData) return {};

    const { labels, seriesList } = processedData;

    return {
      animation: false,
      grid: {
        left: 60,
        right: 20,
        top: 40,
        bottom: 60,
        containLabel: false,
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(16, 16, 16, 0.92)',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        textStyle: {
          fontFamily: 'Geist Mono, monospace',
          fontSize: 12,
          color: '#fff',
        },
        axisPointer: {
          type: 'cross',
          lineStyle: { color: 'rgba(89, 113, 253, 0.4)' },
          crossStyle: { color: 'rgba(89, 113, 253, 0.4)' },
        },
        formatter: (params) => {
          if (!params.length || !processedData?.commits) return '';
          const idx = params[0].dataIndex;
          const commit = processedData.commits[idx];
          const dateStr = commit ? formatDate(commit.timestamp) : params[0].axisValue;
          const author = commit?.author || 'Unknown';
          const sha = commit?.id?.slice(0, 7) || '';
          const msg = commit?.message || '';

          let header = `<div style="margin-bottom:6px;font-family:Geist,sans-serif;font-size:12px;color:rgba(255,255,255,0.7)">`;
          header += `${dateStr} — ${author}<br/><span style="color:rgba(255,255,255,0.5)">(${sha}) ${msg}</span></div>`;

          const sorted = [...params].filter(p => p.value != null).sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
          const top = sorted.slice(0, 10);
          const lines = top.map(p => {
            const val = p.value < 1 ? p.value.toFixed(4) : p.value.toFixed(2);
            return `<div style="display:flex;align-items:center;gap:6px;margin:2px 0">` +
              `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color}"></span>` +
              `<span style="flex:1">${p.seriesName}</span>` +
              `<span style="font-weight:600">${val} ${unit || ''}</span></div>`;
          }).join('');

          return header + lines;
        },
        confine: true,
      },
      legend: {
        type: 'scroll',
        top: 0,
        left: 0,
        itemWidth: 14,
        itemHeight: 10,
        textStyle: {
          fontFamily: 'Geist, sans-serif',
          fontSize: 12,
          color: '#333',
        },
        selected: Object.fromEntries(
          seriesList.map(s => [s.name, !s.hidden])
        ),
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          fontFamily: 'Geist, sans-serif',
          fontSize: 11,
          rotate: 45,
          interval: Math.max(0, Math.floor(labels.length / 10) - 1),
        },
        axisLine: { lineStyle: { color: 'rgba(0,0,0,0.12)' } },
        splitLine: { show: true, lineStyle: { color: 'rgba(0,0,0,0.06)' } },
      },
      yAxis: {
        type: 'value',
        min: 0,
        name: unit || '',
        nameTextStyle: {
          fontFamily: 'Geist, sans-serif',
          fontSize: 12,
          padding: [0, 0, 0, 40],
        },
        axisLabel: {
          fontFamily: 'Geist Mono, monospace',
          fontSize: 12,
        },
        axisLine: { lineStyle: { color: 'rgba(0,0,0,0.12)' } },
        splitLine: { lineStyle: { color: 'rgba(0,0,0,0.08)' } },
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'none',
        },
      ],
      series: seriesList.map(s => ({
        name: s.name,
        type: 'line',
        data: s.data,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { width: 1.5, color: s.color },
        itemStyle: { color: s.color },
        connectNulls: true,
        emphasis: {
          focus: 'series',
          lineStyle: { width: 2.5 },
        },
      })),
    };
  }, [processedData, unit]);

  // Handle click to open commit on GitHub
  const onChartClick = useCallback((params) => {
    if (!processedData?.commits) return;
    const commit = processedData.commits[params.dataIndex];
    if (commit?.id) {
      window.open(`https://github.com/vortex-data/vortex/commit/${commit.id}`, '_blank');
    }
  }, [processedData]);

  const handleFullscreen = useCallback(() => {
    if (processedData) {
      onFullscreen({
        title: displayName,
        groupName,
        chartName,
        unit,
        config,
        initialData: processedData,
        totalCommits,
        currentRange: getCurrentRange(),
      });
    }
  }, [processedData, displayName, groupName, chartName, unit, config, totalCommits, getCurrentRange, onFullscreen]);

  const showPlaceholder = !processedData && (loading || error);
  const showOverlay = loading && processedData;

  if (showPlaceholder) {
    return (
      <div className="chart-container">
        <div className="chart-header">
          <span className="chart-title">{displayName}</span>
        </div>
        <div className="chart-canvas-placeholder">
          {error ? (
            <p style={{ color: 'var(--text-secondary)' }}>Error loading chart</p>
          ) : (
            <div className="chart-loading-spinner" />
          )}
        </div>
      </div>
    );
  }

  if (!loading && error) {
    return (
      <div className="chart-container">
        <div className="chart-header">
          <span className="chart-title">{displayName}</span>
        </div>
        <div className="chart-canvas-placeholder">
          <p style={{ color: 'var(--text-secondary)' }}>Error loading chart</p>
        </div>
      </div>
    );
  }

  if (!processedData || processedData.seriesList.length === 0) {
    return (
      <div className="chart-container">
        <div className="chart-header">
          <span className="chart-title">{displayName}</span>
        </div>
        <div className="chart-canvas-placeholder">
          <p style={{ color: 'var(--text-secondary)' }}>No data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chart-container">
      <div className="chart-header">
        <span className="chart-title">
          {displayName}
          {chartData?.downsampleLevel && chartData.downsampleLevel !== '1x' && (
            <span className="downsample-indicator" title="Data is downsampled for performance">
              {chartData.downsampleLevel} downsampled
            </span>
          )}
        </span>
        <div className="chart-actions">
          <div className="chart-zoom-controls">
            <button className="chart-zoom-btn" onClick={handleGoToStart} disabled={loading || isAtStart} data-tooltip="Go to beginning">
              <SkipBack size={14} />
            </button>
            <button className="chart-zoom-btn" onClick={handleMoveBackward} disabled={loading || isAtStart} data-tooltip="Move backwards">
              <ChevronLeft size={14} />
            </button>
            <button className="chart-zoom-btn" onClick={handleZoomOut} disabled={loading || isFullRange} data-tooltip="Zoom out">
              <ZoomOut size={14} />
            </button>
            <button className="chart-zoom-btn" onClick={handleZoomIn} disabled={loading || currentRangeSize <= 10} data-tooltip="Zoom in">
              <ZoomIn size={14} />
            </button>
            <button className="chart-zoom-btn" onClick={handleMoveForward} disabled={loading || isAtEnd} data-tooltip="Move forwards">
              <ChevronRight size={14} />
            </button>
            <button className="chart-zoom-btn" onClick={handleGoToEnd} disabled={loading || isAtEnd} data-tooltip="Go to end">
              <SkipForward size={14} />
            </button>
            <button className="chart-zoom-btn" onClick={handleShowFullRange} disabled={loading || isFullRange} data-tooltip="Show full range">
              <MoveHorizontal size={14} />
            </button>
          </div>
          <button className="chart-zoom-btn" onClick={handleFullscreen} disabled={loading} data-tooltip="Fullscreen">
            <Expand size={14} />
          </button>
        </div>
      </div>
      <div className={`chart-canvas-wrapper ${showOverlay ? 'loading' : ''}`}>
        <ReactEChartsCore
          ref={chartRef}
          echarts={echarts}
          option={echartsOption}
          style={{ height: '100%', minHeight: '320px' }}
          notMerge={true}
          lazyUpdate={true}
          onEvents={{
            click: onChartClick,
          }}
        />
        {showOverlay && (
          <div className="chart-loading-overlay">
            <div className="chart-loading-spinner" />
          </div>
        )}
      </div>
    </div>
  );
}
