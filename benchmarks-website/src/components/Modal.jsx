import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
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
  SkipBack,
  ChevronLeft,
  ChevronRight,
  SkipForward,
  ZoomIn,
  ZoomOut,
  MoveHorizontal,
  X,
} from 'lucide-react';
import { fetchChartData } from '../lib/api';
import { stringToColor, formatDate } from '../lib/utils';

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
  CanvasRenderer,
]);

export default function Modal({ chartData: chartProps, onClose }) {
  const [loading, setLoading] = useState(false);
  const [viewRange, setViewRange] = useState(null);
  const [currentData, setCurrentData] = useState(null);
  const [totalCommits, setTotalCommits] = useState(chartProps?.totalCommits || 0);
  const chartRef = useRef(null);

  useEffect(() => {
    if (chartProps?.initialData) {
      setCurrentData(chartProps.initialData);
      setTotalCommits(chartProps.totalCommits || chartProps.initialData.commits?.length || 0);
      if (chartProps.currentRange) {
        setViewRange({
          startIdx: chartProps.currentRange.startIdx,
          endIdx: chartProps.currentRange.endIdx,
        });
      }
    }
  }, [chartProps]);

  useEffect(() => {
    const handleKeyDown = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = ''; };
  }, []);

  const handleBackdropClick = useCallback((e) => {
    if (e.target === e.currentTarget) onClose();
  }, [onClose]);

  const fetchRange = useCallback(async (range) => {
    if (!chartProps?.groupName || !chartProps?.chartName) return;
    setLoading(true);
    try {
      let options = {};
      if (range.last) {
        options = { last: range.last };
      } else if (range.startIdx !== undefined && range.endIdx !== undefined) {
        options = { startIdx: range.startIdx, endIdx: range.endIdx };
      }
      const data = await fetchChartData(chartProps.groupName, chartProps.chartName, options);
      if (data) {
        const processed = processChartData(data, chartProps.config);
        setCurrentData(processed);
        if (data.originalLength) setTotalCommits(data.originalLength);
      }
    } catch (err) {
      console.error('Error fetching range:', err);
    } finally {
      setLoading(false);
    }
  }, [chartProps]);

  const processChartData = useCallback((data, config) => {
    if (!data?.series || !data?.commits) return null;
    const { series, commits } = data;
    const seriesList = [];
    const labels = commits.map(c => formatDate(c.timestamp));

    Object.entries(series).forEach(([seriesName, points]) => {
      if (config?.removedDatasets?.has(seriesName)) return;

      let displaySeriesName = seriesName;
      if (config?.renamedDatasets) {
        const caseInsensitive = {};
        Object.entries(config.renamedDatasets).forEach(([k, v]) => {
          caseInsensitive[k.toLowerCase()] = v;
        });
        displaySeriesName = caseInsensitive[seriesName.toLowerCase()] || seriesName;
      }

      const hidden = config?.hiddenDatasets?.has(seriesName) ||
                     config?.hiddenDatasets?.has(displaySeriesName);

      seriesList.push({
        name: displaySeriesName,
        data: points,
        color: stringToColor(displaySeriesName),
        hidden,
      });
    });

    return { labels, seriesList, commits };
  }, []);

  const getCurrentRange = useCallback(() => {
    if (!viewRange) {
      return { startIdx: 0, endIdx: totalCommits - 1, total: totalCommits, rangeSize: totalCommits };
    }
    const startIdx = viewRange.startIdx ?? 0;
    const endIdx = viewRange.endIdx ?? (totalCommits - 1);
    return { startIdx, endIdx, total: totalCommits, rangeSize: endIdx - startIdx + 1 };
  }, [viewRange, totalCommits]);

  const range = getCurrentRange();
  const isAtStart = range.startIdx === 0;
  const isAtEnd = range.endIdx >= range.total - 1;
  const currentRangeSize = range.rangeSize;
  const isFullRange = isAtStart && isAtEnd;

  const navigate = useCallback((newRange) => {
    setViewRange(newRange);
    fetchRange(newRange);
  }, [fetchRange]);

  const handleGoToStart = useCallback(() => {
    if (isAtStart || !range.total) return;
    navigate({ startIdx: 0, endIdx: Math.min(currentRangeSize - 1, range.total - 1) });
  }, [isAtStart, currentRangeSize, range.total, navigate]);

  const handleGoToEnd = useCallback(() => {
    if (isAtEnd || !range.total) return;
    setViewRange({ startIdx: range.total - currentRangeSize, endIdx: range.total - 1 });
    fetchRange({ last: currentRangeSize });
  }, [isAtEnd, currentRangeSize, range.total, fetchRange]);

  const handleMoveBackward = useCallback(() => {
    if (isAtStart || !range.total) return;
    const moveAmount = Math.max(1, Math.floor(currentRangeSize / 2));
    const newStartIdx = Math.max(0, range.startIdx - moveAmount);
    navigate({ startIdx: newStartIdx, endIdx: Math.min(newStartIdx + currentRangeSize - 1, range.total - 1) });
  }, [isAtStart, range, currentRangeSize, navigate]);

  const handleMoveForward = useCallback(() => {
    if (isAtEnd || !range.total) return;
    const moveAmount = Math.max(1, Math.floor(currentRangeSize / 2));
    const newEndIdx = Math.min(range.total - 1, range.endIdx + moveAmount);
    const newStartIdx = Math.max(0, newEndIdx - currentRangeSize + 1);
    navigate({ startIdx: newStartIdx, endIdx: newEndIdx });
  }, [isAtEnd, range, currentRangeSize, navigate]);

  const handleZoomIn = useCallback(() => {
    if (!range.total || currentRangeSize <= 10) return;
    const center = Math.floor((range.startIdx + range.endIdx) / 2);
    const newRangeSize = Math.max(10, Math.floor(currentRangeSize / 2));
    const halfRange = Math.floor(newRangeSize / 2);
    let newStartIdx = center - halfRange;
    let newEndIdx = newStartIdx + newRangeSize - 1;
    if (newStartIdx < 0) { newStartIdx = 0; newEndIdx = newRangeSize - 1; }
    if (newEndIdx >= range.total) { newEndIdx = range.total - 1; newStartIdx = Math.max(0, newEndIdx - newRangeSize + 1); }
    navigate({ startIdx: newStartIdx, endIdx: newEndIdx });
  }, [range, currentRangeSize, navigate]);

  const handleZoomOut = useCallback(() => {
    if (!range.total) return;
    const center = Math.floor((range.startIdx + range.endIdx) / 2);
    const newRangeSize = Math.min(range.total, currentRangeSize * 2);
    const halfRange = Math.floor(newRangeSize / 2);
    let newStartIdx = center - halfRange;
    let newEndIdx = newStartIdx + newRangeSize - 1;
    if (newStartIdx < 0) { newStartIdx = 0; newEndIdx = Math.min(newRangeSize - 1, range.total - 1); }
    if (newEndIdx >= range.total) { newEndIdx = range.total - 1; newStartIdx = Math.max(0, newEndIdx - newRangeSize + 1); }
    navigate({ startIdx: newStartIdx, endIdx: newEndIdx });
  }, [range, currentRangeSize, navigate]);

  const handleShowFullRange = useCallback(() => {
    if (!range.total) return;
    navigate({ startIdx: 0, endIdx: range.total - 1 });
  }, [range.total, navigate]);

  const echartsOption = useMemo(() => {
    if (!currentData) return {};

    const { labels, seriesList } = currentData;

    return {
      animation: false,
      grid: {
        left: 70,
        right: 30,
        top: 50,
        bottom: 80,
        containLabel: false,
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(16, 16, 16, 0.92)',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        textStyle: {
          fontFamily: 'Geist Mono, monospace',
          fontSize: 11,
          color: '#fff',
        },
        axisPointer: {
          type: 'cross',
          lineStyle: { color: 'rgba(89, 113, 253, 0.4)' },
          crossStyle: { color: 'rgba(89, 113, 253, 0.4)' },
        },
        formatter: (params) => {
          if (!params.length || !currentData?.commits) return '';
          const idx = params[0].dataIndex;
          const commit = currentData.commits[idx];
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
              `<span style="font-weight:600">${val} ${chartProps?.unit || ''}</span></div>`;
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
          fontSize: 11,
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
          fontSize: 10,
          rotate: 45,
          interval: Math.max(0, Math.floor(labels.length / 15) - 1),
        },
        axisLine: { lineStyle: { color: 'rgba(0,0,0,0.12)' } },
        splitLine: { show: true, lineStyle: { color: 'rgba(0,0,0,0.06)' } },
      },
      yAxis: {
        type: 'value',
        min: 0,
        name: chartProps?.unit || '',
        nameTextStyle: {
          fontFamily: 'Geist, sans-serif',
          fontSize: 11,
          padding: [0, 0, 0, 50],
        },
        axisLabel: {
          fontFamily: 'Geist Mono, monospace',
          fontSize: 11,
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
        {
          type: 'slider',
          xAxisIndex: 0,
          bottom: 10,
          height: 20,
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
  }, [currentData, chartProps]);

  if (!chartProps) return null;

  return (
    <div className="chart-modal active" onClick={handleBackdropClick}>
      <div className="modal-content">
        <div className="modal-header">
          <h2>{chartProps.title}</h2>
          <div className="modal-controls">
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
            <button className="modal-close-btn" onClick={onClose} data-tooltip="Close">
              <X size={18} />
            </button>
          </div>
        </div>
        <div className={`modal-chart-container ${loading ? 'loading' : ''}`}>
          {currentData && (
            <ReactEChartsCore
              ref={chartRef}
              echarts={echarts}
              option={echartsOption}
              style={{ height: '100%', width: '100%' }}
              notMerge={true}
              lazyUpdate={true}
            />
          )}
          {loading && (
            <div className="chart-loading-overlay">
              <div className="chart-loading-spinner" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
