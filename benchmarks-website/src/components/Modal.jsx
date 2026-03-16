import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Line } from 'react-chartjs-2';
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
import { fetchChartData } from '../api';
import { stringToColor, formatDate } from '../utils';

export default function Modal({ chartData, onClose }) {
  const [loading, setLoading] = useState(false);
  const [viewRange, setViewRange] = useState(null);
  const [currentData, setCurrentData] = useState(null);
  const [totalCommits, setTotalCommits] = useState(chartData?.totalCommits || 0);
  const chartRef = useRef(null);
  const isResettingZoom = useRef(false);

  // Initialize with the data passed from parent
  useEffect(() => {
    if (chartData?.initialData) {
      setCurrentData(chartData.initialData);
      setTotalCommits(chartData.totalCommits || chartData.initialData.commits?.length || 0);
      if (chartData.currentRange) {
        setViewRange({
          startIdx: chartData.currentRange.startIdx,
          endIdx: chartData.currentRange.endIdx,
        });
      }
    }
  }, [chartData]);

  // Close on escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = '';
    };
  }, []);

  const handleBackdropClick = useCallback((e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  }, [onClose]);

  // Fetch data for a new range
  const fetchRange = useCallback(async (range) => {
    if (!chartData?.groupName || !chartData?.chartName) return;

    setLoading(true);
    try {
      let options = {};
      if (range.last) {
        options = { last: range.last };
      } else if (range.startIdx !== undefined && range.endIdx !== undefined) {
        options = { startIdx: range.startIdx, endIdx: range.endIdx };
      }

      const data = await fetchChartData(chartData.groupName, chartData.chartName, options);
      if (data) {
        // Process the data similar to ChartContainer
        const processedData = processChartData(data, chartData.config);
        setCurrentData(processedData);
        if (data.originalLength) {
          setTotalCommits(data.originalLength);
        }
      }
    } catch (err) {
      console.error('Error fetching range:', err);
    } finally {
      setLoading(false);
    }
  }, [chartData]);

  // Process chart data (similar to ChartContainer)
  const processChartData = useCallback((data, config) => {
    if (!data?.series || !data?.commits) return null;

    const { series, commits } = data;
    const datasets = [];
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

      const dataPoints = points.map(p => p);
      const hidden = config?.hiddenDatasets?.has(seriesName) ||
                     config?.hiddenDatasets?.has(displaySeriesName);

      datasets.push({
        label: displaySeriesName,
        data: dataPoints,
        borderColor: stringToColor(displaySeriesName),
        backgroundColor: stringToColor(displaySeriesName) + '20',
        pointRadius: 2,
        pointHoverRadius: 5,
        pointStyle: 'cross',
        borderWidth: 1.5,
        tension: 0,
        spanGaps: true,
        hidden,
      });
    });

    return { labels, datasets, commits };
  }, []);

  // Get current range info
  const getCurrentRange = useCallback(() => {
    if (!viewRange) {
      return { startIdx: 0, endIdx: totalCommits - 1, total: totalCommits, rangeSize: totalCommits };
    }
    const startIdx = viewRange.startIdx ?? 0;
    const endIdx = viewRange.endIdx ?? (totalCommits - 1);
    return {
      startIdx,
      endIdx,
      total: totalCommits,
      rangeSize: endIdx - startIdx + 1,
    };
  }, [viewRange, totalCommits]);

  const range = getCurrentRange();
  const isAtStart = range.startIdx === 0;
  const isAtEnd = range.endIdx >= range.total - 1;
  const currentRangeSize = range.rangeSize;
  const isFullRange = isAtStart && isAtEnd;

  // Navigation handlers
  const handleGoToStart = useCallback(() => {
    if (isAtStart || !range.total) return;
    const newRange = {
      startIdx: 0,
      endIdx: Math.min(currentRangeSize - 1, range.total - 1),
    };
    setViewRange(newRange);
    fetchRange(newRange);
  }, [isAtStart, currentRangeSize, range.total, fetchRange]);

  const handleGoToEnd = useCallback(() => {
    if (isAtEnd || !range.total) return;
    const newRange = { last: currentRangeSize };
    setViewRange({
      startIdx: range.total - currentRangeSize,
      endIdx: range.total - 1,
    });
    fetchRange(newRange);
  }, [isAtEnd, currentRangeSize, range.total, fetchRange]);

  const handleMoveBackward = useCallback(() => {
    if (isAtStart || !range.total) return;
    const moveAmount = Math.max(1, Math.floor(currentRangeSize / 2));
    const newStartIdx = Math.max(0, range.startIdx - moveAmount);
    const newEndIdx = newStartIdx + currentRangeSize - 1;
    const newRange = {
      startIdx: newStartIdx,
      endIdx: Math.min(newEndIdx, range.total - 1),
    };
    setViewRange(newRange);
    fetchRange(newRange);
  }, [isAtStart, range, currentRangeSize, fetchRange]);

  const handleMoveForward = useCallback(() => {
    if (isAtEnd || !range.total) return;
    const moveAmount = Math.max(1, Math.floor(currentRangeSize / 2));
    const newEndIdx = Math.min(range.total - 1, range.endIdx + moveAmount);
    const newStartIdx = Math.max(0, newEndIdx - currentRangeSize + 1);
    const newRange = { startIdx: newStartIdx, endIdx: newEndIdx };
    setViewRange(newRange);
    fetchRange(newRange);
  }, [isAtEnd, range, currentRangeSize, fetchRange]);

  const handleZoomIn = useCallback(() => {
    if (!range.total || currentRangeSize <= 10) return;
    const center = Math.floor((range.startIdx + range.endIdx) / 2);
    const newRangeSize = Math.max(10, Math.floor(currentRangeSize / 2));
    const halfRange = Math.floor(newRangeSize / 2);
    let newStartIdx = center - halfRange;
    let newEndIdx = newStartIdx + newRangeSize - 1;

    if (newStartIdx < 0) {
      newStartIdx = 0;
      newEndIdx = newRangeSize - 1;
    }
    if (newEndIdx >= range.total) {
      newEndIdx = range.total - 1;
      newStartIdx = Math.max(0, newEndIdx - newRangeSize + 1);
    }

    const newRange = { startIdx: newStartIdx, endIdx: newEndIdx };
    setViewRange(newRange);
    fetchRange(newRange);
  }, [range, currentRangeSize, fetchRange]);

  const handleZoomOut = useCallback(() => {
    if (!range.total) return;
    const center = Math.floor((range.startIdx + range.endIdx) / 2);
    const newRangeSize = Math.min(range.total, currentRangeSize * 2);
    const halfRange = Math.floor(newRangeSize / 2);
    let newStartIdx = center - halfRange;
    let newEndIdx = newStartIdx + newRangeSize - 1;

    if (newStartIdx < 0) {
      newStartIdx = 0;
      newEndIdx = Math.min(newRangeSize - 1, range.total - 1);
    }
    if (newEndIdx >= range.total) {
      newEndIdx = range.total - 1;
      newStartIdx = Math.max(0, newEndIdx - newRangeSize + 1);
    }

    const newRange = { startIdx: newStartIdx, endIdx: newEndIdx };
    setViewRange(newRange);
    fetchRange(newRange);
  }, [range, currentRangeSize, fetchRange]);

  const handleShowFullRange = useCallback(() => {
    if (!range.total) return;
    const newRange = { startIdx: 0, endIdx: range.total - 1 };
    setViewRange(newRange);
    fetchRange(newRange);
  }, [range.total, fetchRange]);

  // Handle drag selection zoom
  const handleDragZoom = useCallback((startDataIdx, endDataIdx) => {
    if (!currentData?.commits) return;

    const numCommits = currentData.commits.length;
    if (numCommits < 2) return;

    const rangeStart = range.startIdx;
    const rangeEnd = range.endIdx;
    const total = range.total;

    const minIdx = Math.min(startDataIdx, endDataIdx);
    const maxIdx = Math.max(startDataIdx, endDataIdx);
    const globalStartIdx = rangeStart + Math.round(minIdx / (numCommits - 1) * (rangeEnd - rangeStart));
    const globalEndIdx = rangeStart + Math.round(maxIdx / (numCommits - 1) * (rangeEnd - rangeStart));

    if (globalEndIdx - globalStartIdx < 5) return;

    const newRange = {
      startIdx: Math.max(0, globalStartIdx),
      endIdx: Math.min(total - 1, globalEndIdx),
    };
    setViewRange(newRange);
    fetchRange(newRange);
  }, [currentData, range, fetchRange]);

  // Chart options
  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        align: 'start',
        labels: {
          boxWidth: 12,
          padding: 8,
          font: { size: 11, family: 'Geist, sans-serif' },
          usePointStyle: true,
          pointStyle: 'rectRounded',
        },
      },
      tooltip: {
        backgroundColor: 'rgba(16, 16, 16, 0.9)',
        titleFont: { family: 'Geist, sans-serif', size: 12 },
        bodyFont: { family: 'Geist Mono, monospace', size: 11 },
        padding: 12,
        cornerRadius: 4,
        caretSize: 0,
        position: 'topCorner',
        itemSort: (a, b) => b.parsed.y - a.parsed.y,
        callbacks: {
          title: (items) => {
            if (!items.length || !currentData?.commits) return '';
            const commit = currentData.commits[items[0].dataIndex];
            if (!commit) return items[0].label;
            const author = commit.author || 'Unknown';
            return `${formatDate(commit.timestamp)} — ${author}\n(${commit.id?.slice(0, 7) || ''}) ${commit.message || ''}`;
          },
          label: (item) => {
            const value = item.parsed.y;
            if (value == null) return null;
            const formattedValue = value < 1 ? value.toFixed(4) : value.toFixed(2);
            return `${item.dataset.label}: ${formattedValue} ${chartData?.unit || ''}`;
          },
        },
      },
      zoom: {
        zoom: {
          drag: {
            enabled: true,
            backgroundColor: 'rgba(99, 102, 241, 0.2)',
            borderColor: 'rgba(99, 102, 241, 0.8)',
            borderWidth: 1,
          },
          mode: 'x',
          onZoomComplete: ({ chart }) => {
            if (isResettingZoom.current) {
              isResettingZoom.current = false;
              return;
            }
            const { min, max } = chart.scales.x;
            const startIdx = Math.floor(min);
            const endIdx = Math.ceil(max);
            if (startIdx >= 0 && endIdx > startIdx) {
              handleDragZoom(startIdx, endIdx);
            }
            isResettingZoom.current = true;
            chart.resetZoom();
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: { display: true, color: 'rgba(0, 0, 0, 0.12)' },
        ticks: {
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10, family: 'Geist, sans-serif' },
          maxTicksLimit: 15,
          callback: function(value, index, ticks) {
            // Always show first and last tick
            if (index === 0 || index === ticks.length - 1) {
              return this.getLabelForValue(value);
            }
            // Show intermediate ticks based on maxTicksLimit
            const step = Math.ceil(ticks.length / 15);
            if (index % step === 0) {
              return this.getLabelForValue(value);
            }
            return null;
          },
        },
      },
      y: {
        display: true,
        beginAtZero: true,
        grid: { color: 'rgba(0, 0, 0, 0.12)' },
        ticks: { font: { size: 11, family: 'Geist Mono, monospace' } },
        title: {
          display: !!chartData?.unit,
          text: chartData?.unit || '',
          font: { size: 11, family: 'Geist, sans-serif' },
        },
      },
    },
  }), [currentData, chartData, handleDragZoom]);

  if (!chartData) return null;

  return (
    <div className="chart-modal active" onClick={handleBackdropClick}>
      <div className="modal-content">
        <div className="modal-header">
          <h2>{chartData.title}</h2>
          <div className="modal-controls">
            <div className="chart-zoom-controls">
              <button
                className="chart-zoom-btn"
                onClick={handleGoToStart}
                disabled={loading || isAtStart}
                data-tooltip="Go to beginning"
              >
                <SkipBack size={14} />
              </button>
              <button
                className="chart-zoom-btn"
                onClick={handleMoveBackward}
                disabled={loading || isAtStart}
                data-tooltip="Move backwards"
              >
                <ChevronLeft size={14} />
              </button>
              <button
                className="chart-zoom-btn"
                onClick={handleZoomOut}
                disabled={loading || isFullRange}
                data-tooltip="Zoom out"
              >
                <ZoomOut size={14} />
              </button>
              <button
                className="chart-zoom-btn"
                onClick={handleZoomIn}
                disabled={loading || currentRangeSize <= 10}
                data-tooltip="Zoom in"
              >
                <ZoomIn size={14} />
              </button>
              <button
                className="chart-zoom-btn"
                onClick={handleMoveForward}
                disabled={loading || isAtEnd}
                data-tooltip="Move forwards"
              >
                <ChevronRight size={14} />
              </button>
              <button
                className="chart-zoom-btn"
                onClick={handleGoToEnd}
                disabled={loading || isAtEnd}
                data-tooltip="Go to end"
              >
                <SkipForward size={14} />
              </button>
              <button
                className="chart-zoom-btn"
                onClick={handleShowFullRange}
                disabled={loading || isFullRange}
                data-tooltip="Show full range"
              >
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
            <Line
              ref={chartRef}
              data={{
                labels: currentData.labels,
                datasets: currentData.datasets,
              }}
              options={options}
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
