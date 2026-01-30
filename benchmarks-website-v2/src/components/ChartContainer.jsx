import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import { Line } from 'react-chartjs-2';
import {
  SkipBack,
  ChevronLeft,
  ChevronRight,
  SkipForward,
  ZoomIn,
  ZoomOut,
  MoveHorizontal,
  Expand,
} from 'lucide-react';
import { fetchChartData } from '../api';
import { stringToColor, formatDate } from '../utils';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  zoomPlugin
);

// Custom tooltip positioner - 50px from cursor, 50px above nearest point
Tooltip.positioners.topCorner = function(elements, eventPosition) {
  const chart = this.chart;
  const chartCenter = (chart.chartArea.left + chart.chartArea.right) / 2;
  const chartVerticalCenter = (chart.chartArea.top + chart.chartArea.bottom) / 2;
  const isOnRightSide = eventPosition.x > chartCenter;
  const isOnTopSide = eventPosition.y < chartVerticalCenter;

  let x = isOnRightSide ? eventPosition.x - 150 : eventPosition.x + 150;
  let y = isOnTopSide ? chart.chartArea.top + 100 : chart.chartArea.bottom - 100;
  return {
    x,
    y,
    xAlign: isOnRightSide ? 'right' : 'left',
    yAlign: isOnTopSide ? 'top' : 'bottom',
  };
};

const DEFAULT_RANGE_SIZE = 100;

export default function ChartContainer({
  groupName,
  chartName,
  displayName,
  unit,
  config,
  engineFilter,
  onFullscreen,
}) {
  const [totalCommits, setTotalCommits] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // viewRange stores the requested range: either { last: N } or { startIdx, endIdx }
  const [viewRange, setViewRange] = useState({ last: DEFAULT_RANGE_SIZE });
  const chartRef = useRef(null);
  const isResettingZoom = useRef(false);

  // Fetch data for the current view range
  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);

      try {
        let options = {};
        if (viewRange.last) {
          // Initial load: get last N commits
          options = { last: viewRange.last };
        } else if (viewRange.startIdx !== undefined && viewRange.endIdx !== undefined) {
          // Navigation: use index-based range
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
        if (!cancelled) {
          setError(err.message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadData();

    return () => {
      cancelled = true;
    };
  }, [groupName, chartName, viewRange]);

  // Compute display range info from chartData (for rendering only)
  const displayRangeInfo = useMemo(() => {
    if (!chartData) return { startIdx: 0, endIdx: 0, total: 0, rangeSize: 0 };
    const total = chartData.originalLength || chartData.commits?.length || 0;
    const rangeSize = chartData.commits?.length || 0;
    const req = chartData.requestedRange || {};
    const startIdx = req.startIndex ?? (total - rangeSize);
    const endIdx = req.endIndex ?? (total - 1);
    return { startIdx, endIdx, total, rangeSize };
  }, [chartData]);

  // Compute current range from viewRange state (for navigation calculations)
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
    return {
      startIdx,
      endIdx,
      total,
      rangeSize: endIdx - startIdx + 1,
    };
  }, [viewRange, totalCommits]);

  const isAtStart = displayRangeInfo.startIdx === 0;
  const isAtEnd = displayRangeInfo.endIdx >= displayRangeInfo.total - 1;
  const currentRangeSize = displayRangeInfo.rangeSize;

  // Navigation handlers - use functional updates to get latest state
  const handleGoToStart = useCallback(() => {
    const range = getCurrentRange();
    if (range.startIdx === 0 || !range.total) return;
    setViewRange({
      startIdx: 0,
      endIdx: Math.min(range.rangeSize - 1, range.total - 1),
    });
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
    setViewRange({
      startIdx: newStartIdx,
      endIdx: Math.min(newEndIdx, range.total - 1),
    });
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

    // Clamp to bounds
    if (newStartIdx < 0) {
      newStartIdx = 0;
      newEndIdx = newRangeSize - 1;
    }
    if (newEndIdx >= range.total) {
      newEndIdx = range.total - 1;
      newStartIdx = Math.max(0, newEndIdx - newRangeSize + 1);
    }

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

    // Clamp to bounds
    if (newStartIdx < 0) {
      newStartIdx = 0;
      newEndIdx = Math.min(newRangeSize - 1, range.total - 1);
    }
    if (newEndIdx >= range.total) {
      newEndIdx = range.total - 1;
      newStartIdx = Math.max(0, newEndIdx - newRangeSize + 1);
    }

    setViewRange({ startIdx: newStartIdx, endIdx: newEndIdx });
  }, [getCurrentRange]);

  const handleShowFullRange = useCallback(() => {
    const range = getCurrentRange();
    if (!range.total) return;
    setViewRange({ startIdx: 0, endIdx: range.total - 1 });
  }, [getCurrentRange]);

  const isFullRange = isAtStart && isAtEnd;

  // Handle drag selection zoom
  const handleDragZoom = useCallback((startDataIdx, endDataIdx) => {
    if (!chartData?.commits || !chartData.requestedRange) return;

    const numCommits = chartData.commits.length;
    if (numCommits < 2) return;

    const rangeStart = chartData.requestedRange.startIndex;
    const rangeEnd = chartData.requestedRange.endIndex;
    const total = chartData.originalLength || rangeEnd + 1;

    // Map chart indices to original dataset indices using linear interpolation
    // This correctly handles downsampled data where numCommits < (rangeEnd - rangeStart + 1)
    const minIdx = Math.min(startDataIdx, endDataIdx);
    const maxIdx = Math.max(startDataIdx, endDataIdx);
    const globalStartIdx = rangeStart + Math.round(minIdx / (numCommits - 1) * (rangeEnd - rangeStart));
    const globalEndIdx = rangeStart + Math.round(maxIdx / (numCommits - 1) * (rangeEnd - rangeStart));

    // Ensure minimum range
    if (globalEndIdx - globalStartIdx < 5) return;

    setViewRange({
      startIdx: Math.max(0, globalStartIdx),
      endIdx: Math.min(total - 1, globalEndIdx),
    });
  }, [chartData]);

  // Process series data with filters and renaming
  const processedData = useMemo(() => {
    if (!chartData?.series || !chartData?.commits) return null;

    const { series, commits } = chartData;
    const datasets = [];
    const labels = commits.map(c => formatDate(c.timestamp));

    Object.entries(series).forEach(([seriesName, points]) => {
      // Apply removed datasets filter
      if (config.removedDatasets?.has(seriesName)) return;

      // Apply engine filter
      if (engineFilter !== 'all') {
        const engine = seriesName.split(':')[0].toLowerCase();
        if (engine !== engineFilter && !seriesName.toLowerCase().includes(engineFilter)) {
          return;
        }
      }

      // Rename series if needed
      let displaySeriesName = seriesName;
      if (config.renamedDatasets) {
        const caseInsensitive = {};
        Object.entries(config.renamedDatasets).forEach(([k, v]) => {
          caseInsensitive[k.toLowerCase()] = v;
        });
        displaySeriesName = caseInsensitive[seriesName.toLowerCase()] || seriesName;
      }

      // Check if hidden by default
      const hidden = config.hiddenDatasets?.has(seriesName) ||
                     config.hiddenDatasets?.has(displaySeriesName);

      datasets.push({
        label: displaySeriesName,
        data: points,
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
  }, [chartData, config, engineFilter]);

  // Handle click on chart point to open commit on GitHub
  const handleChartClick = useCallback((event, elements) => {
    if (!elements.length || !processedData?.commits) return;
    const dataIndex = elements[0].index;
    const commit = processedData.commits[dataIndex];
    if (commit?.id) {
      window.open(`https://github.com/vortex-data/vortex/commit/${commit.id}`, '_blank');
    }
  }, [processedData]);

  // Chart.js options with drag zoom
  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    onClick: handleChartClick,
    onHover: (event, elements) => {
      event.native.target.style.cursor = elements.length ? 'pointer' : 'default';
    },
    interaction: {
      mode: 'index',
      intersect: true,
    },
    plugins: {
      legend: {
        position: 'top',
        align: 'start',
        labels: {
          boxWidth: 12,
          padding: 8,
          font: {
            size: 11,
            family: 'Geist, sans-serif',
          },
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
        position: 'topCorner',
        caretSize: 0,
        itemSort: (a, b) => b.parsed.y - a.parsed.y,
        // Limit to top 10 items by value to prevent tooltip from getting too large
        filter: (item, _index, items) => {
          if (items.length <= 10) return item.parsed.y != null;
          const validItems = items.filter(i => i.parsed.y != null);
          if (validItems.length <= 10) return item.parsed.y != null;
          const sorted = [...validItems].sort((a, b) => (b.parsed.y ?? 0) - (a.parsed.y ?? 0));
          const top10 = sorted.slice(0, 10);
          return top10.some(i => i.datasetIndex === item.datasetIndex);
        },
        callbacks: {
          title: (items) => {
            if (!items.length || !processedData?.commits) return '';
            const commit = processedData.commits[items[0].dataIndex];
            if (!commit) return items[0].label;
            const author = commit.author || 'Unknown';
            return `${formatDate(commit.timestamp)} — ${author}\n(${commit.id?.slice(0, 7) || ''}) ${commit.message || ''}`;
          },
          label: (item) => {
            const value = item.parsed.y;
            if (value == null) return null;
            const formattedValue = value < 1 ? value.toFixed(4) : value.toFixed(2);
            return `${item.dataset.label}: ${formattedValue} ${unit || ''}`;
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
            // Prevent infinite loop from resetZoom triggering onZoomComplete
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
            // Reset chart zoom state
            isResettingZoom.current = true;
            chart.resetZoom();
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.12)',
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45,
          font: {
            size: 10,
            family: 'Geist, sans-serif',
          },
          maxTicksLimit: 10,
          callback: function(value, index, ticks) {
            // Always show first and last tick
            if (index === 0 || index === ticks.length - 1) {
              return this.getLabelForValue(value);
            }
            // Show intermediate ticks based on maxTicksLimit
            const step = Math.ceil(ticks.length / 10);
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
        grid: {
          color: 'rgba(0, 0, 0, 0.12)',
        },
        ticks: {
          font: {
            size: 11,
            family: 'Geist Mono, monospace',
          },
        },
        title: {
          display: !!unit,
          text: unit || '',
          font: {
            size: 11,
            family: 'Geist, sans-serif',
          },
        },
      },
    },
  }), [unit, processedData, handleDragZoom, handleChartClick]);

  // Fullscreen handler
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

  // Show placeholder only on initial load (no data yet)
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

  if (!processedData || processedData.datasets.length === 0) {
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
          <button
            className="chart-zoom-btn"
            onClick={handleFullscreen}
            disabled={loading}
            data-tooltip="Fullscreen"
          >
            <Expand size={14} />
          </button>
        </div>
      </div>
      <div className={`chart-canvas-wrapper ${showOverlay ? 'loading' : ''}`}>
        <Line
          ref={chartRef}
          data={{
            labels: processedData.labels,
            datasets: processedData.datasets,
          }}
          options={options}
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
