import { SERIES_COLOR_MAP, FALLBACK_PALETTE, CHART_NAME_MAP } from './config';

// Simple hash function for color selection
function simpleHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}

export function stringToColor(str) {
  if (SERIES_COLOR_MAP[str]) {
    return SERIES_COLOR_MAP[str];
  }

  const lowerStr = str
    .replace(/^DataFusion:/i, 'datafusion:')
    .replace(/^DuckDB:/i, 'duckdb:')
    .replace(/^Vortex:/i, 'vortex:')
    .replace(/^Arrow:/i, 'arrow:');

  if (lowerStr !== str && SERIES_COLOR_MAP[lowerStr]) {
    return SERIES_COLOR_MAP[lowerStr];
  }

  const index = simpleHash(str) % FALLBACK_PALETTE.length;
  return FALLBACK_PALETTE[index];
}

export function remapChartName(name) {
  if (CHART_NAME_MAP[name]) {
    return CHART_NAME_MAP[name];
  }
  // Convert dashes to spaces for readability
  return name.replace(/-/g, ' ');
}

export function formatDate(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${months[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
}

export function formatTime(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

export function debounce(func, wait) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

export function throttle(func, limit) {
  let inThrottle;
  return function (...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

export function isMobile() {
  return window.innerWidth <= 768;
}

export function getBenchmarkDescription(categoryName) {
  if (categoryName.startsWith('TPC-H')) {
    const match = categoryName.match(/SF=(\d+)/);
    const sf = match ? match[1] : null;
    const sfDesc = sf ? `at SF=${sf} (~${sf === '1' ? '1GB' : sf === '10' ? '10GB' : sf === '100' ? '100GB' : '1TB'} of data)` : '';
    if (categoryName.includes('NVMe')) {
      return `TPC-H benchmark queries on local NVMe storage ${sfDesc}`;
    } else if (categoryName.includes('S3')) {
      return `TPC-H benchmark queries against S3 storage ${sfDesc}`;
    }
  }
  if (categoryName.startsWith('TPC-DS')) {
    const match = categoryName.match(/SF=(\d+)/);
    const sf = match ? match[1] : null;
    const sfDesc = sf ? `at SF=${sf}` : '';
    return `TPC-DS benchmark queries on local NVMe storage ${sfDesc}`;
  }
  const descriptions = {
    'Random Access': 'Tests performance of selecting arbitrary row indices from a file on NVMe storage',
    'Compression': 'Measures encoding and decoding throughput (MB/s) for Vortex and Parquet files',
    'Compression Size': 'Compares compressed file sizes across different encoding strategies',
    'Clickbench': "ClickHouse's analytical benchmark suite on web analytics data",
    'Clickbench Sorted': 'ClickBench queries over data globally sorted by event date and event time',
    'Statistical and Population Genetics': 'Statistical and population genetics queries on gnomAD dataset',
  };
  return descriptions[categoryName] || '';
}
