"use strict";

import { CONFIG, SERIES_COLOR_MAP, FALLBACK_PALETTE } from './config.js';

// Utility functions
export const utils = {
  throttle(func, limit) {
    let inThrottle;
    return function (...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => (inThrottle = false), limit);
      }
    };
  },

  debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  },

  isMobile() {
    return window.innerWidth <= CONFIG.MOBILE_BREAKPOINT;
  },

  getDebounceDelay() {
    return utils.isMobile()
      ? CONFIG.MOBILE_DEBOUNCE_DELAY
      : CONFIG.DEBOUNCE_DELAY;
  },

  stringToColor(str) {
    // First try the exact string
    if (SERIES_COLOR_MAP[str]) {
      return SERIES_COLOR_MAP[str];
    }

    // Try lowercase version for backward compatibility with old data
    // This handles cases like "DataFusion:parquet" -> "datafusion:parquet"
    const lowerStr = str
      .replace(/^DataFusion:/i, "datafusion:")
      .replace(/^DuckDB:/i, "duckdb:")
      .replace(/^Vortex:/i, "vortex:")
      .replace(/^Arrow:/i, "arrow:");

    if (lowerStr !== str && SERIES_COLOR_MAP[lowerStr]) {
      return SERIES_COLOR_MAP[lowerStr];
    }

    const hash = new Hashes.MD5().hex(str);
    const index = parseInt(hash.slice(0, 2), 16) % FALLBACK_PALETTE.length;
    return FALLBACK_PALETTE[index];
  },

  batchDOMUpdates(updates) {
    requestAnimationFrame(() => {
      updates.forEach((update) => update());
    });
  },
};