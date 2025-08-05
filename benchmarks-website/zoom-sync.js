"use strict";

import { CONFIG } from './config.js';

// Zoom synchronization module
// This module handles synchronizing zoom levels across multiple charts in the same category
// Dependencies: CONFIG, utils, and state objects must be available globally

// Initialize debounced zoom sync function
let debouncedSyncZoom = null;
let throttledZoomHandler = null;

// Cache for chart containers to avoid repeated DOM queries
const chartContainerCache = new Map();

// IntersectionObserver for tracking visible charts
let visibilityObserver = null;
const visibleCharts = new Set();

// Initialize visibility observer for smart batching
function initializeVisibilityObserver() {
  if (!visibilityObserver && "IntersectionObserver" in window) {
    visibilityObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const container = entry.target;
          const chartKey = container.getAttribute('data-chart-key');
          
          if (entry.isIntersecting) {
            visibleCharts.add(chartKey);
          } else {
            visibleCharts.delete(chartKey);
          }
        });
      },
      {
        rootMargin: '100px', // Include charts slightly outside viewport
        threshold: 0.1
      }
    );
  }
}

// Update container cache for a category
function updateContainerCache(categoryName) {
  const categorySection = document.querySelector(`[data-category="${categoryName}"]`);
  if (!categorySection) return;
  
  const containers = categorySection.querySelectorAll(".chart-container");
  chartContainerCache.set(categoryName, Array.from(containers));
  
  // Set up visibility tracking for new containers
  containers.forEach((container, index) => {
    const chartKey = `${categoryName}-${index}`;
    container.setAttribute('data-chart-key', chartKey);
    
    if (visibilityObserver) {
      visibilityObserver.observe(container);
    }
  });
}

// Initialize the debounced zoom sync function
function initializeDebouncedSync(state, utils) {
  debouncedSyncZoom = utils.debounce((categoryName) => {
    const update = state.pendingZoomUpdates.get(categoryName);
    if (!update) return;
    
    const { min, max, sourceIndex } = update;
    
    // Performance timing (optional)
    let startTime;
    if (CONFIG.ENABLE_ZOOM_PERFORMANCE_TIMING) {
      startTime = performance.now();
    }
    
    // Get cached containers or update cache
    let chartContainers = chartContainerCache.get(categoryName);
    if (!chartContainers) {
      updateContainerCache(categoryName);
      chartContainers = chartContainerCache.get(categoryName);
    }
    
    if (!chartContainers) return;
    
    // Use requestAnimationFrame for smooth updates
    requestAnimationFrame(() => {
      const updateQueue = [];
      
      chartContainers.forEach((container, index) => {
        // Skip the source chart
        if (index === sourceIndex) return;
        
        const chartKey = `${categoryName}-${index}`;
        
        // Only update visible charts for better performance
        if (!visibleCharts.has(chartKey)) return;
        
        const chartData = state.chartInstances.get(chartKey);
        
        if (chartData?.chart) {
          updateQueue.push({
            chart: chartData.chart,
            min,
            max
          });
        }
      });
      
      // Batch update all charts
      updateQueue.forEach(({ chart, min, max }) => {
        chart.options.scales.x.min = min;
        chart.options.scales.x.max = max;
        chart.update("none");
      });
      
      // Performance timing (optional)
      if (CONFIG.ENABLE_ZOOM_PERFORMANCE_TIMING && startTime) {
        const endTime = performance.now();
        console.log(`Zoom sync for ${categoryName}: ${(endTime - startTime).toFixed(2)}ms (${updateQueue.length} charts updated, ${visibleCharts.size} visible)`);
      }
    });
    
    // Clear the pending update
    state.pendingZoomUpdates.delete(categoryName);
  }, utils.getDebounceDelay());
  
  // Initialize throttled zoom handler for wheel events
  throttledZoomHandler = utils.throttle((categoryName, sourceChart, sourceIndex, state, utils) => {
    synchronizeZoomForCategory(categoryName, sourceChart, sourceIndex, true, state, utils);
  }, CONFIG.ZOOM_THROTTLE_DELAY);
}

// Simplified synchronization function
function synchronizeZoomForCategory(
  categoryName,
  sourceChart,
  sourceIndex,
  isZoom = true,
  state,
  utils
) {
  // Get the current zoom state from the source chart
  const xScale = sourceChart.scales.x;
  const min = xScale.min;
  const max = xScale.max;

  // Store the update for this category (simplified - no complex anchor logic)
  state.pendingZoomUpdates.set(categoryName, { min, max, sourceIndex });

  // Debounce the actual sync operation
  if (debouncedSyncZoom) {
    debouncedSyncZoom(categoryName);
  }
}

// Zoom synchronization module
export const zoomSync = {
  // Initialize the zoom sync module with state reference
  init(state, utils) {
    initializeDebouncedSync(state, utils);
    initializeVisibilityObserver();
  },

  // Recreate debounced sync zoom with new delay (used when mobile/desktop changes)
  recreateDebouncedSync(state, utils) {
    initializeDebouncedSync(state, utils);
  },

  // Main synchronization function with throttling support
  synchronizeZoomForCategory(
    categoryName,
    sourceChart,
    sourceIndex,
    isZoom = true,
    state,
    utils
  ) {
    // Use throttling for wheel-based zoom events for better performance
    if (isZoom && throttledZoomHandler) {
      throttledZoomHandler(categoryName, sourceChart, sourceIndex, state, utils);
    } else {
      synchronizeZoomForCategory(categoryName, sourceChart, sourceIndex, isZoom, state, utils);
    }
  },

  // Update container cache when new charts are added
  updateCacheForCategory(categoryName) {
    updateContainerCache(categoryName);
  },

  // Clean up observers and caches
  cleanup() {
    if (visibilityObserver) {
      visibilityObserver.disconnect();
      visibilityObserver = null;
    }
    chartContainerCache.clear();
    visibleCharts.clear();
  },

  resetZoomForCategory(categoryName, state, utils, CONFIG) {
    // Get cached containers or update cache
    let containers = chartContainerCache.get(categoryName);
    if (!containers) {
      updateContainerCache(categoryName);
      containers = chartContainerCache.get(categoryName);
    }
    
    if (!containers) return;

    const isCurrentlyMobile = utils.isMobile();

    // Use requestAnimationFrame for smooth updates
    requestAnimationFrame(() => {
      containers.forEach((container, index) => {
        const chartKey = `${categoryName}-${index}`;
        const chartData = state.chartInstances.get(chartKey);

        if (chartData?.chart) {
          const chart = chartData.chart;
          const totalCommits = chart.data.labels.length;
          const minIndex = isCurrentlyMobile
            ? 0
            : Math.max(0, totalCommits - CONFIG.DEFAULT_VISIBLE_COMMITS);

          chart.options.scales.x.min = minIndex;
          chart.options.scales.x.max = totalCommits - 1;
          chart.update("none");
        }
      });
    });
  },
};