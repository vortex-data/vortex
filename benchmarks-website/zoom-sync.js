"use strict";

// Zoom synchronization module
// This module handles synchronizing zoom levels across multiple charts in the same category
// Dependencies: CONFIG, utils, and state objects must be available globally

// Initialize debounced zoom sync function
let debouncedSyncZoom = null;

// Initialize the debounced zoom sync function
function initializeDebouncedSync(state, utils) {
  debouncedSyncZoom = utils.debounce((categoryName) => {
    const update = state.pendingZoomUpdates.get(categoryName);
    if (!update) return;
    
    const { min, max, sourceIndex } = update;
    
    // Find all charts in this category
    const categorySection = document.querySelector(
      `[data-category="${categoryName}"]`
    );
    if (!categorySection) return;
    
    const chartContainers =
      categorySection.querySelectorAll(".chart-container");
    
    // Use requestAnimationFrame for smooth updates
    requestAnimationFrame(() => {
      chartContainers.forEach((container, index) => {
        // Skip the source chart
        if (index === sourceIndex) return;
        
        const chartKey = `${categoryName}-${index}`;
        const chartData = state.chartInstances.get(chartKey);
        
        if (chartData?.chart) {
          // Apply the same zoom to this chart
          const chart = chartData.chart;
          chart.options.scales.x.min = min;
          chart.options.scales.x.max = max;
          chart.update("none");
        }
      });
    });
    
    // Clear the pending update
    state.pendingZoomUpdates.delete(categoryName);
  }, utils.getDebounceDelay());
}

// Zoom synchronization module
export const zoomSync = {
  // Initialize the zoom sync module with state reference
  init(state, utils) {
    initializeDebouncedSync(state, utils);
  },

  // Recreate debounced sync zoom with new delay (used when mobile/desktop changes)
  recreateDebouncedSync(state, utils) {
    initializeDebouncedSync(state, utils);
  },

  synchronizeZoomForCategory(
    categoryName,
    sourceChart,
    sourceIndex,
    isZoom = true,
    state,
    utils
  ) {
    // Get the current zoom state from the source chart
    const xScale = sourceChart.scales.x;
    let min = xScale.min;
    let max = xScale.max;

    const isCurrentlyMobile = utils.isMobile();

    // Always anchor to the most recent commit when zooming (not on mobile)
    if (isZoom && !isCurrentlyMobile) {
      const totalCommits = sourceChart.data.labels.length;
      const currentRange = max - min;

      // Always keep the most recent commit visible
      max = totalCommits - 1;
      min = Math.max(0, max - currentRange);
    }

    // Store the update for this category
    state.pendingZoomUpdates.set(categoryName, { min, max, sourceIndex });

    // Debounce the actual sync operation
    if (debouncedSyncZoom) {
      debouncedSyncZoom(categoryName);
    }
  },

  resetZoomForCategory(categoryName, state, utils, CONFIG) {
    const section = document.querySelector(
      `[data-category="${categoryName}"]`
    );
    if (!section) return;

    const isCurrentlyMobile = utils.isMobile();
    const containers = section.querySelectorAll(".chart-container");

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
  },
};