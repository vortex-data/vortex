"use strict";

import { CATEGORY_TAGS } from './config.js';

/**
 * BenchmarkRenderer class handles the rendering of benchmark sections and charts.
 * Each renderer instance manages one benchmark section in the UI.
 */
export class BenchmarkRenderer {
  /**
   * @param {BaseBenchmark} benchmark - The benchmark instance to render
   * @param {Object} chartManager - The chart manager for creating charts
   */
  constructor(benchmark, chartManager) {
    this.benchmark = benchmark;
    this.chartManager = chartManager;
    this.container = null;
    this.charts = [];
    this.chartsContainer = null;
    this.isExpanded = false;
  }

  /**
   * Main render method that creates the complete benchmark section.
   * @param {HTMLElement} parentElement - Parent element to append to
   * @param {Map} benchSet - The benchmark data set
   * @param {boolean} isExpanded - Whether the section should start expanded
   * @returns {HTMLElement} The created container element
   */
  render(parentElement, benchSet, isExpanded = false) {
    this.isExpanded = isExpanded;
    this.container = this.createSection(benchSet);

    // Create the header structure
    const stickyWrapper = this.createStickyWrapper();
    this.container.appendChild(stickyWrapper);

    // Add score summary if applicable
    if (this.benchmark.calculateScore) {
      this.renderScoreSummary(benchSet);
    }

    // Add the description if available
    if (this.benchmark.description) {
      this.renderDescription();
    }

    // Render the charts
    this.renderCharts(benchSet, isExpanded);

    // Append to parent
    parentElement.appendChild(this.container);

    return this.container;
  }

  /**
   * Create the main section container.
   */
  createSection(benchSet) {
    const section = document.createElement("div");
    section.className = "benchmark-set";
    section.setAttribute("data-category", this.benchmark.name);

    // Add tags as data attributes
    const tags = CATEGORY_TAGS[this.benchmark.name] || this.benchmark.tags || [];
    tags.forEach(tag => {
      section.setAttribute(`data-tag-${tag.replace(/[^\w-]/g, '-').toLowerCase()}`, "true");
    });

    // Check if this benchmark has any data
    if (!this.benchmark.hasData(benchSet)) {
      section.classList.add("no-data");
    }

    return section;
  }

  /**
   * Create the sticky header wrapper structure.
   */
  createStickyWrapper() {
    const stickyWrapper = document.createElement("div");
    stickyWrapper.className = "sticky-header-wrapper";

    const stickyContainer = document.createElement("div");
    stickyContainer.className = "sticky-header-container";

    // Add header
    const header = this.createHeader();
    stickyContainer.appendChild(header);

    // Add controls if needed
    const controls = this.createControls();
    if (controls) {
      stickyContainer.appendChild(controls);
    }

    stickyWrapper.appendChild(stickyContainer);
    return stickyWrapper;
  }

  /**
   * Create the expandable header element.
   */
  createHeader() {
    const header = document.createElement("h2");
    header.className = "benchmark-header";
    header.setAttribute("data-expanded", this.isExpanded ? "true" : "false");

    // Toggle icon
    const toggleIcon = document.createElement("span");
    toggleIcon.className = "toggle-icon";
    toggleIcon.textContent = this.isExpanded ? "▼" : "▶";

    // Title
    const title = document.createElement("span");
    title.className = "benchmark-title";
    title.textContent = this.benchmark.displayName || this.benchmark.name;

    // Chart count (if applicable)
    const chartCount = document.createElement("span");
    chartCount.className = "chart-count";

    header.appendChild(toggleIcon);
    header.appendChild(title);
    header.appendChild(chartCount);

    // Add click handler for expansion
    header.addEventListener("click", () => this.toggle());

    return header;
  }

  /**
   * Create control buttons for the section.
   */
  createControls() {
    // Currently no additional controls, but this is where
    // reset zoom, export data, etc. buttons would go
    return null;
  }

  /**
   * Render the description below the header.
   */
  renderDescription() {
    if (!this.benchmark.description) return;

    const descContainer = document.createElement("div");
    descContainer.className = "benchmark-description";

    const descText = document.createElement("p");
    descText.textContent = this.benchmark.description;

    descContainer.appendChild(descText);
    this.container.appendChild(descContainer);
  }

  /**
   * Render the score summary for query benchmarks.
   */
  renderScoreSummary(benchSet) {
    const scores = this.benchmark.calculateScore(benchSet);
    if (!scores) return;

    const summary = this.benchmark.formatScore(scores);
    if (summary) {
      // Insert after sticky wrapper but before charts
      const stickyWrapper = this.container.querySelector('.sticky-header-wrapper');
      if (stickyWrapper && stickyWrapper.nextSibling) {
        this.container.insertBefore(summary, stickyWrapper.nextSibling);
      } else {
        this.container.appendChild(summary);
      }
    }
  }

  /**
   * Render all charts for this benchmark.
   */
  renderCharts(benchSet, isExpanded) {
    // Create charts container
    this.chartsContainer = document.createElement("div");
    this.chartsContainer.className = "chart-grid";

    if (!isExpanded) {
      this.chartsContainer.classList.add("collapsed");
    }

    // Update chart count in header
    let chartCount = 0;
    let visibleChartCount = 0;

    // Render each chart
    if (benchSet && benchSet.size > 0) {
      let chartIndex = 0;

      for (const [queryName, queryData] of benchSet.entries()) {
        // Check if this chart should be shown
        if (!this.benchmark.shouldShowChart(queryName)) continue;

        chartCount++;

        // Skip if no data
        if (!queryData || !queryData.series || queryData.series.size === 0) continue;

        visibleChartCount++;

        // Render the chart
        const chart = this.chartManager.renderChart(
          this.chartsContainer,
          this.benchmark.name,
          queryName,
          queryData,
          this.benchmark.hiddenDatasets,
          this.benchmark.removedDatasets,
          this.benchmark.renamedDatasets,
          chartIndex++
        );

        if (chart) {
          this.charts.push(chart);
        }
      }
    }

    // Update the chart count in the header
    const countElement = this.container.querySelector('.chart-count');
    if (countElement) {
      if (visibleChartCount > 0) {
        countElement.textContent = `(${visibleChartCount} chart${visibleChartCount !== 1 ? 's' : ''})`;
      } else if (chartCount > 0) {
        countElement.textContent = "(no data)";
      } else {
        countElement.textContent = "";
      }
    }

    this.container.appendChild(this.chartsContainer);
  }

  /**
   * Toggle the expanded/collapsed state of this section.
   */
  toggle() {
    this.isExpanded = !this.isExpanded;
    this.setExpanded(this.isExpanded);
  }

  /**
   * Set the expanded state of this section.
   * @param {boolean} expanded - Whether the section should be expanded
   */
  setExpanded(expanded) {
    this.isExpanded = expanded;

    // Update header
    const header = this.container.querySelector('.benchmark-header');
    if (header) {
      header.setAttribute("data-expanded", expanded ? "true" : "false");

      // Update toggle icon
      const toggleIcon = header.querySelector('.toggle-icon');
      if (toggleIcon) {
        toggleIcon.textContent = expanded ? "▼" : "▶";
      }
    }

    // Update charts container
    if (this.chartsContainer) {
      if (expanded) {
        this.chartsContainer.classList.remove("collapsed");
        // Trigger lazy loading of charts if needed
        this.loadVisibleCharts();
      } else {
        this.chartsContainer.classList.add("collapsed");
      }
    }
  }

  /**
   * Load charts that are now visible (for lazy loading).
   */
  loadVisibleCharts() {
    // This would interact with the chart observer for lazy loading
    // The actual implementation depends on the chartManager
    const containers = this.chartsContainer.querySelectorAll('.chart-container');
    containers.forEach(container => {
      // Check if chart needs to be created
      if (container.chartData && !container.chartInstance) {
        // Trigger chart creation through observer or directly
        if (window.chartObserver) {
          window.chartObserver.observe(container);
        }
      }
    });
  }

  /**
   * Destroy this renderer and clean up resources.
   */
  destroy() {
    // Destroy all chart instances
    this.charts.forEach(chart => {
      if (chart && chart.destroy) {
        chart.destroy();
      }
    });
    this.charts = [];

    // Remove the container from DOM
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }

    this.container = null;
    this.chartsContainer = null;
  }

  /**
   * Update the visibility based on filters.
   * @param {Object} filters - Filter criteria
   */
  updateVisibility(filters) {
    const { category, tag, engine, search } = filters;

    let visible = true;

    // Check category filter
    if (category && category !== 'all') {
      const tags = CATEGORY_TAGS[this.benchmark.name] || this.benchmark.tags || [];
      visible = visible && tags.includes(category);
    }

    // Check search filter
    if (search) {
      const searchLower = search.toLowerCase();
      visible = visible && (
        this.benchmark.name.toLowerCase().includes(searchLower) ||
        (this.benchmark.description && this.benchmark.description.toLowerCase().includes(searchLower))
      );
    }

    // Update visibility
    if (this.container) {
      this.container.style.display = visible ? 'block' : 'none';
    }

    return visible;
  }

  /**
   * Get the current expanded state.
   */
  getExpanded() {
    return this.isExpanded;
  }

  /**
   * Get the benchmark instance.
   */
  getBenchmark() {
    return this.benchmark;
  }

  /**
   * Get all chart instances.
   */
  getCharts() {
    return this.charts;
  }
}