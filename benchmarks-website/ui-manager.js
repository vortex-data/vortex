"use strict";

import { BenchmarkFactory } from './benchmark-factory.js';
import { BenchmarkRenderer } from './benchmark-renderer.js';
import { BENCHMARK_GROUPS } from './config.js';
import { utils } from './utils.js';

/**
 * UIManager class coordinates the rendering and management of all benchmark sections.
 * It handles state management, event handling, and coordination between components.
 */
export class UIManager {
  /**
   * @param {Object} state - The application state
   * @param {Object} chartManager - The chart manager instance
   */
  constructor(state, chartManager) {
    this.state = state;
    this.chartManager = chartManager;
    this.renderers = new Map();
    this.benchmarkData = null;
    this.container = null;
  }

  /**
   * Initialize the UI manager and set up the container.
   * @param {string} containerId - ID of the container element
   */
  initialize(containerId = 'benchmarks-container') {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      // Create container if it doesn't exist
      this.container = document.createElement('div');
      this.container.id = containerId;
      this.container.className = 'benchmarks-container';

      // Find the main content area or body
      const mainContent = document.querySelector('main') || document.body;
      mainContent.appendChild(this.container);
    }
  }

  /**
   * Render benchmarks using the new modular system.
   * @param {Array} benchmarkData - Array of benchmark data objects
   * @param {Array} customConfigs - Optional array of custom configurations
   */
  renderBenchmarks(benchmarkData, customConfigs = null) {
    this.benchmarkData = benchmarkData;

    // Clear existing content
    this.clearContainer();

    // Create benchmark instances
    const benchmarkInstances = this.createBenchmarkInstances(benchmarkData, customConfigs);

    // Render each benchmark
    benchmarkInstances.forEach((instance, index) => {
      this.renderBenchmark(instance, benchmarkData[index]);
    });

    // Update state
    this.updateState();
  }

  /**
   * Create benchmark instances based on data and custom configs.
   */
  createBenchmarkInstances(benchmarkData, customConfigs) {
    if (customConfigs) {
      // Use custom configurations if provided (backward compatibility)
      return customConfigs.map(config => {
        const [name, options] = Array.isArray(config) ? config : [config.name, config];
        return BenchmarkFactory.create(name, options || {});
      }).filter(Boolean);
    } else {
      // Create instances based on benchmark data
      return benchmarkData.map(data => {
        return BenchmarkFactory.create(data.name);
      }).filter(Boolean);
    }
  }

  /**
   * Render a single benchmark.
   */
  renderBenchmark(benchmark, data) {
    if (!benchmark) return;

    const renderer = new BenchmarkRenderer(benchmark, this.chartManager);
    const isExpanded = this.state.expandedSections.has(benchmark.name);

    const element = renderer.render(
      this.container,
      data.dataSet || data.dataset, // Handle both property names
      isExpanded
    );

    this.renderers.set(benchmark.name, renderer);
    this.attachEventListeners(element, benchmark.name);
  }

  /**
   * Clear the container and destroy existing renderers.
   */
  clearContainer() {
    // Destroy existing renderers
    for (const renderer of this.renderers.values()) {
      renderer.destroy();
    }
    this.renderers.clear();

    // Clear the container
    if (this.container) {
      this.container.innerHTML = '';
    }
  }

  /**
   * Attach event listeners to a benchmark section.
   */
  attachEventListeners(element, benchmarkName) {
    // Header click is handled in the renderer
    // Add any additional event listeners here

    // Listen for custom events
    element.addEventListener('benchmark:toggle', (e) => {
      this.handleToggle(benchmarkName, e.detail);
    });

    element.addEventListener('benchmark:zoom', (e) => {
      this.handleZoom(benchmarkName, e.detail);
    });
  }

  /**
   * Handle section toggle.
   */
  handleToggle(benchmarkName, detail) {
    const renderer = this.renderers.get(benchmarkName);
    if (!renderer) return;

    const isExpanded = renderer.getExpanded();

    // Update state
    if (isExpanded) {
      this.state.expandedSections.add(benchmarkName);
    } else {
      this.state.expandedSections.delete(benchmarkName);
    }

    // Update URL if needed
    this.updateURL();
  }

  /**
   * Handle zoom events for synchronization.
   */
  handleZoom(benchmarkName, zoomDetails) {
    // This would integrate with the zoom-sync module
    if (window.zoomSync) {
      window.zoomSync.syncZoom(benchmarkName, zoomDetails);
    }
  }

  /**
   * Toggle a specific section.
   * @param {string} benchmarkName - Name of the benchmark to toggle
   */
  toggleSection(benchmarkName) {
    const renderer = this.renderers.get(benchmarkName);
    if (renderer) {
      renderer.toggle();
      this.handleToggle(benchmarkName);
    }
  }

  /**
   * Expand all sections.
   */
  expandAll() {
    for (const [name, renderer] of this.renderers) {
      renderer.setExpanded(true);
      this.state.expandedSections.add(name);
    }
    this.updateURL();
  }

  /**
   * Collapse all sections.
   */
  collapseAll() {
    for (const [name, renderer] of this.renderers) {
      renderer.setExpanded(false);
    }
    this.state.expandedSections.clear();
    this.updateURL();
  }

  /**
   * Apply filters to the benchmarks.
   * @param {Object} filters - Filter criteria
   */
  applyFilters(filters = {}) {
    const {
      category = this.state.activeCategory,
      tag = this.state.activeTag,
      engine = this.state.activeEngines,
      search = this.state.searchTerm
    } = filters;

    // Update state
    this.state.activeCategory = category;
    this.state.activeTag = tag;
    this.state.activeEngines = engine;
    this.state.searchTerm = search;

    // Apply filters to each renderer
    let visibleCount = 0;
    for (const [name, renderer] of this.renderers) {
      const visible = renderer.updateVisibility({
        category,
        tag,
        engine,
        search
      });
      if (visible) visibleCount++;
    }

    // Update UI elements
    this.updateFilterUI(visibleCount);

    // Update URL
    this.updateURL();
  }

  /**
   * Filter by category.
   * @param {string} category - Category to filter by
   */
  filterByCategory(category) {
    this.applyFilters({ category });
  }

  /**
   * Search benchmarks.
   * @param {string} searchTerm - Search term
   */
  search(searchTerm) {
    this.applyFilters({ search: searchTerm });
  }

  /**
   * Update filter UI elements.
   */
  updateFilterUI(visibleCount) {
    // Update category dropdown if it exists
    const categoryFilter = document.getElementById('category-filter');
    if (categoryFilter) {
      categoryFilter.value = this.state.activeCategory;
    }

    // Update search box if it exists
    const searchFilter = document.getElementById('search-filter');
    if (searchFilter && searchFilter.value !== this.state.searchTerm) {
      searchFilter.value = this.state.searchTerm;
    }

    // Update result count if element exists
    const resultCount = document.getElementById('result-count');
    if (resultCount) {
      const total = this.renderers.size;
      if (visibleCount === total) {
        resultCount.textContent = `Showing all ${total} benchmarks`;
      } else {
        resultCount.textContent = `Showing ${visibleCount} of ${total} benchmarks`;
      }
    }
  }

  /**
   * Update the application state.
   */
  updateState() {
    // Update charts array in state
    this.state.charts = [];
    this.state.chartInstances.clear();

    for (const [name, renderer] of this.renderers) {
      const charts = renderer.getCharts();
      charts.forEach((chart, index) => {
        const chartKey = `${name}-${index}`;
        this.state.charts.push(chart);
        if (chart) {
          this.state.chartInstances.set(chartKey, chart);
        }
      });
    }
  }

  /**
   * Update URL with current state.
   */
  updateURL() {
    if (typeof window === 'undefined') return;

    const params = new URLSearchParams();

    // Add expanded sections
    if (this.state.expandedSections.size > 0) {
      params.set('expanded', Array.from(this.state.expandedSections).join(','));
    }

    // Add filters
    if (this.state.activeCategory && this.state.activeCategory !== 'all') {
      params.set('category', this.state.activeCategory);
    }

    if (this.state.searchTerm) {
      params.set('search', this.state.searchTerm);
    }

    // Update URL without reload
    const newURL = params.toString() ? `?${params.toString()}` : window.location.pathname;
    window.history.replaceState({}, '', newURL);
  }

  /**
   * Load state from URL parameters.
   */
  loadStateFromURL() {
    if (typeof window === 'undefined') return;

    const params = new URLSearchParams(window.location.search);

    // Load expanded sections
    const expanded = params.get('expanded');
    if (expanded) {
      this.state.expandedSections = new Set(expanded.split(','));
    }

    // Load category filter
    const category = params.get('category');
    if (category) {
      this.state.activeCategory = category;
    }

    // Load search term
    const search = params.get('search');
    if (search) {
      this.state.searchTerm = search;
    }
  }

  /**
   * Get a specific renderer.
   * @param {string} benchmarkName - Name of the benchmark
   * @returns {BenchmarkRenderer} The renderer instance or null
   */
  getRenderer(benchmarkName) {
    return this.renderers.get(benchmarkName);
  }

  /**
   * Get all renderers.
   * @returns {Map} Map of all renderers
   */
  getAllRenderers() {
    return this.renderers;
  }

  /**
   * Refresh a specific benchmark.
   * @param {string} benchmarkName - Name of the benchmark to refresh
   */
  refreshBenchmark(benchmarkName) {
    const renderer = this.renderers.get(benchmarkName);
    const data = this.benchmarkData.find(d => d.name === benchmarkName);

    if (renderer && data) {
      const parent = renderer.container.parentNode;
      const nextSibling = renderer.container.nextSibling;

      renderer.destroy();
      this.renderers.delete(benchmarkName);

      const benchmark = BenchmarkFactory.create(benchmarkName);
      const newRenderer = new BenchmarkRenderer(benchmark, this.chartManager);
      const isExpanded = this.state.expandedSections.has(benchmarkName);

      const element = newRenderer.render(
        document.createElement('div'), // Temporary container
        data.dataSet || data.dataset,
        isExpanded
      );

      // Insert at original position
      parent.insertBefore(element, nextSibling);

      this.renderers.set(benchmarkName, newRenderer);
      this.attachEventListeners(element, benchmarkName);
    }
  }

  /**
   * Destroy the UI manager and clean up resources.
   */
  destroy() {
    this.clearContainer();
    this.benchmarkData = null;
    this.container = null;
  }
}