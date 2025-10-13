"use strict";

/**
 * BenchmarkApp - Future application class for when we fully adopt the modular system.
 * Currently not used to avoid interference with the existing system.
 * This class shows how the modular system would work when fully activated.
 */

import { UIManager } from './ui-manager.js';
import { chartManager } from './chart-manager.js';
import { workerManager } from './worker-manager.js';
import { zoomSync } from './zoom-sync.js';
import { utils } from './utils.js';
import { CONFIG } from './config.js';

export class BenchmarkApp {
  constructor() {
    // Initialize state
    this.state = {
      currentView: "grid",
      expandedSections: new Set(),
      activeCategory: "all",
      activeTag: "all",
      activeEngines: new Set(["all"]),
      searchTerm: "",
      charts: [],
      chartInstances: new Map(),
      pendingZoomUpdates: new Map(),
      lastWindowWidth: window.innerWidth,
      isResizing: false,
    };

    // Initialize managers
    this.uiManager = new UIManager(this.state, chartManager);
    this.customConfigs = null;
    this.isInitialized = false;
  }

  /**
   * Initialize the application.
   * Note: Currently simplified to avoid interference with existing system.
   * @param {Array} customConfigs - Optional custom configurations
   */
  async initialize(customConfigs = null) {
    if (this.isInitialized) {
      console.log('Application already initialized, refreshing with new configs');
      this.refresh(customConfigs);
      return;
    }

    console.log('Initializing Vortex Benchmarks application (modular system)');

    // Store custom configs for later use
    this.customConfigs = customConfigs;

    // Initialize UI manager
    this.uiManager.initialize();

    // Initialize workers
    workerManager.init();

    // Initialize zoom sync
    this.initializeZoomSync();

    // Load state from URL
    this.uiManager.loadStateFromURL();

    // Load and render data
    await this.loadAndRender();

    // Set up event listeners
    this.setupEventListeners();

    // Mark as initialized
    this.isInitialized = true;

    console.log('Application initialized successfully');
  }

  /**
   * Initialize zoom synchronization.
   */
  initializeZoomSync() {
    // Set up the debounced sync function
    window.debouncedSyncZoom = utils.debounce((categoryName, update) => {
      this.state.pendingZoomUpdates.set(categoryName, update);
      zoomSync.syncCharts(categoryName, this.state, utils);
    }, CONFIG.DEBOUNCE_DELAY);
  }

  /**
   * Load benchmark data and render the UI.
   */
  async loadAndRender() {
    try {
      // Show loading state
      this.showLoadingState();

      // Load the data
      const data = await this.loadBenchmarkData();

      // Hide loading state
      this.hideLoadingState();

      // Render benchmarks
      this.uiManager.renderBenchmarks(data, this.customConfigs);

      // Apply initial filters if any
      this.applyInitialFilters();

    } catch (error) {
      console.error('Failed to load benchmark data:', error);
      this.showError('Failed to load benchmark data. Please refresh the page.');
    }
  }

  /**
   * Load benchmark data from the server.
   */
  async loadBenchmarkData() {
    // Check if we're using the old data loading method
    if (window.downloadedData) {
      // Use existing downloaded data
      return window.downloadedData;
    }

    // Use worker manager to load and process data
    const benchmarkDataUrl = 'https://vortex-benchmark-results-database.s3.amazonaws.com/data.json.gz';
    const commitsDataUrl = 'https://vortex-benchmark-results-database.s3.amazonaws.com/commits.json';

    // Fetch data
    const [benchmarkResponse, commitsResponse] = await Promise.all([
      fetch(benchmarkDataUrl),
      fetch(commitsDataUrl)
    ]);

    // Decompress benchmark data if needed
    let benchmarkData;
    if (benchmarkDataUrl.endsWith('.gz')) {
      const blob = await benchmarkResponse.blob();
      const stream = blob.stream().pipeThrough(new DecompressionStream('gzip'));
      const text = await new Response(stream).text();
      benchmarkData = text;
    } else {
      benchmarkData = await benchmarkResponse.text();
    }

    const commitsData = await commitsResponse.text();

    // Process data using worker
    const processedData = await workerManager.processData(
      benchmarkData,
      commitsData,
      this.customConfigs?.map(c => Array.isArray(c) ? c[0] : c.name) || null,
      (progress, message) => {
        console.log(`Processing: ${progress}% - ${message}`);
      }
    );

    // Store globally for compatibility
    window.downloadedData = processedData;

    return processedData;
  }

  /**
   * Set up event listeners for UI controls.
   */
  setupEventListeners() {
    // Expand/Collapse All buttons
    const expandAll = document.getElementById('expand-all');
    const collapseAll = document.getElementById('collapse-all');

    if (expandAll) {
      expandAll.addEventListener('click', () => this.uiManager.expandAll());
    }

    if (collapseAll) {
      collapseAll.addEventListener('click', () => this.uiManager.collapseAll());
    }

    // Category filter
    const categoryFilter = document.getElementById('category-filter');
    if (categoryFilter) {
      categoryFilter.addEventListener('change', (e) => {
        this.uiManager.filterByCategory(e.target.value);
      });
    }

    // Search filter
    const searchFilter = document.getElementById('search-filter');
    if (searchFilter) {
      const debouncedSearch = utils.debounce((term) => {
        this.uiManager.search(term);
      }, CONFIG.SEARCH_DEBOUNCE);

      searchFilter.addEventListener('input', (e) => {
        debouncedSearch(e.target.value);
      });
    }

    // Window resize handler
    window.addEventListener('resize', utils.debounce(() => {
      this.handleResize();
    }, CONFIG.RESIZE_DEBOUNCE));
  }

  /**
   * Apply initial filters based on URL parameters or state.
   */
  applyInitialFilters() {
    if (this.state.activeCategory !== 'all' || this.state.searchTerm) {
      this.uiManager.applyFilters({
        category: this.state.activeCategory,
        search: this.state.searchTerm
      });
    }
  }

  /**
   * Handle window resize events.
   */
  handleResize() {
    const currentWidth = window.innerWidth;
    const wasDesktop = this.state.lastWindowWidth > CONFIG.MOBILE_BREAKPOINT;
    const isDesktop = currentWidth > CONFIG.MOBILE_BREAKPOINT;

    // Check if we've crossed the mobile/desktop boundary
    if (wasDesktop !== isDesktop) {
      console.log(`Switched to ${isDesktop ? 'desktop' : 'mobile'} view`);
      // Could trigger a re-render or adjustment here if needed
    }

    this.state.lastWindowWidth = currentWidth;
  }

  /**
   * Show loading state.
   */
  showLoadingState() {
    // Remove any existing loading indicator
    this.hideLoadingState();

    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading-indicator';
    loadingDiv.className = 'loading-indicator';
    loadingDiv.innerHTML = `
      <div class="loading-spinner"></div>
      <div class="loading-text">Loading benchmark data...</div>
    `;

    if (this.uiManager.container) {
      this.uiManager.container.appendChild(loadingDiv);
    }
  }

  /**
   * Hide loading state.
   */
  hideLoadingState() {
    const loadingDiv = document.getElementById('loading-indicator');
    if (loadingDiv) {
      loadingDiv.remove();
    }
  }

  /**
   * Show error message.
   */
  showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;

    if (this.uiManager.container) {
      this.uiManager.container.appendChild(errorDiv);
    }
  }

  /**
   * Refresh the application with new configurations.
   */
  refresh(customConfigs = null) {
    this.customConfigs = customConfigs || this.customConfigs;
    this.loadAndRender();
  }

  /**
   * Destroy the application and clean up resources.
   */
  destroy() {
    this.uiManager.destroy();
    this.state = null;
    this.customConfigs = null;
    this.isInitialized = false;
  }
}