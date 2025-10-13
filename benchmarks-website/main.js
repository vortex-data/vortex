"use strict";

/**
 * Main entry point for the modularized Vortex Benchmarks application.
 *
 * IMPORTANT: This modular system is designed to run alongside the existing code.js system
 * without any interference. The existing website continues to work exactly as before.
 *
 * The modular system provides:
 * - Centralized configuration (benchmark-config.js)
 * - Type-specific benchmark classes (benchmark-types.js)
 * - Factory pattern for creating instances (benchmark-factory.js)
 * - UI components ready for future use (benchmark-renderer.js, ui-manager.js)
 *
 * Access the modular API via window.VortexBenchmarks
 */

import { BenchmarkApp } from './benchmark-app.js';
import { BenchmarkFactory } from './benchmark-factory.js';
import { MigrationAdapter } from './migration-adapter.js';
import { UIManager } from './ui-manager.js';
import { chartManager } from './chart-manager.js';
import { workerManager } from './worker-manager.js';
import { zoomSync } from './zoom-sync.js';
import { utils } from './utils.js';
import { scoring } from './scoring.js';

// Store the existing initAndRender from code.js (if it exists)
const existingInitAndRender = window.initAndRender;

// DO NOT override the existing initAndRender from code.js
// The modular system is available but passive - it doesn't interfere with the existing system

// Expose the modular API for future use
window.VortexBenchmarks = {
  // Core classes
  BenchmarkApp,
  BenchmarkFactory,
  UIManager,
  MigrationAdapter,

  // Utility modules
  chartManager,
  workerManager,
  zoomSync,
  utils,
  scoring,

  // Convenience methods
  createApp() {
    return new BenchmarkApp();
  },

  // Initialize a new app with configurations
  async initialize(configs) {
    const app = new BenchmarkApp();
    await app.initialize(configs);
    return app;
  },

  // Get benchmark configuration without creating instance
  getBenchmarkConfig(name) {
    return BenchmarkFactory.getConfig(name);
  },

  // Create a benchmark instance
  createBenchmark(name, config) {
    return BenchmarkFactory.create(name, config);
  },

  // Convert legacy config format to new format
  convertLegacyConfig(legacyArray) {
    return MigrationAdapter.convertLegacyConfig(legacyArray);
  }
};

// Preserve the old downloadedData global if it exists
if (window.downloadedData) {
  window.VortexBenchmarks.downloadedData = window.downloadedData;
}

// Log status - quieter logging to avoid confusion
console.log('Vortex Benchmarks: Modular system ready (passive mode)');
console.log('Access via window.VortexBenchmarks');

// Export for ES6 modules
export { BenchmarkApp, BenchmarkFactory, UIManager, MigrationAdapter };
export { chartManager, workerManager, zoomSync, utils, scoring };