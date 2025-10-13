"use strict";

import { BenchmarkFactory } from './benchmark-factory.js';

/**
 * MigrationAdapter provides utilities for converting between old and new configuration formats.
 * This is designed to be a lightweight compatibility layer that doesn't interfere with the existing system.
 *
 * The adapter is currently in "passive mode" - it provides conversion utilities but doesn't
 * actually initialize anything to avoid conflicts with the existing code.js system.
 */
export class MigrationAdapter {
  /**
   * Convert legacy configuration format to new format.
   * Legacy format: Array of [name, config] tuples
   * New format: Array of {name, ...config} objects
   *
   * @param {Array} legacyConfigArray - Array of [name, config] tuples
   * @returns {Array} Array of configuration objects
   */
  static convertLegacyConfig(legacyConfigArray) {
    if (!Array.isArray(legacyConfigArray)) {
      return [];
    }

    return legacyConfigArray.map(item => {
      // Handle both array format [name, config] and object format
      if (Array.isArray(item)) {
        const [name, config] = item;
        return {
          name,
          ...(config || {})
        };
      } else if (typeof item === 'object' && item.name) {
        // Already in new format
        return item;
      } else if (typeof item === 'string') {
        // Just a name, no config
        return { name: item };
      }

      console.warn('Unknown configuration format:', item);
      return null;
    }).filter(Boolean);
  }

  /**
   * Create benchmark instances from legacy configuration.
   * This is a convenience method for testing and gradual migration.
   *
   * @param {Array} legacyConfigArray - Array of [name, config] tuples
   * @returns {Array} Array of benchmark instances
   */
  static createBenchmarksFromLegacy(legacyConfigArray) {
    const configs = this.convertLegacyConfig(legacyConfigArray);
    return configs.map(config => {
      return BenchmarkFactory.create(config.name, config);
    }).filter(Boolean);
  }

  /**
   * Migrate state from old format to new format.
   * This can be used when transitioning from the old to new system.
   *
   * @param {Object} oldState - Old state object
   * @returns {Object} New state object
   */
  static migrateState(oldState) {
    const newState = {
      currentView: oldState.currentView || "grid",
      expandedSections: oldState.expandedSections || new Set(),
      activeCategory: oldState.activeCategory || "all",
      activeTag: oldState.activeTag || "all",
      activeEngines: oldState.activeEngines || new Set(["all"]),
      searchTerm: oldState.searchTerm || "",
      charts: oldState.charts || [],
      chartInstances: oldState.chartInstances || new Map(),
      pendingZoomUpdates: oldState.pendingZoomUpdates || new Map(),
      lastWindowWidth: oldState.lastWindowWidth || window.innerWidth,
      isResizing: oldState.isResizing || false,
    };

    return newState;
  }
}