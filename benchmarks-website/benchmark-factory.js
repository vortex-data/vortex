"use strict";

import { BENCHMARK_CONFIGS, getBenchmarkConfig, getAllBenchmarkNames } from './benchmark-config.js';
import { QueryBenchmark } from './benchmark-types.js';

/**
 * Factory class for creating benchmark instances.
 * Manages caching and provides utility methods for benchmark operations.
 */
export class BenchmarkFactory {
  // Cache for benchmark instances to avoid recreating them
  static cache = new Map();

  /**
   * Create a benchmark instance.
   * @param {string} benchmarkName - Name of the benchmark to create
   * @param {Object} customConfig - Optional custom configuration to override defaults
   * @returns {BaseBenchmark} Instance of the appropriate benchmark class
   */
  static create(benchmarkName, customConfig = {}) {
    // Create a cache key based on the name and custom config
    const cacheKey = JSON.stringify({ name: benchmarkName, config: customConfig });

    // Check cache first
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    // Get the configuration for this benchmark
    const configEntry = getBenchmarkConfig(benchmarkName);
    if (!configEntry) {
      console.error(`Unknown benchmark type: ${benchmarkName}`);
      // Return null instead of throwing to maintain backward compatibility
      return null;
    }

    const { type: BenchmarkClass, config: defaultConfig } = configEntry;

    // Merge configurations - custom config takes precedence
    const finalConfig = {
      name: benchmarkName,
      ...defaultConfig,
      ...customConfig
    };

    // Special handling for Set objects in config merging
    if (customConfig.hiddenDatasets !== undefined) {
      finalConfig.hiddenDatasets = customConfig.hiddenDatasets instanceof Set
        ? customConfig.hiddenDatasets
        : new Set(customConfig.hiddenDatasets);
    }
    if (customConfig.removedDatasets !== undefined) {
      finalConfig.removedDatasets = customConfig.removedDatasets instanceof Set
        ? customConfig.removedDatasets
        : new Set(customConfig.removedDatasets);
    }

    // Create the benchmark instance
    const instance = new BenchmarkClass(finalConfig);

    // Cache the instance
    this.cache.set(cacheKey, instance);

    return instance;
  }

  /**
   * Create benchmark instances from legacy configuration format.
   * This maintains backward compatibility with the old index.html format.
   * @param {Array} legacyConfigs - Array of [name, config] tuples
   * @returns {Array} Array of benchmark instances
   */
  static createFromLegacy(legacyConfigs) {
    return legacyConfigs.map(([name, customConfig]) => {
      return this.create(name, customConfig || {});
    }).filter(Boolean); // Filter out any null values
  }

  /**
   * Get all benchmark instances with default configurations.
   * @returns {Array} Array of all benchmark instances
   */
  static getAll() {
    return getAllBenchmarkNames().map(name => this.create(name));
  }

  /**
   * Get benchmark instances by tag.
   * @param {string} tag - Tag to filter by
   * @returns {Array} Array of benchmark instances with the specified tag
   */
  static getByTag(tag) {
    return this.getAll().filter(benchmark =>
      benchmark.tags && benchmark.tags.includes(tag)
    );
  }

  /**
   * Get all query benchmark instances.
   * @returns {Array} Array of query benchmark instances
   */
  static getQueryBenchmarks() {
    return this.getAll().filter(benchmark =>
      benchmark instanceof QueryBenchmark
    );
  }

  /**
   * Clear the cache.
   * Useful for testing or when configurations change.
   */
  static clearCache() {
    this.cache.clear();
  }

  /**
   * Check if a benchmark configuration exists.
   * @param {string} benchmarkName - Name of the benchmark to check
   * @returns {boolean} True if the benchmark exists
   */
  static exists(benchmarkName) {
    return getBenchmarkConfig(benchmarkName) !== undefined;
  }

  /**
   * Get the configuration for a benchmark without creating an instance.
   * @param {string} benchmarkName - Name of the benchmark
   * @returns {Object} The configuration object or null
   */
  static getConfig(benchmarkName) {
    const configEntry = getBenchmarkConfig(benchmarkName);
    return configEntry ? configEntry.config : null;
  }

  /**
   * Apply custom configurations to multiple benchmarks.
   * Useful for batch operations.
   * @param {Object} configMap - Map of benchmark names to custom configs
   * @returns {Array} Array of configured benchmark instances
   */
  static createMultiple(configMap) {
    return Object.entries(configMap).map(([name, config]) =>
      this.create(name, config)
    ).filter(Boolean);
  }
}