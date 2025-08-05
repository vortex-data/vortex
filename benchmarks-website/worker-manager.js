"use strict";

import { dataProcessor } from './data-processor.js';
import { utils } from './utils.js';

// Worker manager for handling web worker communication and fallbacks
export const workerManager = {
  dataWorker: null,
  chartWorker: null,
  supportsWorkers: typeof Worker !== 'undefined',
  isProcessing: false,
  
  // Initialize workers
  init() {
    if (this.supportsWorkers) {
      try {
        this.dataWorker = new Worker('./data-worker.js');
        this.chartWorker = new Worker('./chart-worker.js');
        return true;
      } catch (error) {
        this.supportsWorkers = false;
        return false;
      }
    }
    return false;
  },

  // Process data using worker or fallback
  async processData(benchmarkData, commitsData, keptGroups, onProgress) {
    if (this.supportsWorkers && this.dataWorker) {
      this.isProcessing = true;
      try {
        const result = await this.processDataWithWorker(benchmarkData, commitsData, keptGroups, onProgress);
        this.isProcessing = false;
        return result;
      } catch (error) {
        this.isProcessing = false;
        throw error;
      }
    } else {
      return this.processDataFallback(benchmarkData, commitsData, keptGroups, onProgress);
    }
  },

  // Process data using web worker
  processDataWithWorker(benchmarkData, commitsData, keptGroups, onProgress) {
    return new Promise((resolve, reject) => {
      const handleMessage = (e) => {
        const { type, result, progress, message, error, stack } = e.data;
        
        switch (type) {
          case 'progress':
            if (onProgress) {
              onProgress(progress, message);
            }
            break;
            
          case 'dataProcessed':
            if (onProgress) {
              onProgress(progress, message);
            }
            // Deserialize the result
            const deserializedResult = result.map(group => ({
              name: group.name,
              dataSet: this.deserializeMap(group.dataSet)
            }));
            
            this.dataWorker.removeEventListener('message', handleMessage);
            resolve(deserializedResult);
            break;
            
          case 'error':
            this.dataWorker.removeEventListener('message', handleMessage);
            reject(new Error(`Worker error: ${error}\n${stack}`));
            break;
        }
      };

      this.dataWorker.addEventListener('message', handleMessage);
      
      this.dataWorker.postMessage({
        type: 'parseData',
        data: {
          benchmarkData,
          commitsData,
          keptGroups
        }
      });
    });
  },

  // Fallback data processing on main thread
  async processDataFallback(benchmarkData, commitsData, keptGroups, onProgress) {
    if (onProgress) onProgress(10, 'Parsing benchmark data...');
    
    // Parse JSONL data
    const parsedBenchmarkData = this.parseJsonl(benchmarkData);
    
    if (onProgress) onProgress(30, 'Parsing commit data...');
    
    const parsedCommitsData = this.parseJsonl(commitsData);
    
    if (onProgress) onProgress(50, 'Processing and grouping data...');
    
    // Convert commits array to object
    const commits = {};
    parsedCommitsData.forEach((commit) => {
      commits[commit.id] = commit;
    });

    // Use existing data processor
    const result = dataProcessor.downloadAndGroupData(
      parsedBenchmarkData,
      commits,
      keptGroups
    );

    if (onProgress) onProgress(100, 'Data processing complete!');
    
    return result;
  },

  // Prepare chart data using worker or fallback
  async prepareChart(chartConfig, onProgress) {
    if (this.supportsWorkers && this.chartWorker) {
      return this.prepareChartWithWorker(chartConfig, onProgress);
    } else {
      return this.prepareChartFallback(chartConfig, onProgress);
    }
  },

  // Prepare chart using web worker
  prepareChartWithWorker(chartConfig, onProgress) {
    return new Promise((resolve, reject) => {
      const handleMessage = (e) => {
        const { type, result, progress, message, error, stack } = e.data;
        
        switch (type) {
          case 'progress':
            if (onProgress) {
              onProgress(progress, message);
            }
            break;
            
          case 'chartPrepared':
            this.chartWorker.removeEventListener('message', handleMessage);
            resolve(result);
            break;
            
          case 'error':
            this.chartWorker.removeEventListener('message', handleMessage);
            reject(new Error(`Worker error: ${error}\n${stack}`));
            break;
        }
      };

      this.chartWorker.addEventListener('message', handleMessage);
      
      this.chartWorker.postMessage({
        type: 'prepareChart',
        data: {
          ...chartConfig,
          isMobile: utils.isMobile()
        }
      });
    });
  },

  // Prepare multiple charts in batch
  async prepareBatchCharts(chartConfigs, onProgress) {
    if (this.supportsWorkers && this.chartWorker) {
      return this.prepareBatchChartsWithWorker(chartConfigs, onProgress);
    } else {
      return this.prepareBatchChartsFallback(chartConfigs, onProgress);
    }
  },

  // Prepare batch charts using web worker
  prepareBatchChartsWithWorker(chartConfigs, onProgress) {
    return new Promise((resolve, reject) => {
      const handleMessage = (e) => {
        const { type, results, progress, message, error, stack } = e.data;
        
        switch (type) {
          case 'progress':
            if (onProgress) {
              onProgress(progress, message);
            }
            break;
            
          case 'chartsBatchPrepared':
            if (onProgress) {
              onProgress(progress, message);
            }
            this.chartWorker.removeEventListener('message', handleMessage);
            resolve(results);
            break;
            
          case 'error':
            this.chartWorker.removeEventListener('message', handleMessage);
            reject(new Error(`Worker error: ${error}\n${stack}`));
            break;
        }
      };

      this.chartWorker.addEventListener('message', handleMessage);
      
      this.chartWorker.postMessage({
        type: 'prepareBatchCharts',  
        data: {
          charts: chartConfigs,
          isMobile: utils.isMobile()
        }
      });
    });
  },

  // Fallback chart preparation on main thread
  async prepareChartFallback(chartConfig, onProgress) {
    // This would use the existing chart manager logic
    // For now, return the config as-is to maintain compatibility
    if (onProgress) onProgress(100, 'Chart prepared (fallback)');
    return chartConfig;
  },

  // Fallback batch chart preparation on main thread
  async prepareBatchChartsFallback(chartConfigs, onProgress) {
    const results = [];
    for (let i = 0; i < chartConfigs.length; i++) {
      const config = chartConfigs[i];
      results.push(await this.prepareChartFallback(config));
      
      if (onProgress && i % 10 === 0) {
        onProgress((i / chartConfigs.length) * 100, `Preparing charts: ${i}/${chartConfigs.length}`);
      }
    }
    
    if (onProgress) onProgress(100, 'Chart preparation complete!');
    return results;
  },

  // Utility function to parse JSONL
  parseJsonl(jsonl) {
    return jsonl
      .split("\n")
      .filter((line) => line.trim().length !== 0)
      .map((line) => JSON.parse(line));
  },

  // Helper to deserialize Maps from worker
  deserializeMap(serializedMap) {
    if (!Array.isArray(serializedMap)) {
      return serializedMap; // Return as-is if not a serialized array
    }
    
    const result = new Map();
    for (const [key, value] of serializedMap) {
      if (value && value.series && typeof value.series === 'object') {
        // Convert series object back to Map
        const seriesMap = new Map();
        for (const [seriesKey, seriesValue] of Object.entries(value.series)) {
          seriesMap.set(seriesKey, seriesValue);
        }
        result.set(key, {
          ...value,
          series: seriesMap
        });
      } else {
        result.set(key, value);
      }
    }
    return result;
  },

  // Clean up workers (with safety check)
  terminate() {
    if (this.isProcessing) {
      setTimeout(() => this.terminate(), 1000);
      return;
    }
    
    if (this.dataWorker) {
      this.dataWorker.terminate();
      this.dataWorker = null;
    }
    if (this.chartWorker) {
      this.chartWorker.terminate();
      this.chartWorker = null;
    }
  }
};