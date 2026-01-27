"use strict";

import { dataProcessor } from './data-processor.js';

// Worker manager for handling web worker communication and fallbacks
export const workerManager = {
  dataWorker: null,
  supportsWorkers: typeof Worker !== 'undefined',
  isProcessing: false,
  
  // Initialize workers
  init() {
    if (this.supportsWorkers) {
      try {
        this.dataWorker = new Worker('./data-worker.js', { type: 'module' });
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

    // benchmarkData is now already parsed as an array
    const parsedBenchmarkData = Array.isArray(benchmarkData)
      ? benchmarkData
      : this.parseJsonl(benchmarkData);

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
  }
};
