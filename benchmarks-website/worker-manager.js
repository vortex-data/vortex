// Web Worker Manager - handles communication with data processing worker
class WorkerManager {
  constructor() {
    this.worker = null;
    this.messageId = 0;
    this.pendingPromises = new Map();
    this.isReady = false;
    this.fallbackMode = false;
  }

  async initialize() {
    try {
      // Check if Web Workers are supported
      if (typeof Worker === 'undefined') {
        console.warn('Web Workers not supported, falling back to main thread');
        this.fallbackMode = true;
        return true;
      }

      this.worker = new Worker('./data-worker.js');
      
      // Set up message handler
      this.worker.addEventListener('message', (event) => {
        this.handleWorkerMessage(event.data);
      });

      // Set up error handler
      this.worker.addEventListener('error', (error) => {
        console.error('Worker error:', error);
        this.handleWorkerError(error);
      });

      // Wait for worker to be ready
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Worker initialization timeout'));
        }, 5000);

        const readyHandler = (event) => {
          if (event.data.type === 'READY') {
            clearTimeout(timeout);
            this.isReady = true;
            resolve(true);
          }
        };

        this.worker.addEventListener('message', readyHandler, { once: true });
      });

    } catch (error) {
      console.warn('Failed to initialize worker, falling back to main thread:', error);
      this.fallbackMode = true;
      return true;
    }
  }

  handleWorkerMessage(data) {
    const { type, id, payload } = data;

    if (type === 'READY') {
      this.isReady = true;
      return;
    }

    const promise = this.pendingPromises.get(id);
    if (!promise) {
      console.warn('Received message for unknown request ID:', id);
      return;
    }

    this.pendingPromises.delete(id);

    if (type === 'SUCCESS') {
      promise.resolve(payload);
    } else if (type === 'ERROR') {
      promise.reject(new Error(payload.message));
    }
  }

  handleWorkerError(error) {
    // Reject all pending promises
    this.pendingPromises.forEach((promise) => {
      promise.reject(error);
    });
    this.pendingPromises.clear();

    // Switch to fallback mode
    this.fallbackMode = true;
    console.warn('Worker encountered error, switching to fallback mode:', error);
  }

  async sendMessage(type, payload) {
    if (this.fallbackMode) {
      return this.fallbackHandler(type, payload);
    }

    if (!this.isReady) {
      throw new Error('Worker not ready');
    }

    const id = ++this.messageId;
    
    return new Promise((resolve, reject) => {
      this.pendingPromises.set(id, { resolve, reject });

      this.worker.postMessage({
        type,
        payload,
        id
      });

      // Set timeout for the request
      setTimeout(() => {
        if (this.pendingPromises.has(id)) {
          this.pendingPromises.delete(id);
          reject(new Error('Worker request timeout'));
        }
      }, 30000); // 30 second timeout
    });
  }

  // Fallback implementations for when worker is not available
  async fallbackHandler(type, payload) {
    // Import fallback functions (these would be the original functions from code.js)
    switch (type) {
      case 'LOAD_DATA':
        return this.fallbackLoadData();
      
      case 'PROCESS_CHART_DATA':
        return this.fallbackProcessChartData(payload);
      
      case 'FILTER_DATA':
        return this.fallbackFilterData(payload);
      
      case 'CALCULATE_STATS':
        return this.fallbackCalculateStats(payload);
      
      case 'PARSE_JSONL':
        return this.fallbackParseJsonl(payload);
      
      default:
        throw new Error(`Unknown fallback operation: ${type}`);
    }
  }

  // Fallback implementations (simplified versions)
  async fallbackLoadData() {
    const [dataResponse, commitsResponse] = await Promise.all([
      this.fallbackFetchGzippedData(
        "https://vortex-benchmark-results-database.s3.amazonaws.com/data.json.gz"
      ),
      fetch(
        "https://vortex-benchmark-results-database.s3.amazonaws.com/commits.json"
      ).then((r) => r.text()),
    ]);

    const data = this.fallbackParseJsonl({ jsonl: dataResponse });
    const commitsArray = this.fallbackParseJsonl({ jsonl: commitsResponse });

    const commits = {};
    commitsArray.forEach((commit) => {
      commits[commit.id] = commit;
    });

    return { data, commits };
  }

  async fallbackFetchGzippedData(url) {
    const response = await fetch(url);
    const decompressedStream = response.body.pipeThrough(
      new DecompressionStream("gzip")
    );
    const reader = decompressedStream.getReader();
    const decoder = new TextDecoder();
    let result = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      result += decoder.decode(value, { stream: true });
    }

    result += decoder.decode();
    return result;
  }

  fallbackParseJsonl({ jsonl }) {
    return jsonl
      .split("\n")
      .filter((line) => line.trim().length !== 0)
      .map((line) => JSON.parse(line));
  }

  fallbackProcessChartData(payload) {
    // Simplified chart data processing
    const { data, commits, groupName, chartConfig } = payload;
    // Implementation would mirror the worker version but run on main thread
    return this.processDataForCharts(data, commits, groupName, chartConfig);
  }

  fallbackFilterData({ data, filters }) {
    let filtered = data;

    if (filters.category && filters.category !== 'all') {
      filtered = filtered.filter(item => 
        item.benchmark_group === filters.category
      );
    }

    if (filters.search) {
      const searchTerm = filters.search.toLowerCase();
      filtered = filtered.filter(item =>
        item.benchmark?.toLowerCase().includes(searchTerm) ||
        item.engine_dataset?.toLowerCase().includes(searchTerm)
      );
    }

    return filtered;
  }

  fallbackCalculateStats({ data }) {
    // Basic stats calculation on main thread
    const stats = {};
    // Implementation would be similar to worker version
    return stats;
  }

  processDataForCharts(data, commits, groupName, chartConfig) {
    // This mirrors the worker implementation for fallback
    const processedData = {};
    
    const grouped = {};
    data.forEach((item) => {
      if (!grouped[item.benchmark]) {
        grouped[item.benchmark] = [];
      }
      grouped[item.benchmark].push(item);
    });

    Object.entries(grouped).forEach(([benchmarkName, benchmarkData]) => {
      benchmarkData.sort((a, b) => {
        const commitA = commits[a.commit_id];
        const commitB = commits[b.commit_id];
        return new Date(commitA?.timestamp || 0) - new Date(commitB?.timestamp || 0);
      });

      const series = {};
      benchmarkData.forEach((item) => {
        if (!series[item.engine_dataset]) {
          series[item.engine_dataset] = [];
        }
        series[item.engine_dataset].push({
          commit: item.commit_id,
          value: item.value,
          timestamp: commits[item.commit_id]?.timestamp
        });
      });

      processedData[benchmarkName] = {
        series,
        commits: [...new Set(benchmarkData.map(item => item.commit_id))]
          .map(id => commits[id])
          .filter(Boolean)
          .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
      };
    });

    return processedData;
  }

  // Public API methods
  async loadBenchmarkData() {
    return this.sendMessage('LOAD_DATA');
  }

  async processChartData(data, commits, groupName, chartConfig) {
    return this.sendMessage('PROCESS_CHART_DATA', {
      data,
      commits,
      groupName,
      chartConfig
    });
  }

  async filterData(data, filters) {
    return this.sendMessage('FILTER_DATA', { data, filters });
  }

  async calculateStats(data) {
    return this.sendMessage('CALCULATE_STATS', { data });
  }

  async parseJsonl(jsonl) {
    return this.sendMessage('PARSE_JSONL', { jsonl });
  }

  // Cleanup
  terminate() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.pendingPromises.clear();
    this.isReady = false;
  }
}

// Export for use in main application
window.WorkerManager = WorkerManager;