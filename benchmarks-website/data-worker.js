// Data processing web worker for benchmark data
// Handles heavy data transformations off the main thread

class DataProcessor {
  parseJsonl(jsonl) {
    return jsonl
      .split("\n")
      .filter((line) => line.trim().length !== 0)
      .map((line) => JSON.parse(line));
  }

  async fetchGzippedData(url) {
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

  async loadBenchmarkData() {
    try {
      const [dataResponse, commitsResponse] = await Promise.all([
        this.fetchGzippedData(
          "https://vortex-benchmark-results-database.s3.amazonaws.com/data.json.gz"
        ),
        fetch(
          "https://vortex-benchmark-results-database.s3.amazonaws.com/commits.json"
        ).then((r) => r.text()),
      ]);

      const data = this.parseJsonl(dataResponse);
      const commitsArray = this.parseJsonl(commitsResponse);

      const commits = {};
      commitsArray.forEach((commit) => {
        commits[commit.id] = commit;
      });

      return { data, commits };
    } catch (error) {
      throw new Error(`Failed to load benchmark data: ${error.message}`);
    }
  }

  processDataForCharts(data, commits, groupName, chartConfig) {
    // Process and transform data for chart rendering
    const processedData = {};
    
    // Group data by benchmark type
    const grouped = {};
    data.forEach((item) => {
      if (!grouped[item.benchmark]) {
        grouped[item.benchmark] = [];
      }
      grouped[item.benchmark].push(item);
    });

    // Transform data for each chart
    Object.entries(grouped).forEach(([benchmarkName, benchmarkData]) => {
      // Sort by commit timestamp for consistent ordering
      benchmarkData.sort((a, b) => {
        const commitA = commits[a.commit_id];
        const commitB = commits[b.commit_id];
        return new Date(commitA?.timestamp || 0) - new Date(commitB?.timestamp || 0);
      });

      // Extract series data
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

  filterData(data, filters) {
    // Apply various filters to the data
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

    if (filters.engines && filters.engines.length > 0) {
      filtered = filtered.filter(item =>
        filters.engines.some(engine => 
          item.engine_dataset?.includes(engine)
        )
      );
    }

    return filtered;
  }

  calculateSummaryStats(data) {
    // Calculate summary statistics for datasets
    const stats = {};
    
    const grouped = {};
    data.forEach((item) => {
      const key = `${item.benchmark}-${item.engine_dataset}`;
      if (!grouped[key]) {
        grouped[key] = [];
      }
      grouped[key].push(item.value);
    });

    Object.entries(grouped).forEach(([key, values]) => {
      const sorted = values.sort((a, b) => a - b);
      const len = sorted.length;
      
      stats[key] = {
        min: sorted[0],
        max: sorted[len - 1],
        mean: values.reduce((a, b) => a + b, 0) / len,
        median: len % 2 === 0 
          ? (sorted[len / 2 - 1] + sorted[len / 2]) / 2
          : sorted[Math.floor(len / 2)],
        count: len
      };
    });

    return stats;
  }
}

const processor = new DataProcessor();

// Handle messages from main thread
self.addEventListener('message', async (event) => {
  const { type, payload, id } = event.data;

  try {
    let result;

    switch (type) {
      case 'LOAD_DATA':
        result = await processor.loadBenchmarkData();
        break;

      case 'PROCESS_CHART_DATA':
        result = processor.processDataForCharts(
          payload.data, 
          payload.commits, 
          payload.groupName, 
          payload.chartConfig
        );
        break;

      case 'FILTER_DATA':
        result = processor.filterData(payload.data, payload.filters);
        break;

      case 'CALCULATE_STATS':
        result = processor.calculateSummaryStats(payload.data);
        break;

      case 'PARSE_JSONL':
        result = processor.parseJsonl(payload.jsonl);
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }

    // Send result back to main thread
    self.postMessage({
      type: 'SUCCESS',
      id,
      payload: result
    });

  } catch (error) {
    // Send error back to main thread
    self.postMessage({
      type: 'ERROR',
      id,
      payload: {
        message: error.message,
        stack: error.stack
      }
    });
  }
});

// Signal that worker is ready
self.postMessage({ type: 'READY' });