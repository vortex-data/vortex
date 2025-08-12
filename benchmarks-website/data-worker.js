"use strict";

// Data processing web worker
// Handles heavy data parsing and processing operations off the main thread

// Fun progress messages for data processing  
const DATA_PROCESSING_MESSAGES = [
  "Crunching numbers like a statistical Sarlacc pit...",
  "Processing data faster than C-3PO speaks languages...",
  "Analyzing benchmarks with the precision of a Terminator...",
  "Computing results with the power of Deep Thought...",
  "Processing benchmarks in parallel universes...",
  "Analyzing performance like a Mentat...",
];

const PARSING_MESSAGES = [
  "Reading files like Gandalf reads ancient texts...",
  "Interpreting data with the wisdom of Yoda...",
  "Parsing JSON like Neo sees the Matrix...",
  "Reading benchmark data like Sherlock reads clues...",
  "Translating data like the Babel fish...",
];

// Fun progress messages for serialization
const NERDY_PROGRESS_MESSAGES = [
  // Hitchhiker's Guide to the Galaxy
  "Calculating the meaning of life, universe, and everything...",
  "Don't panic! Serializing data...",
  "Consulting the Guide about data serialization...",
  "Generating infinite improbability data...",
  
  // Lord of the Rings
  "One does not simply serialize into Mordor...",
  "Precious data must be kept safe, yes precious...",
  "The data shall not pass! (until serialized)",
  "Even the smallest dataset can change the course of loading...",
  
  // Star Wars
  "These aren't the data structures you're looking for...",
  "Using the Force to serialize data...",
  "That's no moon, that's a benchmark dataset!",
  "Help me Obi-Wan, you're my only hope for fast serialization!",
  
  // Star Trek
  "Engaging warp drive for data serialization...",
  "Make it so, Number Data!",
  "Beam me up, Scotty! (after serialization)",
  "Fascinating... this data appears to be logical...",
  
  // Dune
  "The spice must flow... and so must the data!",
  "Fear is the mind-killer, serialization is the data-thriller...",
  "He who controls the data, controls the universe!",
  
  // Matrix
  "There is no spoon... only serialized data...",
  "Welcome to the real world of data processing...",
  "I know kung fu... and data serialization!",
  
  // Douglas Adams / General Sci-Fi
  "Serializing data with the efficiency of a Babel fish...",
  "Processing data at the Restaurant at the End of the Universe...",
  "Vogon poetry is nothing compared to this beautiful data...",
  "Time is an illusion, lunchtime doubly so, but data serialization is eternal...",
];

const EMPTY_GROUP_MESSAGES = [
  "Nothing to see here, move along...",
  "This group is as empty as the void of space...",
  "Skipping faster than the Millennium Falcon in 12 parsecs...",
  "Empty group detected. Resistance is futile, but unnecessary...",
  "These aren't the benchmarks you're looking for...",
  "Group more empty than Tatooine's social calendar...",
];

function getRandomMessage(messages) {
  return messages[Math.floor(Math.random() * messages.length)];
}

// Import configuration constants (need to be duplicated for worker context)
const BENCHMARK_GROUPS = [
  "Random Access",
  "Compression", 
  "Compression Size",
  "Clickbench",
  "TPC-H (NVMe) (SF=1)",
  "TPC-H (S3) (SF=1)", 
  "TPC-H (NVMe) (SF=10)",
  "TPC-H (S3) (SF=10)",
  "TPC-H (NVMe) (SF=100)", 
  "TPC-H (S3) (SF=100)",
  "TPC-H (NVMe) (SF=1000)",
  "TPC-H (S3) (SF=1000)",
  "TPC-DS (NVMe) (SF=1)",
  "TPC-DS (NVMe) (SF=10)"
];

const QUERY_NAME_MAP = {
  "TPCH Q1": "TPC-H Q1",
  "TPCH Q2": "TPC-H Q2", 
  "TPCH Q3": "TPC-H Q3",
  "TPCH Q4": "TPC-H Q4",
  "TPCH Q5": "TPC-H Q5",
  "TPCH Q6": "TPC-H Q6",
  "TPCH Q7": "TPC-H Q7",
  "TPCH Q8": "TPC-H Q8",
  "TPCH Q9": "TPC-H Q9",
  "TPCH Q10": "TPC-H Q10",
  "TPCH Q11": "TPC-H Q11", 
  "TPCH Q12": "TPC-H Q12",
  "TPCH Q13": "TPC-H Q13",
  "TPCH Q14": "TPC-H Q14",
  "TPCH Q15": "TPC-H Q15",
  "TPCH Q16": "TPC-H Q16",
  "TPCH Q17": "TPC-H Q17",
  "TPCH Q18": "TPC-H Q18",
  "TPCH Q19": "TPC-H Q19",
  "TPCH Q20": "TPC-H Q20",
  "TPCH Q21": "TPC-H Q21",
  "TPCH Q22": "TPC-H Q22"
};

// Data processor implementation for worker context
const dataProcessor = {
  parseCommits(commitMetadata) {
    const commits = [];
    Object.values(commitMetadata)
      .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
      .forEach((commit, index) => {
        commit.sortedIndex = index;
        commits.push(commit);
      });
    return commits;
  },

  createMissingCommit(commitId) {
    return {
      author: { email: "daniel.zidan.king@gmail.com", name: "Dan King" },
      committer: { email: "noreply@github.com", name: "GitHub" },
      id: commitId,
      message: "!! This commit is missing from commits.json !!",
      timestamp: "1970-01-01T00:00:00Z",
      tree_id: null,
      url: `https://github.com/vortex-data/vortex/commit/${commitId}`,
    };
  },

  determineGroupId(benchmark) {
    const { name, dataset, storage } = benchmark;

    if (dataset?.tpch) {
      const scaleFactor = dataset.tpch.scale_factor;
      const isNvme = storage === undefined || storage === "nvme";
      return this.getTpchGroupId(scaleFactor, isNvme);
    }

    if (dataset?.tpcds) {
      const scaleFactor = dataset.tpcds.scale_factor;
      const isNvme = storage === undefined || storage === "nvme";
      return this.getTpcdsGroupId(scaleFactor, isNvme);
    }

    if (dataset?.clickbench) return "Clickbench";
    if (name.startsWith("random-access/")) return "Random Access";
    if (name.includes("compress time/")) return "Compression";
    if (name.startsWith("vortex size/")) return "Compression Size";
    if (
      name.startsWith("vortex:raw size/") ||
      name.startsWith("vortex:parquet-zstd size/")
    ) {
      return "Compression Size";
    }
    if (name.startsWith("tpch_q")) {
      const isNvme = storage === undefined || storage === "nvme";
      return isNvme ? "TPC-H (NVMe) (SF=1)" : "TPC-H (S3) (SF=1)";
    }
    if (name.startsWith("tpcds_q")) {
      const isNvme = storage === undefined || storage === "nvme";
      return isNvme ? "TPC-DS (NVMe) (SF=1)" : "TPC-DS (S3) (SF=1)";
    }
    if (name.startsWith("clickbench")) return "Clickbench";

    return null;
  },

  getTpchGroupId(scaleFactor, isNvme) {
    const sf = Number(scaleFactor);
    const storage = isNvme ? "NVMe" : "S3";

    switch (sf) {
      case 1:
        return `TPC-H (${storage}) (SF=1)`;
      case 10:
        return `TPC-H (${storage}) (SF=10)`;
      case 100:
        return `TPC-H (${storage}) (SF=100)`;
      case 1000:
        return `TPC-H (${storage}) (SF=1000)`;
      default:
        console.warn("Unknown scale factor:", scaleFactor);
        return null;
    }
  },

  getTpcdsGroupId(scaleFactor, isNvme) {
    const sf = Number(scaleFactor);
    const storage = isNvme ? "NVMe" : "S3";

    switch (sf) {
      case 1:
        return `TPC-DS (${storage}) (SF=1)`;
      case 10:
        return `TPC-DS (${storage}) (SF=10)`;
      case 100:
        return `TPC-DS (${storage}) (SF=100)`;
      case 1000:
        return `TPC-DS (${storage}) (SF=1000)`;
      default:
        console.warn("Unknown scale factor:", scaleFactor);
        return null;
    }
  },

  normalizeSeriesName(name, seriesName) {
    let normalizedName = seriesName;
    let normalizedQuery = name;

    if (
      seriesName.endsWith(" throughput") ||
      seriesName.endsWith("throughput")
    ) {
      const suffix = seriesName.endsWith(" throughput")
        ? " throughput"
        : "throughput";
      normalizedName = seriesName.slice(0, seriesName.length - suffix.length);
      normalizedQuery = name.replace("time", "throughput");
    }

    return { name: normalizedQuery, seriesName: normalizedName };
  },

  formatQueryName(query) {
    let prettyQ = query.replace(/_/g, " ").toUpperCase();
    prettyQ = QUERY_NAME_MAP[prettyQ] || prettyQ;
    prettyQ = prettyQ.replace(/^TPCH\s/, "TPC-H ");
    prettyQ = prettyQ.replace(/^TPCDS\s/, "TPC-DS ");
    return prettyQ;
  },

  convertValue(value, unit) {
    const isNanos = unit === "ns/iter" || unit === "ns";
    const isBytes = unit === "bytes";
    const isThroughput = unit === "bytes/ns";

    if (isNanos) return value / 1_000_000;
    if (isBytes) return value / 1_048_576;
    if (isThroughput) return (value * 1_000_000_000) / 1_048_576;
    return value;
  },

  getUnit(unit) {
    const isNanos = unit === "ns/iter" || unit === "ns";
    const isBytes = unit === "bytes";
    const isThroughput = unit === "bytes/ns";

    if (isNanos) return "ms/iter";
    if (isBytes) return "MiB";
    if (isThroughput) return "MiB/s";
    return unit;
  },

  async downloadAndGroupData(data, commitMetadata, seriesRenameFn) {
    const commits = this.parseCommits(commitMetadata);
    const groups = this.initializeGroups();
    const uncategorizableNames = new Set();
    const missingCommits = new Set();

    let processed = 0;
    const total = data.length;

    for (const benchmark of data) {
      this.processBenchmark(
        benchmark,
        commitMetadata,
        commits,
        groups,
        seriesRenameFn,
        missingCommits,
        uncategorizableNames
      );

      let msg = getRandomMessage(DATA_PROCESSING_MESSAGES);

      processed++;
      // Send progress updates every 1000 items
      if (processed % 1000 === 0) {
        self.postMessage({
          type: 'progress',
          progress: (processed / total) * 90, // Reserve 10% for serialization
          message: `${msg} ${processed}/${total}`
        });
      }
    }

    this.sortGroups(groups);

    if (missingCommits.size > 0) {
      console.warn(
        "These commits were missing from commits.json so the commit message is missing and the datetime is set to 1970-01-01T00:00:00Z",
        missingCommits
      );
    }
    if (uncategorizableNames.size > 0) {
      console.warn(
        "Could not categorize benchmarks with these names, they will not be shown:",
        uncategorizableNames
      );
    }

    // Convert Maps to serializable objects for transfer using async serialization
    const serializedGroups = await this.serializeGroupsAsync(groups);

    return serializedGroups;
  },

  // Helper to serialize Maps for transfer across worker boundary
  serializeMap(map) {
    if (!map || typeof map.entries !== 'function') {
      return map; // Return as-is if not a Map
    }
    
    const result = [];
    for (const [key, value] of map.entries()) {
      if (value && value.series) {
        // For series data, convert Map to plain object to avoid deep recursion
        const serializedSeries = {};
        if (value.series instanceof Map) {
          for (const [seriesKey, seriesValue] of value.series.entries()) {
            serializedSeries[seriesKey] = seriesValue;
          }
        } else {
          Object.assign(serializedSeries, value.series);
        }
        
        result.push([key, {
          ...value,
          series: serializedSeries
        }]);
      } else {
        result.push([key, value]);
      }
    }
    return result;
  },

  // Async version of serializeMap that processes in chunks
  async serializeMapAsync(map, onProgress = null) {
    if (!map || typeof map.entries !== 'function') {
      return map; // Return as-is if not a Map
    }
    
    const result = [];
    const entries = Array.from(map.entries());
    const chunkSize = 50; // Process 50 entries at a time
    
    for (let i = 0; i < entries.length; i += chunkSize) {
      const chunk = entries.slice(i, i + chunkSize);
      
      for (const [key, value] of chunk) {
        if (value && value.series) {
          // For series data, convert Map to plain object to avoid deep recursion
          const serializedSeries = {};
          if (value.series instanceof Map) {
            for (const [seriesKey, seriesValue] of value.series.entries()) {
              serializedSeries[seriesKey] = seriesValue;
            }
          } else {
            Object.assign(serializedSeries, value.series);
          }
          
          result.push([key, {
            ...value,
            series: serializedSeries
          }]);
        } else {
          result.push([key, value]);
        }
      }
      
      // Yield control and report progress after each chunk
      if (onProgress) {
        const progress = ((i + chunkSize) / entries.length) * 100;
        onProgress(Math.min(progress, 100));
      }
      
      // Yield control to prevent blocking
      if (i + chunkSize < entries.length) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    return result;
  },

  // Async method to serialize all groups with progress reporting
  async serializeGroupsAsync(groups) {
    const groupNames = Object.keys(groups);
    const serializedGroups = [];
    
    // Note: We process all groups to maintain structure, but skip heavy processing for empty ones

    const progress_message = getRandomMessage(NERDY_PROGRESS_MESSAGES);
    
    for (let i = 0; i < groupNames.length; i++) {
      const name = groupNames[i];
      const group = groups[name];
      
      // Skip empty groups but still report progress
      if (!group || group.size === 0) {
        const overallProgress = 90 + ((i + 1) / groupNames.length) * 10;
        self.postMessage({
          type: 'progress',
          progress: Math.min(overallProgress, 100),
          message: `${getRandomMessage(EMPTY_GROUP_MESSAGES)} Skipping: ${name} (${i + 1}/${groupNames.length})`
        });
        
        // Still add empty group to maintain structure
        serializedGroups.push({
          name,
          dataSet: []
        });
        
        // Brief yield
        await new Promise(resolve => setTimeout(resolve, 0));
        continue;
      }
      
      // Report progress for this group serialization
      const groupProgress = (progress) => {
        const overallProgress = 90 + ((i + (progress / 100)) / groupNames.length) * 10;
        self.postMessage({
          type: 'progress',
          progress: Math.min(overallProgress, 100),
          message: `${progress_message} ${name} (${i + 1}/${groupNames.length})`
        });
      };
      
      const serializedMap = await this.serializeMapAsync(group, groupProgress);
      serializedGroups.push({
        name,
        dataSet: serializedMap
      });
      
      // Brief yield between groups
      if (i < groupNames.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    return serializedGroups;
  },

  initializeGroups() {
    const groups = {};
    BENCHMARK_GROUPS.forEach((name) => {
      groups[name] = new Map();
    });
    return groups;
  },

  processBenchmark(
    benchmark,
    commitMetadata,
    commits,
    groups,
    seriesRenameFn,
    missingCommits,
    uncategorizableNames
  ) {
    // Ensure commit metadata
    if (!benchmark.commit) {
      benchmark.commit = commitMetadata[benchmark.commit_id];
      if (!benchmark.commit) {
        missingCommits.add(benchmark.commit_id);
        benchmark.commit = commitMetadata[benchmark.commit_id] =
          this.createMissingCommit(benchmark.commit_id);
      }
    }

    // Determine group
    const groupId = this.determineGroupId(benchmark);
    if (!groupId) {
      uncategorizableNames.add(benchmark.name);
      return;
    }

    const group = groups[groupId];
    if (!group) {
      console.warn("Cannot find group element in group:", groupId);
      return;
    }

    // Process benchmark data
    let [query, seriesName] = benchmark.name.split("/");
    const normalized = this.normalizeSeriesName(query, seriesName);
    query = normalized.name;
    seriesName = normalized.seriesName;

    // Apply series renaming
    seriesName = this.applySeriesRenaming(
      seriesName,
      groupId,
      seriesRenameFn
    );

    // Format query name
    const prettyQ = this.formatQueryName(query);
    if (prettyQ.includes("PARQUET-UNC")) return;

    // Set units
    let unit = benchmark.unit;
    if (!unit && benchmark.name.startsWith("vortex size/")) {
      unit = "bytes";
    } else if (
      !unit &&
      (benchmark.name.startsWith("vortex:raw size/") ||
        benchmark.name.startsWith("vortex:parquet-zstd size/"))
    ) {
      unit = "ratio";
    }

    // Calculate sort position
    const sortPosition =
      query.slice(0, 4) === "tpch" || query.slice(0, 5) === "tpcds"
        ? parseInt(prettyQ.split(" ")[1].substring(1), 10)
        : 0;

    // Add to group
    this.addToGroup(
      group,
      prettyQ,
      seriesName,
      benchmark,
      unit,
      sortPosition,
      commits
    );
  },

  applySeriesRenaming(seriesName, groupId, seriesRenameFn) {
    if (!seriesRenameFn) return seriesName;

    const renamer = seriesRenameFn.find(([name]) => name === groupId);
    if (renamer?.[1]?.renamedDatasets) {
      const renameDict = renamer[1].renamedDatasets;
      return renameDict[seriesName] || seriesName;
    }
    return seriesName;
  },

  addToGroup(
    group,
    queryName,
    seriesName,
    benchmark,
    unit,
    sortPosition,
    commits
  ) {
    let arr = group.get(queryName);
    if (!arr) {
      group.set(queryName, {
        sort_position: sortPosition,
        commits,
        unit: this.getUnit(unit),
        series: new Map(),
      });
      arr = group.get(queryName);
    }

    let series = arr.series.get(seriesName);
    if (!series) {
      arr.series.set(seriesName, new Array(commits.length).fill(null));
      series = arr.series.get(seriesName);
    }

    const convertedValue = this.convertValue(benchmark.value, unit);
    const sortedIndex = benchmark.commit.sortedIndex;
    
    
    series[sortedIndex] = {
      range: "this was the range",
      value: convertedValue,
    };
  },

  sortGroups(groups) {
    const sortByPositionThenName = (a, b) => {
      const positionCompare = a[1].sort_position - b[1].sort_position;
      return positionCompare !== 0
        ? positionCompare
        : a[0].localeCompare(b[0]);
    };

    Object.entries(groups).forEach(([name, charts]) => {
      groups[name] = new Map(
        [...charts.entries()].sort(sortByPositionThenName)
      );
    });
  },
};

// JSONL parser for worker context
function parseJsonl(jsonl) {
  return jsonl
    .split("\n")
    .filter((line) => line.trim().length !== 0)
    .map((line) => JSON.parse(line));
}

// Worker message handler
self.addEventListener('message', async function(e) {
  const { type, data } = e.data;

  try {
    switch (type) {
      case 'parseData':
        self.postMessage({
          type: 'progress',
          progress: 10,
          message: `${getRandomMessage(PARSING_MESSAGES)} Benchmark data...`
        });

        const benchmarkData = parseJsonl(data.benchmarkData);
        
        self.postMessage({
          type: 'progress',
          progress: 30,
          message: `${getRandomMessage(PARSING_MESSAGES)} Commit data...`
        });

        const commitsData = parseJsonl(data.commitsData);

        self.postMessage({
          type: 'progress',
          progress: 50,
          message: `${getRandomMessage(DATA_PROCESSING_MESSAGES).replace('...', ' and grouping data...')}`
        });

        const commits = {};
        commitsData.forEach((commit) => {
          commits[commit.id] = commit;
        });

        const result = await dataProcessor.downloadAndGroupData(
          benchmarkData,
          commits,
          data.keptGroups
        );

        self.postMessage({
          type: 'dataProcessed',
          result: result,
          progress: 100,
          message: 'Data processing complete!'
        });
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: error.message,
      stack: error.stack
    });
  }
});