"use strict";

import { shared } from './data-shared.js';
import { BENCHMARK_GROUPS } from './config.js';

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

// Data processor implementation for worker context
const dataProcessor = {
  parseCommits: shared.parseCommits,

  createMissingCommit: shared.createMissingCommit,

  determineGroupId: shared.determineGroupId,

  getTpchGroupId: shared.getTpchGroupId,

  getTpcdsGroupId: shared.getTpcdsGroupId,

  normalizeSeriesName: shared.normalizeSeriesName,

  formatQueryName: shared.formatQueryName,

  convertValue: shared.convertValue,

  getUnit: shared.getUnit,

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

  initializeGroups: shared.initializeGroups,

  processBenchmark: shared.processBenchmark,

  applySeriesRenaming: shared.applySeriesRenaming,

  addToGroup: shared.addToGroup,

  sortGroups: shared.sortGroups,
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

        // benchmarkData might already be parsed as an array
        const benchmarkData = Array.isArray(data.benchmarkData)
          ? data.benchmarkData
          : parseJsonl(data.benchmarkData);

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
