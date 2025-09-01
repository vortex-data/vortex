"use strict";

import { shared } from './data-shared.js';
import { BENCHMARK_GROUPS } from './config.js';

// Data processing module
export const dataProcessor = {
  parseCommits: shared.parseCommits,

  createMissingCommit: shared.createMissingCommit,

  determineGroupId: shared.determineGroupId,

  getTpchGroupId: shared.getTpchGroupId,

  getTpcdsGroupId: shared.getTpcdsGroupId,

  normalizeSeriesName: shared.normalizeSeriesName,

  formatQueryName: shared.formatQueryName,

  convertValue: shared.convertValue,

  getUnit: shared.getUnit,

  downloadAndGroupData(data, commitMetadata, seriesRenameFn) {
    const commits = this.parseCommits(commitMetadata);
    const groups = this.initializeGroups();
    const uncategorizableNames = new Set();
    const missingCommits = new Set();

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

    return Object.keys(groups).map((name) => ({
      name,
      dataSet: groups[name],
    }));
  },

  initializeGroups: shared.initializeGroups,

  processBenchmark: shared.processBenchmark,

  applySeriesRenaming: shared.applySeriesRenaming,

  addToGroup: shared.addToGroup,

  sortGroups: shared.sortGroups,
};
