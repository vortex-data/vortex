import { BENCHMARK_GROUPS, QUERY_NAME_MAP } from './config.js';

export const shared = {
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
    if (dataset?.statpopgen) return "Statistical and Population Genetics";
    if (name.startsWith("random-access/")) return "Random Access";
    if (name.includes("compress time/")) return "Compression";
    if (name.includes("size/")) return "Compression Size";
    // Handle ratio patterns for compression throughput
    if (
      name.includes("vortex:parquet-zstd ratio") ||
      name.includes("vortex:lance ratio")
    ) {
      return "Compression";
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

  normalizeSeriesName(name, seriesName, groupId) {
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

    // Normalize engine names to lowercase for all benchmarks except Compression ones
    // Keep original capitalization for "Compression" and "Compression Size" benchmarks
    if (groupId !== "Compression" && groupId !== "Compression Size") {
      normalizedName = normalizedName
        .replace(/^DataFusion:/i, "datafusion:")
        .replace(/^DuckDB:/i, "duckdb:")
        .replace(/^Vortex:/i, "vortex:")
        .replace(/^Arrow:/i, "arrow:");
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
    const normalized = this.normalizeSeriesName(query, seriesName, groupId);
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
        benchmark.name.startsWith("vortex:parquet-zstd size/") ||
        benchmark.name.startsWith("vortex:lance size/"))
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

    series[benchmark.commit.sortedIndex] = {
      range: "this was the range",
      value: this.convertValue(benchmark.value, unit),
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
