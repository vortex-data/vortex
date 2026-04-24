import { DuckDBInstance } from "@duckdb/node-api";
import { prepareInputFiles } from "./cache.js";
import { queryRows, withConnection } from "./db.js";
import { downsample, downsampleLevel } from "./downsample.js";
import { buildMetadata, collectDiagnostics } from "./metadata.js";
import { buildBootstrapSql } from "./sql.js";
import { firstLine } from "./utils.js";

const CHART_RANGE_QUERY = `
  with requested_commits as (
    select commit_idx
    from active_commits
    where commit_idx between $start_idx and $end_idx
  ),
  requested_series as (
    select distinct series_name
    from benchmark_points_active
    where group_name = $group_name
      and chart_name = $chart_name
  ),
  dense_points as (
    select
      rs.series_name,
      rc.commit_idx,
      bpa.value
    from requested_series rs
    cross join requested_commits rc
    left join benchmark_points_active bpa
      on bpa.group_name = $group_name
     and bpa.chart_name = $chart_name
     and bpa.series_name = rs.series_name
     and bpa.commit_idx = rc.commit_idx
  )
  select
    series_name,
    list(value order by commit_idx) as values
  from dense_points
  group by 1
  order by 1
`;

async function buildStoreState({ instance, inputs }) {
  await withConnection(instance, async (connection) => {
    await connection.run("begin transaction");
    try {
      await connection.run(buildBootstrapSql(inputs.dataPath, inputs.commitsPath));
      await connection.run("commit");
    } catch (error) {
      try {
        await connection.run("rollback");
      } catch {
        // Best effort: keep the previous committed state if the rebuild failed.
      }
      throw error;
    }
  });

  const lastUpdated = new Date().toISOString();
  const [{ commits, metadata, chartIndex }, diagnostics] = await Promise.all([
    buildMetadata(instance, lastUpdated),
    withConnection(instance, collectDiagnostics),
  ]);

  return {
    commits,
    metadata,
    chartIndex,
    lastUpdated,
    diagnostics,
  };
}

function commitTimestamp(commit) {
  return typeof commit.timestamp === "number"
    ? commit.timestamp
    : new Date(commit.timestamp).getTime();
}

function resolveRequestedRange(commits, options) {
  const totalCommits = commits.length;
  let startIdx = 0;
  let endIdx = totalCommits - 1;

  if (
    options.last &&
    !options.start &&
    !options.end &&
    options.startIdx == null &&
    options.endIdx == null
  ) {
    const count = Number.parseInt(options.last, 10);
    if (count > 0 && count < totalCommits) {
      startIdx = totalCommits - count;
    }
  } else if (options.startIdx != null || options.endIdx != null) {
    if (options.startIdx != null) {
      startIdx = Math.max(0, Number.parseInt(options.startIdx, 10));
    }
    if (options.endIdx != null) {
      endIdx = Math.min(totalCommits - 1, Number.parseInt(options.endIdx, 10));
    }
  } else {
    if (options.start) {
      const startTs = Number(options.start);
      const idx = commits.findIndex((commit) => commitTimestamp(commit) >= startTs);
      if (idx !== -1) startIdx = idx;
    }

    if (options.end) {
      const endTs = Number(options.end);
      for (let i = endIdx; i >= 0; i--) {
        if (commitTimestamp(commits[i]) <= endTs) {
          endIdx = i;
          break;
        }
      }
    }
  }

  return {
    startIdx: Math.max(0, Math.min(startIdx, totalCommits - 1)),
    endIdx: Math.max(startIdx, Math.min(endIdx, totalCommits - 1)),
  };
}

function serializeRequestedCommits(commits, startIdx, endIdx) {
  return commits
    .slice(startIdx, endIdx + 1)
    .map(({ id, message, timestamp, author, url }) => ({
      id,
      message: firstLine(message),
      timestamp,
      author,
      url,
    }));
}

function buildSeriesMap(rows) {
  return new Map(
    rows.map((row) => [
      row.series_name,
      (row.values || []).map((value) => (value == null ? null : value)),
    ]),
  );
}

export class BenchmarkStore {
  constructor(options) {
    this.options = options;
    this.state = null;
    this.instance = null;
    this.instancePromise = null;
    this.refreshPromise = null;
    this.remoteCheckTimer = null;
    this.lastRefreshStartedAt = null;
    this.lastRefreshCompletedAt = null;
    this.lastRefreshError = null;
  }

  get metadata() {
    return this.state?.metadata || null;
  }

  get status() {
    let state = "idle";

    if (this.state) {
      if (this.refreshPromise) {
        state = "refreshing";
      } else if (this.lastRefreshError) {
        state = "stale";
      } else {
        state = "ready";
      }
    } else if (this.lastRefreshError) {
      state = "error";
    } else if (this.refreshPromise) {
      state = "loading";
    }

    return {
      state,
      ready: Boolean(this.state),
      refreshing: Boolean(this.refreshPromise),
      hasData: Boolean(this.state?.metadata),
      lastUpdated: this.state?.lastUpdated || null,
      lastRefreshStartedAt: this.lastRefreshStartedAt,
      lastRefreshCompletedAt: this.lastRefreshCompletedAt,
      lastRefreshError: this.lastRefreshError,
    };
  }

  async getInstance() {
    if (this.instance) return this.instance;

    if (!this.instancePromise) {
      this.instancePromise = DuckDBInstance.create(":memory:")
        .then((instance) => {
          this.instance = instance;
          return instance;
        })
        .finally(() => {
          this.instancePromise = null;
        });
    }

    return this.instancePromise;
  }

  scheduleRemoteCheck() {
    if (this.remoteCheckTimer) return;

    this.remoteCheckTimer = setTimeout(() => {
      this.remoteCheckTimer = null;
      this.refresh({ forceRemoteCheck: true }).catch(() => {});
    }, 0);
  }

  async refresh({ forceRemoteCheck = false } = {}) {
    if (this.refreshPromise) return this.refreshPromise;

    this.lastRefreshStartedAt = new Date().toISOString();
    this.lastRefreshError = null;
    this.refreshPromise = (async () => {
      const startedAt = Date.now();
      const hasState = Boolean(this.state);
      const inputs = await prepareInputFiles({
        ...this.options,
        preferCached: !hasState && !forceRemoteCheck,
        forceRemoteCheck,
      });

      if (hasState && !inputs.changed) {
        this.lastRefreshCompletedAt = new Date().toISOString();
        console.log(
          `Refresh skipped in ${Date.now() - startedAt}ms (${inputs.source})`,
        );
        return;
      }

      const instance = await this.getInstance();
      const nextState = await buildStoreState({ instance, inputs });

      console.log(
        `Processed ${nextState.diagnostics.benchmarkCount} benchmarks, ${nextState.commits.length} commits`,
      );

      if (nextState.diagnostics.uncategorized.length > 0) {
        console.log(
          `Uncategorized benchmark prefixes (${nextState.diagnostics.uncategorized.length}):`,
          nextState.diagnostics.uncategorized.join(", "),
        );
      }

      const chartCounts = nextState.diagnostics.groupCounts
        .map((row) => `${row.group_name}: ${row.chart_count}`)
        .filter((entry) => !entry.endsWith(": 0"));
      console.log("Charts per group:", chartCounts.join(", "));

      this.state = nextState;
      this.lastRefreshCompletedAt = nextState.lastUpdated;

      console.log(
        `Refresh done in ${Date.now() - startedAt}ms (${nextState.diagnostics.missingCommits} missing commits, source: ${inputs.source})`,
      );

      if (inputs.deferRemoteCheck) {
        console.log(
          "Serving cached benchmark files for startup; scheduling remote revalidation",
        );
        this.scheduleRemoteCheck();
      }
    })()
      .catch((error) => {
        this.lastRefreshError = error?.message || String(error);
        console.error("Refresh error:", error);
        throw error;
      })
      .finally(() => {
        this.refreshPromise = null;
      });

    return this.refreshPromise;
  }

  async close() {
    if (this.remoteCheckTimer) {
      clearTimeout(this.remoteCheckTimer);
      this.remoteCheckTimer = null;
    }

    this.state = null;

    if (this.instance) {
      this.instance.closeSync();
      this.instance = null;
    }
  }

  async getChartData(groupName, chartName, options = {}) {
    if (!this.state) {
      throw new Error("Loading");
    }

    const chart = this.state.chartIndex.get(`${groupName}\u0000${chartName}`);
    if (!chart) {
      const error = new Error("Chart not found");
      error.statusCode = 404;
      throw error;
    }

    const { commits, metadata } = this.state;
    const instance = this.instance;
    if (!instance) {
      throw new Error("Loading");
    }

    const { startIdx, endIdx } = resolveRequestedRange(commits, options);
    const rows = await withConnection(instance, async (connection) =>
      queryRows(connection, CHART_RANGE_QUERY, {
        group_name: groupName,
        chart_name: chartName,
        start_idx: startIdx,
        end_idx: endIdx,
      }),
    );

    const requestedCommits = serializeRequestedCommits(commits, startIdx, endIdx);
    const selected = {
      commits: requestedCommits,
      series: buildSeriesMap(rows),
    };
    const level = downsampleLevel(requestedCommits.length);
    const sampled =
      level === "1x"
        ? selected
        : downsample(selected, Number.parseInt(level, 10));

    return {
      group: groupName,
      chart: chartName,
      unit: chart.unit,
      downsampleLevel: level,
      originalLength: metadata.totalCommits,
      requestedRange: {
        startIndex: startIdx,
        endIndex: endIdx,
        length: endIdx - startIdx + 1,
      },
      commits: sampled.commits,
      series: Object.fromEntries(sampled.series),
    };
  }
}
