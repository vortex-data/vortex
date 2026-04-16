import { GROUPS, QUERY_GROUP_EXCLUSIONS } from "./constants.js";
import { queryRows } from "./db.js";
import { firstLine } from "./utils.js";

const COMMITS_QUERY = `
  select
    commit_idx,
    id,
    message,
    timestamp,
    author,
    url
  from active_commits
  order by commit_idx
`;

const CHART_ROWS_QUERY = `
  select
    cd.group_name,
    cd.chart_name,
    cd.unit,
    cd.sort_position,
    list(cslv.series_name order by cslv.series_name) as series_names,
    json_group_object(cslv.series_name, cslv.latest_value) as latest_values
  from chart_defs cd
  join chart_series_latest_values cslv
    on cslv.group_name = cd.group_name
   and cslv.chart_name = cd.chart_name
  group by 1, 2, 3, 4
  order by 1, 4, 2
`;

const RANDOM_ACCESS_SUMMARY_QUERY = `
  with selected_chart as (
    select chart_name
    from chart_defs
    where group_name = 'Random Access'
    order by sort_position, chart_name
    limit 1
  )
  select
    series_name as name,
    value as time,
    value / min(value) over () as ratio
  from chart_latest_values
  where group_name = 'Random Access'
    and chart_name = (select chart_name from selected_chart)
  order by value, series_name
`;

const COMPRESSION_SUMMARY_QUERY = `
  with anchor_chart as (
    select chart_name
    from chart_defs
    where group_name = 'Compression'
      and chart_name in (
        'VORTEX:PARQUET ZSTD RATIO COMPRESS TIME',
        'VORTEX:PARQUET ZSTD RATIO DECOMPRESS TIME'
      )
    order by case
      when chart_name = 'VORTEX:PARQUET ZSTD RATIO COMPRESS TIME' then 0
      else 1
    end
    limit 1
  ),
  anchor_idx as (
    select latest_commit_idx as commit_idx
    from chart_latest_idx
    where group_name = 'Compression'
      and chart_name = (select chart_name from anchor_chart)
  )
  select
    (
      select geomean(1.0 / value)
      from benchmark_points_active
      where group_name = 'Compression'
        and chart_name = 'VORTEX:PARQUET ZSTD RATIO COMPRESS TIME'
        and commit_idx = (select commit_idx from anchor_idx)
        and lower(series_name) not like '%wide table%'
        and value > 0
    ) as compress_ratio,
    (
      select geomean(1.0 / value)
      from benchmark_points_active
      where group_name = 'Compression'
        and chart_name = 'VORTEX:PARQUET ZSTD RATIO DECOMPRESS TIME'
        and commit_idx = (select commit_idx from anchor_idx)
        and lower(series_name) not like '%wide table%'
        and value > 0
    ) as decompress_ratio,
    (
      select count(*)
      from benchmark_points_active
      where group_name = 'Compression'
        and chart_name = 'VORTEX:PARQUET ZSTD RATIO COMPRESS TIME'
        and commit_idx = (select commit_idx from anchor_idx)
        and lower(series_name) not like '%wide table%'
        and value > 0
    ) as dataset_count
`;

const COMPRESSION_SIZE_SUMMARY_QUERY = `
  with anchor_idx as (
    select latest_commit_idx as commit_idx
    from chart_latest_idx
    where group_name = 'Compression Size'
      and chart_name = 'VORTEX:PARQUET ZSTD SIZE'
  )
  select
    min(value) as min_ratio,
    geomean(value) as mean_ratio,
    max(value) as max_ratio,
    count(*) as dataset_count
  from benchmark_points_active
  where group_name = 'Compression Size'
    and chart_name = 'VORTEX:PARQUET ZSTD SIZE'
    and commit_idx = (select commit_idx from anchor_idx)
    and lower(series_name) not like '%wide table%'
    and value > 0
`;

const QUERY_SUMMARY_QUERY = `
  with per_series as (
    select *
    from chart_series_latest_values
    where group_name not in (
      'Random Access',
      'Compression',
      'Compression Size'
    )
  ),
  chart_bases as (
    select
      group_name,
      chart_name,
      min(latest_value) as base_value
    from per_series
    group by 1, 2
  ),
  series_totals as (
    select
      group_name,
      series_name,
      sum(latest_value) as total_runtime,
      max(latest_value) as max_runtime
    from per_series
    group by 1, 2
  ),
  group_series as (
    select distinct group_name, series_name
    from per_series
  ),
  ratios as (
    select
      gs.group_name,
      gs.series_name,
      ((10.0 + coalesce(ps.latest_value, greatest(300000.0, st.max_runtime) * 2.0))
        / (10.0 + cb.base_value)) as ratio
    from group_series gs
    join series_totals st
      on st.group_name = gs.group_name
     and st.series_name = gs.series_name
    join chart_defs cd
      on cd.group_name = gs.group_name
    join chart_bases cb
      on cb.group_name = cd.group_name
     and cb.chart_name = cd.chart_name
    left join per_series ps
      on ps.group_name = cd.group_name
     and ps.chart_name = cd.chart_name
     and ps.series_name = gs.series_name
  )
  select
    gs.group_name,
    gs.series_name as name,
    geomean(r.ratio) as score,
    st.total_runtime
  from group_series gs
  join ratios r
    on r.group_name = gs.group_name
   and r.series_name = gs.series_name
  join series_totals st
    on st.group_name = gs.group_name
   and st.series_name = gs.series_name
  group by 1, 2, 4
  order by 1, 3, 4, 2
`;

const DIAGNOSTICS_COUNTS_QUERY = `
  select
    (select count(*) from raw_benchmarks) as benchmark_count,
    (
      select count(*)
      from classified_benchmarks cb
      left join commit_dim cd
        on cd.id = cb.resolved_commit_id
      where cb.resolved_commit_id is null
         or cd.id is null
    ) as missing_commits
`;

const UNCATEGORIZED_QUERY = `
  select distinct split_part(name, '/', 1) as prefix
  from classified_benchmarks
  where group_name is null
  order by prefix
  limit 20
`;

const GROUP_COUNTS_QUERY = `
  select
    group_name,
    count(*) as chart_count
  from chart_defs
  group by 1
  order by 1
`;

function buildEmptyGroups() {
  return Object.fromEntries(
    GROUPS.map((groupName) => [
      groupName,
      {
        charts: [],
        totalCharts: 0,
        hasData: false,
        summary: null,
      },
    ]),
  );
}

function buildChartIndex(groupMap) {
  const chartIndex = new Map();
  for (const [groupName, groupData] of Object.entries(groupMap)) {
    for (const chart of groupData.charts) {
      chartIndex.set(`${groupName}\u0000${chart.name}`, chart);
    }
  }
  return chartIndex;
}

function appendChartRows(groups, chartRows, totalCommits) {
  for (const row of chartRows) {
    const group = groups[row.group_name];
    if (!group) continue;

    group.charts.push({
      name: row.chart_name,
      unit: row.unit,
      series: row.series_names || [],
      sortPosition: row.sort_position,
      totalPoints: totalCommits,
      latestValues: row.latest_values || {},
    });
  }
}

function applyRandomAccessSummary(groups, rows) {
  if (!rows.length) return;

  groups["Random Access"].summary = {
    type: "randomAccess",
    title: "Random Access Performance",
    rankings: rows.map((row) => ({
      name: row.name,
      time: row.time,
      ratio: row.ratio,
    })),
    explanation: "Random access time | Ratio to fastest (lower is better)",
  };
}

function applyCompressionSummary(groups, row) {
  if (!row?.compress_ratio && !row?.decompress_ratio) return;

  groups.Compression.summary = {
    type: "compression",
    title: "Compression Throughput vs Parquet",
    compressRatio: row.compress_ratio,
    decompressRatio: row.decompress_ratio,
    datasetCount: row.dataset_count,
    explanation: "Inverse geomean of Vortex/Parquet ratios (higher is better)",
  };
}

function applyCompressionSizeSummary(groups, row) {
  if (!row?.mean_ratio) return;

  groups["Compression Size"].summary = {
    type: "compressionSize",
    title: "Compression Size Summary",
    minRatio: row.min_ratio,
    meanRatio: row.mean_ratio,
    maxRatio: row.max_ratio,
    datasetCount: row.dataset_count,
    explanation: "Geomean of Vortex/Parquet size ratios (lower is better)",
  };
}

function applyQuerySummary(groups, rows) {
  for (const row of rows) {
    if (QUERY_GROUP_EXCLUSIONS.has(row.group_name)) continue;

    const group = groups[row.group_name];
    if (!group) continue;

    if (!group.summary) {
      group.summary = {
        type: "queryBenchmark",
        title: "Performance Summary",
        rankings: [],
        explanation: "Geomean of query time ratio to fastest (lower is better)",
      };
    }

    group.summary.rankings.push({
      name: row.name,
      score: row.score,
      totalRuntime: row.total_runtime,
    });
  }
}

function finalizeGroups(groups) {
  for (const group of Object.values(groups)) {
    group.totalCharts = group.charts.length;
    group.hasData = group.charts.length > 0;
  }
}

export async function buildMetadata(connection, lastUpdated) {
  const commits = await queryRows(connection, COMMITS_QUERY);
  const chartRows = await queryRows(connection, CHART_ROWS_QUERY);
  const randomAccessRows = await queryRows(connection, RANDOM_ACCESS_SUMMARY_QUERY);
  const compressionRows = await queryRows(connection, COMPRESSION_SUMMARY_QUERY);
  const compressionSizeRows = await queryRows(connection, COMPRESSION_SIZE_SUMMARY_QUERY);
  const querySummaryRows = await queryRows(connection, QUERY_SUMMARY_QUERY);

  const groups = buildEmptyGroups();
  appendChartRows(groups, chartRows, commits.length);
  applyRandomAccessSummary(groups, randomAccessRows);
  applyCompressionSummary(groups, compressionRows[0]);
  applyCompressionSizeSummary(groups, compressionSizeRows[0]);
  applyQuerySummary(groups, querySummaryRows);
  finalizeGroups(groups);

  return {
    commits,
    metadata: {
      groups,
      totalCommits: commits.length,
      commits: commits.map(({ id, message, timestamp, author }) => ({
        id,
        message: firstLine(message),
        timestamp,
        author,
      })),
      lastUpdated,
    },
    chartIndex: buildChartIndex(groups),
  };
}

export async function collectDiagnostics(connection) {
  const [counts] = await queryRows(connection, DIAGNOSTICS_COUNTS_QUERY);
  const uncategorized = await queryRows(connection, UNCATEGORIZED_QUERY);
  const groupCounts = await queryRows(connection, GROUP_COUNTS_QUERY);

  return {
    benchmarkCount: counts?.benchmark_count ?? 0,
    missingCommits: counts?.missing_commits ?? 0,
    uncategorized: uncategorized.map((row) => row.prefix),
    groupCounts,
  };
}
