import { DuckDBInstance } from "@duckdb/node-api";
import { LTTB } from "downsample";
import fs from "fs";
import fsp from "fs/promises";
import os from "os";
import path from "path";
import { Readable } from "stream";
import { pipeline } from "stream/promises";
import { fileURLToPath } from "url";
import { ENGINE_RENAMES, FAN_OUT_GROUPS, QUERY_SUITES } from "./src/config.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MAX_POINTS = 200;
const GROUPS = [
  "Random Access",
  "Compression",
  "Compression Size",
  ...QUERY_SUITES.filter((suite) => !suite.skip && !suite.fanOut).map(
    (suite) => suite.displayName,
  ),
  ...FAN_OUT_GROUPS,
];

const QUERY_GROUP_EXCLUSIONS = new Set([
  "Random Access",
  "Compression",
  "Compression Size",
]);
const DEFAULT_CACHE_DIR_NAME = "vortex-benchmarks-website-cache";

function sqlStringLiteral(value) {
  return `'${String(value).replace(/'/g, "''")}'`;
}

function sqlValue(value) {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "NULL";
  return sqlStringLiteral(value);
}

function sqlValues(rows) {
  return rows
    .map((row) => `(${row.map((value) => sqlValue(value)).join(", ")})`)
    .join(",\n");
}

function firstLine(message) {
  return String(message || "").split("\n")[0] || "";
}

function normalizeValue(value) {
  if (typeof value === "bigint") {
    const asNumber = Number(value);
    return Number.isSafeInteger(asNumber) ? asNumber : value.toString();
  }
  if (Array.isArray(value)) return value.map(normalizeValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, normalizeValue(child)]),
    );
  }
  return value;
}

async function queryRows(connection, sql, values) {
  const result = await connection.run(sql, values);
  const rows = await result.getRowObjectsJS();
  return rows.map((row) => normalizeValue(row));
}

async function withConnection(instance, fn) {
  const connection = await instance.connect();
  try {
    return await fn(connection);
  } finally {
    connection.closeSync();
  }
}

function cachePaths(cacheDir) {
  return {
    dataPath: path.join(cacheDir, "data.json.gz"),
    commitsPath: path.join(cacheDir, "commits.json"),
    manifestPath: path.join(cacheDir, "manifest.json"),
  };
}

async function pathExists(filePath) {
  try {
    await fsp.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonFile(filePath) {
  try {
    const raw = await fsp.readFile(filePath, "utf8");
    return JSON.parse(raw);
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

async function writeJsonFileAtomic(filePath, value) {
  const tempPath = `${filePath}.tmp-${process.pid}-${Date.now()}`;
  await fsp.writeFile(tempPath, JSON.stringify(value, null, 2));
  await fsp.rename(tempPath, filePath);
}

async function downloadToFile(url, destination, metadata = {}) {
  const headers = {};
  if (metadata.etag) headers["If-None-Match"] = metadata.etag;
  if (metadata.lastModified) headers["If-Modified-Since"] = metadata.lastModified;

  const response = await fetch(url, { headers });
  if (response.status === 304) {
    return {
      changed: false,
      metadata: {
        ...metadata,
        checkedAt: new Date().toISOString(),
      },
    };
  }
  if (!response.ok) {
    throw new Error(`Fetch failed: ${url} ${response.status}`);
  }
  if (!response.body) {
    throw new Error(`Fetch failed: ${url} empty body`);
  }

  const tempPath = `${destination}.tmp-${process.pid}-${Date.now()}`;
  try {
    await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(tempPath));
    await fsp.rename(tempPath, destination);
  } finally {
    if (await pathExists(tempPath)) {
      await fsp.unlink(tempPath);
    }
  }

  return {
    changed: true,
    metadata: {
      url,
      etag: response.headers.get("etag"),
      lastModified: response.headers.get("last-modified"),
      fetchedAt: new Date().toISOString(),
      checkedAt: new Date().toISOString(),
    },
  };
}

async function prepareInputFiles({
  dataUrl,
  commitsUrl,
  useLocalData,
  cacheDir,
  preferCached = false,
  forceRemoteCheck = false,
}) {
  if (useLocalData) {
    return {
      dataPath: path.join(__dirname, "sample/data.json"),
      commitsPath: path.join(__dirname, "sample/commits.json"),
      changed: true,
      source: "local",
      deferRemoteCheck: false,
    };
  }

  const resolvedCacheDir =
    cacheDir || path.join(os.tmpdir(), DEFAULT_CACHE_DIR_NAME);
  await fsp.mkdir(resolvedCacheDir, { recursive: true });

  const { dataPath, commitsPath, manifestPath } = cachePaths(resolvedCacheDir);
  const manifest = (await readJsonFile(manifestPath)) || {};
  const [hasDataFile, hasCommitsFile] = await Promise.all([
    pathExists(dataPath),
    pathExists(commitsPath),
  ]);
  const hasCachedFiles = hasDataFile && hasCommitsFile;

  if (preferCached && hasCachedFiles && !forceRemoteCheck) {
    return {
      dataPath,
      commitsPath,
      changed: true,
      source: "cache",
      deferRemoteCheck: true,
    };
  }

  try {
    const [dataResult, commitsResult] = await Promise.all([
      downloadToFile(dataUrl, dataPath, manifest.data || {}),
      downloadToFile(commitsUrl, commitsPath, manifest.commits || {}),
    ]);

    await writeJsonFileAtomic(manifestPath, {
      version: 1,
      dataUrl,
      commitsUrl,
      data: dataResult.metadata,
      commits: commitsResult.metadata,
      updatedAt: new Date().toISOString(),
    });

    return {
      dataPath,
      commitsPath,
      changed: !hasCachedFiles || dataResult.changed || commitsResult.changed,
      source:
        !hasCachedFiles || dataResult.changed || commitsResult.changed
          ? "remote"
          : "cache",
      deferRemoteCheck: false,
    };
  } catch (error) {
    if (hasCachedFiles) {
      console.warn(
        `Falling back to cached benchmark files after refresh failed: ${error.message}`,
      );
      return {
        dataPath,
        commitsPath,
        changed: false,
        source: "stale-cache",
        deferRemoteCheck: false,
      };
    }
    throw error;
  };
}

function renameSeriesInSql(sourceExpression) {
  return `coalesce(
    (select dst from engine_renames er where er.src = lower(${sourceExpression}) limit 1),
    ${sourceExpression}
  )`;
}

function buildBootstrapSql(dataPath, commitsPath) {
  const suiteRows = QUERY_SUITES.map((suite) => [
    suite.prefix,
    suite.displayName || null,
    suite.queryPrefix || null,
    suite.datasetKey || null,
    Boolean(suite.fanOut),
    Boolean(suite.skip),
  ]);
  const groupRows = GROUPS.map((groupName) => [groupName]);
  const renameRows = Object.entries(ENGINE_RENAMES);

  return `
create or replace table query_suites as
select * from (
  values
  ${sqlValues(suiteRows)}
) as suites(prefix, display_name, query_prefix, dataset_key, fan_out, skip);

create or replace table valid_groups as
select * from (
  values
  ${sqlValues(groupRows)}
) as groups(group_name);

create or replace table engine_renames as
select * from (
  values
  ${sqlValues(renameRows)}
) as renames(src, dst);

create or replace view raw_commits as
select *
from read_json(
  ${sqlStringLiteral(commitsPath)},
  format = 'newline_delimited',
  compression = 'auto_detect',
  columns = {
    id: 'VARCHAR',
    message: 'VARCHAR',
    timestamp: 'VARCHAR',
    author: 'JSON',
    url: 'VARCHAR'
  }
);

create or replace table commit_dim as
select
  row_number() over (order by commit_ts nulls last, id) - 1 as commit_idx,
  id,
  message,
  timestamp as timestamp_text,
  commit_ts,
  author,
  url
from (
  select
    id,
    message,
    timestamp,
    try_cast(timestamp as timestamptz) as commit_ts,
    coalesce(json_extract_string(author, '$.name'), 'Unknown') as author,
    url
  from raw_commits
);

create or replace view raw_benchmarks as
select
  row_number() over () as benchmark_row,
  *
from read_json(
  ${sqlStringLiteral(dataPath)},
  format = 'newline_delimited',
  compression = 'auto_detect',
  columns = {
    name: 'VARCHAR',
    unit: 'VARCHAR',
    value: 'DOUBLE',
    storage: 'VARCHAR',
    dataset: 'JSON',
    commit: 'JSON',
    commit_id: 'VARCHAR'
  }
);

create or replace table benchmarks_base as
select
  benchmark_row,
  name,
  lower(name) as name_lower,
  split_part(name, '/', 1) as part1,
  lower(split_part(name, '/', 1)) as part1_lower,
  coalesce(nullif(split_part(name, '/', 2), ''), 'default') as part2,
  coalesce(nullif(split_part(name, '/', 3), ''), 'default') as part3,
  coalesce(nullif(split_part(name, '/', 4), ''), 'default') as part4,
  array_length(string_split(name, '/')) as part_count,
  unit,
  value as raw_value,
  storage,
  dataset as dataset_json,
  commit as commit_json,
  commit_id
from raw_benchmarks;

create or replace table matched_suites as
select
  b.benchmark_row,
  s.prefix,
  s.display_name,
  s.query_prefix,
  s.dataset_key,
  s.fan_out
from benchmarks_base b
join query_suites s
  on (b.name_lower like s.prefix || '_q%' or b.name_lower like s.prefix || '/%')
where not s.skip
qualify row_number() over (partition by b.benchmark_row order by length(s.prefix) desc) = 1;

create or replace table classified_benchmarks as
with base as (
  select
    b.*,
    ms.prefix as suite_prefix,
    ms.display_name as suite_display_name,
    ms.query_prefix as suite_query_prefix,
    ms.dataset_key as suite_dataset_key,
    coalesce(ms.fan_out, false) as suite_fan_out,
    coalesce(json_extract_string(b.commit_json, '$.id'), b.commit_id) as resolved_commit_id,
    case
      when ms.dataset_key = 'tpch' then json_extract_string(b.dataset_json, '$.tpch.scale_factor')
      when ms.dataset_key = 'tpcds' then json_extract_string(b.dataset_json, '$.tpcds.scale_factor')
      else null
    end as suite_scale_factor
  from benchmarks_base b
  left join matched_suites ms using (benchmark_row)
),
named as (
  select
    *,
    case
      when starts_with(name_lower, 'random-access/') or starts_with(name_lower, 'random access/') then 'Random Access'
      when starts_with(name_lower, 'vortex size/')
        or starts_with(name_lower, 'vortex-file-compressed size/')
        or starts_with(name_lower, 'parquet size/')
        or starts_with(name_lower, 'lance size/')
        or contains(name_lower, ':raw size/')
        or contains(name_lower, ':parquet-zstd size/')
        or contains(name_lower, ':lance size/')
        then 'Compression Size'
      when starts_with(name_lower, 'compress time/')
        or starts_with(name_lower, 'decompress time/')
        or starts_with(name_lower, 'parquet_rs-zstd compress')
        or starts_with(name_lower, 'parquet_rs-zstd decompress')
        or starts_with(name_lower, 'lance compress')
        or starts_with(name_lower, 'lance decompress')
        or starts_with(name_lower, 'vortex:lance ratio')
        or starts_with(name_lower, 'vortex:parquet-zstd ratio')
        or starts_with(name_lower, 'vortex:raw ratio')
        then 'Compression'
      when suite_prefix is not null and not suite_fan_out then suite_display_name
      when suite_prefix is not null and suite_fan_out then
        suite_display_name
        || ' ('
        || case when upper(coalesce(storage, '')) = 'S3' then 'S3' else 'NVMe' end
        || ') (SF='
        || cast(cast(round(coalesce(try_cast(suite_scale_factor as double), 1.0)) as bigint) as varchar)
        || ')'
      else null
    end as group_name,
    case
      when (starts_with(name_lower, 'random-access/') or starts_with(name_lower, 'random access/')) and part_count = 4
        then upper(replace(replace(part2 || '/' || part3, '_', ' '), '-', ' '))
      when (starts_with(name_lower, 'random-access/') or starts_with(name_lower, 'random access/')) and part_count = 2
        then 'RANDOM ACCESS'
      when suite_prefix is not null
        and regexp_extract(part1_lower, '^' || suite_prefix || '[_ ]?q([0-9]+)', 1) <> ''
        then suite_query_prefix || ' Q' || cast(cast(regexp_extract(part1_lower, '^' || suite_prefix || '[_ ]?q([0-9]+)', 1) as integer) as varchar)
      else upper(replace(replace(part1, '_', ' '), '-', ' '))
    end as raw_chart_name,
    case
      when (starts_with(name_lower, 'random-access/') or starts_with(name_lower, 'random access/')) and part_count = 4
        then part4
      else part2
    end as raw_series_name,
    case
      when unit is not null then unit
      when contains(name_lower, ' size/') then 'bytes'
      when contains(name_lower, ' ratio ') then 'ratio'
      else 'ns'
    end as raw_unit,
    coalesce(try_cast(nullif(regexp_extract(part1_lower, 'q([0-9]+)$', 1), '') as integer), 0) as sort_position
  from base
),
renamed as (
  select
    n.*,
    ${renameSeriesInSql("n.raw_series_name")} as series_name
  from named n
)
select
  benchmark_row,
  name,
  resolved_commit_id,
  group_name,
  case
    when group_name = 'Compression Size' and raw_chart_name = 'VORTEX FILE COMPRESSED SIZE' then 'VORTEX SIZE'
    else raw_chart_name
  end as chart_name,
  series_name,
  sort_position,
  case
    when raw_unit = 'ns' then 'ms/iter'
    when raw_unit = 'bytes' then 'MiB'
    else raw_unit
  end as unit,
  case
    when raw_unit = 'ns' then raw_value / 1000000.0
    when raw_unit = 'bytes' then raw_value / 1048576.0
    else raw_value
  end as value
from renamed;

create or replace table benchmark_points as
select
  cb.group_name,
  cb.chart_name,
  cb.series_name,
  cb.sort_position,
  cb.unit,
  cb.value,
  cd.commit_idx
from classified_benchmarks cb
join valid_groups vg on vg.group_name = cb.group_name
join commit_dim cd on cd.id = cb.resolved_commit_id
where cb.group_name is not null
  and cb.chart_name not like '%PARQUET-UNC%'
  and cb.name not like '% throughput%'
  and cb.value is not null;

create or replace table active_commits as
with first_active as (
  select coalesce(min(commit_idx), 0) as min_commit_idx
  from benchmark_points
)
select
  cd.commit_idx as original_commit_idx,
  cd.commit_idx - fa.min_commit_idx as commit_idx,
  cd.id,
  cd.message,
  cd.timestamp_text as timestamp,
  cd.author,
  cd.url
from commit_dim cd
cross join first_active fa
where cd.commit_idx >= fa.min_commit_idx
order by cd.commit_idx;

create or replace table benchmark_points_active as
select
  bp.group_name,
  bp.chart_name,
  bp.series_name,
  bp.sort_position,
  bp.unit,
  bp.value,
  ac.commit_idx
from benchmark_points bp
join active_commits ac
  on ac.original_commit_idx = bp.commit_idx;

create or replace table chart_defs as
select
  group_name,
  chart_name,
  min(sort_position) as sort_position,
  min(unit) as unit
from benchmark_points_active
group by 1, 2;

create or replace table chart_latest_idx as
select
  group_name,
  chart_name,
  max(commit_idx) as latest_commit_idx
from benchmark_points_active
group by 1, 2;

create or replace table chart_latest_values as
select
  bpa.group_name,
  bpa.chart_name,
  bpa.series_name,
  bpa.value
from benchmark_points_active bpa
join chart_latest_idx cli
  on cli.group_name = bpa.group_name
 and cli.chart_name = bpa.chart_name
 and cli.latest_commit_idx = bpa.commit_idx;

create or replace table chart_series_latest_values as
select
  group_name,
  chart_name,
  series_name,
  arg_max(value, commit_idx) as latest_value
from benchmark_points_active
group by 1, 2, 3;
`;
}

function lttbIndices(seriesMap, target) {
  const keys = [...seriesMap.keys()];
  if (!keys.length) return [];
  const len = seriesMap.get(keys[0])?.length || 0;
  if (len <= target) return [...Array(len).keys()];

  const avg = Array(len);
  for (let i = 0; i < len; i++) {
    let sum = 0;
    let count = 0;
    for (const arr of seriesMap.values()) {
      const value = arr[i]?.value ?? arr[i];
      if (value != null && !Number.isNaN(value)) {
        sum += value;
        count++;
      }
    }
    avg[i] = [i, count ? sum / count : 0];
  }

  const idx = LTTB(avg, target).map((point) => Math.round(point[0]));
  if (!idx.includes(0)) idx.unshift(0);
  if (!idx.includes(len - 1)) idx.push(len - 1);
  return idx.sort((a, b) => a - b);
}

function downsample(data, factor) {
  const target = Math.ceil(data.commits.length / factor);
  if (target >= data.commits.length) return data;

  const idx = [...new Set(
    lttbIndices(data.series, target).filter(
      (i) => Number.isInteger(i) && i >= 0 && i < data.commits.length,
    ),
  )].sort((a, b) => a - b);
  if (!idx.length) return data;

  const series = new Map();
  for (const [key, values] of data.series) {
    series.set(
      key,
      idx.map((i) => values[i]),
    );
  }

  return {
    ...data,
    commits: idx.map((i) => data.commits[i]),
    series,
  };
}

function downsampleLevel(length) {
  if (length <= MAX_POINTS) return "1x";
  if (length <= MAX_POINTS * 2) return "2x";
  if (length <= MAX_POINTS * 4) return "4x";
  return "8x";
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

async function buildMetadata(connection, lastUpdated) {
  const commits = await queryRows(
    connection,
    `
      select
        commit_idx,
        id,
        message,
        timestamp,
        author,
        url
      from active_commits
      order by commit_idx
    `,
  );

  const chartRows = await queryRows(
    connection,
    `
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
    `,
  );

  const groups = Object.fromEntries(
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

  for (const row of chartRows) {
    const group = groups[row.group_name];
    if (!group) continue;
    group.charts.push({
      name: row.chart_name,
      unit: row.unit,
      series: row.series_names || [],
      sortPosition: row.sort_position,
      totalPoints: commits.length,
      latestValues: row.latest_values || {},
    });
  }

  const randomAccessRows = await queryRows(
    connection,
    `
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
    `,
  );

  if (randomAccessRows.length > 0) {
    groups["Random Access"].summary = {
      type: "randomAccess",
      title: "Random Access Performance",
      rankings: randomAccessRows.map((row) => ({
        name: row.name,
        time: row.time,
        ratio: row.ratio,
      })),
      explanation: "Random access time | Ratio to fastest (lower is better)",
    };
  }

  const compressionRows = await queryRows(
    connection,
    `
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
    `,
  );

  const compressionSummary = compressionRows[0];
  if (compressionSummary?.compress_ratio || compressionSummary?.decompress_ratio) {
    groups.Compression.summary = {
      type: "compression",
      title: "Compression Throughput vs Parquet",
      compressRatio: compressionSummary.compress_ratio,
      decompressRatio: compressionSummary.decompress_ratio,
      datasetCount: compressionSummary.dataset_count,
      explanation: "Inverse geomean of Vortex/Parquet ratios (higher is better)",
    };
  }

  const compressionSizeRows = await queryRows(
    connection,
    `
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
    `,
  );

  const compressionSizeSummary = compressionSizeRows[0];
  if (compressionSizeSummary?.mean_ratio) {
    groups["Compression Size"].summary = {
      type: "compressionSize",
      title: "Compression Size Summary",
      minRatio: compressionSizeSummary.min_ratio,
      meanRatio: compressionSizeSummary.mean_ratio,
      maxRatio: compressionSizeSummary.max_ratio,
      datasetCount: compressionSizeSummary.dataset_count,
      explanation:
        "Geomean of Vortex/Parquet size ratios (lower is better)",
    };
  }

  const querySummaryRows = await queryRows(
    connection,
    `
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
    `,
  );

  for (const row of querySummaryRows) {
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

  for (const group of Object.values(groups)) {
    group.totalCharts = group.charts.length;
    group.hasData = group.charts.length > 0;
  }

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
  const { commits, metadata, chartIndex } = await withConnection(
    instance,
    async (connection) => buildMetadata(connection, lastUpdated),
  );

  const diagnostics = await withConnection(instance, async (connection) => {
    const [counts] = await queryRows(
      connection,
      `
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
      `,
    );
    const uncategorized = await queryRows(
      connection,
      `
        select distinct split_part(name, '/', 1) as prefix
        from classified_benchmarks
        where group_name is null
        order by prefix
        limit 20
      `,
    );
    const groupCounts = await queryRows(
      connection,
      `
        select
          group_name,
          count(*) as chart_count
        from chart_defs
        group by 1
        order by 1
      `,
    );
    return {
      benchmarkCount: counts?.benchmark_count ?? 0,
      missingCommits: counts?.missing_commits ?? 0,
      uncategorized: uncategorized.map((row) => row.prefix),
      groupCounts,
    };
  });

  return {
    commits,
    metadata,
    chartIndex,
    lastUpdated,
    diagnostics,
  };
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
      this.instancePromise = DuckDBInstance.create(":memory:", { threads: "4" })
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
    if (this.state) {
      this.state = null;
    }
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
    const totalCommits = commits.length;
    const commitTimestamp = (commit) =>
      typeof commit.timestamp === "number"
        ? commit.timestamp
        : new Date(commit.timestamp).getTime();

    let startIdx = 0;
    let endIdx = totalCommits - 1;

    if (options.last && !options.start && !options.end && options.startIdx == null && options.endIdx == null) {
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

    startIdx = Math.max(0, Math.min(startIdx, totalCommits - 1));
    endIdx = Math.max(startIdx, Math.min(endIdx, totalCommits - 1));

    const rows = await withConnection(instance, async (connection) =>
      queryRows(
        connection,
        `
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
        `,
        {
          group_name: groupName,
          chart_name: chartName,
          start_idx: startIdx,
          end_idx: endIdx,
        },
      ),
    );

    const requestedCommits = commits
      .slice(startIdx, endIdx + 1)
      .map(({ id, message, timestamp, author, url }) => ({
        id,
        message: firstLine(message),
        timestamp,
        author,
        url,
      }));

    const series = new Map(
      rows.map((row) => [
        row.series_name,
        (row.values || []).map((value) => (value == null ? null : value)),
      ]),
    );

    const selected = {
      commits: requestedCommits,
      series,
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
