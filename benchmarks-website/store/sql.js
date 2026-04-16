import { ENGINE_RENAMES, QUERY_SUITES } from "../src/config.js";
import { GROUPS } from "./constants.js";

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

function joinStatements(statements) {
  return statements
    .map((statement) => statement.trim().replace(/;?$/, ";"))
    .join("\n\n");
}

function randomAccessCondition(expression = "name_lower") {
  return `(starts_with(${expression}, 'random-access/') or starts_with(${expression}, 'random access/'))`;
}

function renameSeriesInSql(sourceExpression) {
  return `coalesce(
    (select dst from engine_renames er where er.src = lower(${sourceExpression}) limit 1),
    ${sourceExpression}
  )`;
}

function buildQuerySuitesStatement() {
  const rows = QUERY_SUITES.map((suite) => [
    suite.prefix,
    suite.displayName || null,
    suite.queryPrefix || null,
    suite.datasetKey || null,
    Boolean(suite.fanOut),
    Boolean(suite.skip),
  ]);

  return `
create or replace table query_suites as
select * from (
  values
  ${sqlValues(rows)}
) as suites(prefix, display_name, query_prefix, dataset_key, fan_out, skip)
`;
}

function buildValidGroupsStatement() {
  const rows = GROUPS.map((groupName) => [groupName]);

  return `
create or replace table valid_groups as
select * from (
  values
  ${sqlValues(rows)}
) as groups(group_name)
`;
}

function buildEngineRenamesStatement() {
  const rows = Object.entries(ENGINE_RENAMES);

  return `
create or replace table engine_renames as
select * from (
  values
  ${sqlValues(rows)}
) as renames(src, dst)
`;
}

function buildRawCommitsViewStatement(commitsPath) {
  return `
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
)
`;
}

function buildCommitDimStatement() {
  return `
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
)
`;
}

function buildRawBenchmarksViewStatement(dataPath) {
  return `
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
)
`;
}

function buildBenchmarksBaseStatement() {
  return `
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
from raw_benchmarks
`;
}

function buildMatchedSuitesStatement() {
  return `
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
qualify row_number() over (partition by b.benchmark_row order by length(s.prefix) desc) = 1
`;
}

function buildGroupNameExpressionSql() {
  return `
case
  when ${randomAccessCondition()} then 'Random Access'
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
end
`;
}

function buildRawChartNameExpressionSql() {
  return `
case
  when ${randomAccessCondition()} and part_count = 4
    then upper(replace(replace(part2 || '/' || part3, '_', ' '), '-', ' '))
  when ${randomAccessCondition()} and part_count = 2
    then 'RANDOM ACCESS'
  when suite_prefix is not null
    and regexp_extract(part1_lower, '^' || suite_prefix || '[_ ]?q([0-9]+)', 1) <> ''
    then suite_query_prefix || ' Q' || cast(cast(regexp_extract(part1_lower, '^' || suite_prefix || '[_ ]?q([0-9]+)', 1) as integer) as varchar)
  else upper(replace(replace(part1, '_', ' '), '-', ' '))
end
`;
}

function buildRawSeriesNameExpressionSql() {
  return `
case
  when ${randomAccessCondition()} and part_count = 4 then part4
  else part2
end
`;
}

function buildRawUnitExpressionSql() {
  return `
case
  when unit is not null then unit
  when contains(name_lower, ' size/') then 'bytes'
  when contains(name_lower, ' ratio ') then 'ratio'
  else 'ns'
end
`;
}

function buildSortPositionExpressionSql() {
  return `
coalesce(
  try_cast(nullif(regexp_extract(part1_lower, 'q([0-9]+)$', 1), '') as integer),
  0
)
`;
}

function buildChartNameExpressionSql() {
  return `
case
  when group_name = 'Compression Size' and raw_chart_name = 'VORTEX FILE COMPRESSED SIZE' then 'VORTEX SIZE'
  else raw_chart_name
end
`;
}

function buildDisplayUnitExpressionSql() {
  return `
case
  when raw_unit = 'ns' then 'ms/iter'
  when raw_unit = 'bytes' then 'MiB'
  else raw_unit
end
`;
}

function buildDisplayValueExpressionSql() {
  return `
case
  when raw_unit = 'ns' then raw_value / 1000000.0
  when raw_unit = 'bytes' then raw_value / 1048576.0
  else raw_value
end
`;
}

function buildClassifiedBenchmarksStatement() {
  return `
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
    ${buildGroupNameExpressionSql()} as group_name,
    ${buildRawChartNameExpressionSql()} as raw_chart_name,
    ${buildRawSeriesNameExpressionSql()} as raw_series_name,
    ${buildRawUnitExpressionSql()} as raw_unit,
    ${buildSortPositionExpressionSql()} as sort_position
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
  ${buildChartNameExpressionSql()} as chart_name,
  series_name,
  sort_position,
  ${buildDisplayUnitExpressionSql()} as unit,
  ${buildDisplayValueExpressionSql()} as value
from renamed
`;
}

function buildBenchmarkPointsStatement() {
  return `
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
  and cb.value is not null
`;
}

function buildActiveCommitsStatement() {
  return `
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
order by cd.commit_idx
`;
}

function buildBenchmarkPointsActiveStatement() {
  return `
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
  on ac.original_commit_idx = bp.commit_idx
`;
}

function buildChartDefinitionsStatement() {
  return `
create or replace table chart_defs as
select
  group_name,
  chart_name,
  min(sort_position) as sort_position,
  min(unit) as unit
from benchmark_points_active
group by 1, 2
`;
}

function buildChartLatestIndexStatement() {
  return `
create or replace table chart_latest_idx as
select
  group_name,
  chart_name,
  max(commit_idx) as latest_commit_idx
from benchmark_points_active
group by 1, 2
`;
}

function buildChartLatestValuesStatement() {
  return `
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
 and cli.latest_commit_idx = bpa.commit_idx
`;
}

function buildChartSeriesLatestValuesStatement() {
  return `
create or replace table chart_series_latest_values as
select
  group_name,
  chart_name,
  series_name,
  arg_max(value, commit_idx) as latest_value
from benchmark_points_active
group by 1, 2, 3
`;
}

export function buildBootstrapSql(dataPath, commitsPath) {
  return joinStatements([
    buildQuerySuitesStatement(),
    buildValidGroupsStatement(),
    buildEngineRenamesStatement(),
    buildRawCommitsViewStatement(commitsPath),
    buildCommitDimStatement(),
    buildRawBenchmarksViewStatement(dataPath),
    buildBenchmarksBaseStatement(),
    buildMatchedSuitesStatement(),
    buildClassifiedBenchmarksStatement(),
    buildBenchmarkPointsStatement(),
    buildActiveCommitsStatement(),
    buildBenchmarkPointsActiveStatement(),
    buildChartDefinitionsStatement(),
    buildChartLatestIndexStatement(),
    buildChartLatestValuesStatement(),
    buildChartSeriesLatestValuesStatement(),
  ]);
}
