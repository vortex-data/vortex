# Benchmarks Website

This directory contains the benchmark website frontend, the Node HTTP server, and the DuckDB-based
refresh pipeline that turns the raw benchmark artifacts into chartable time series.

For the data model and table relationships, start with [SCHEMA.md](./SCHEMA.md).
For the upstream artifact generation and refresh/materialization flow, see [ETL.md](./ETL.md).

## Prerequisites

- Node.js `>=18`
- npm
- Optional: DuckDB CLI, if you want to query the cached artifacts directly

Install dependencies:

```bash
cd benchmarks-website
npm install
```

## Development Server

Run the frontend and backend together:

```bash
cd benchmarks-website
npm run dev
```

That starts:

- Vite on `http://localhost:5173`
- the API/static server on `http://localhost:3000`

Useful endpoints:

- `http://localhost:3000/api/metadata`
- `http://localhost:3000/api/health`

The backend refreshes from these artifact URLs by default:

- `https://vortex-ci-benchmark-results.s3.amazonaws.com/data.json.gz`
- `https://vortex-ci-benchmark-results.s3.amazonaws.com/commits.json`

Relevant environment variables:

```bash
PORT=3000
REFRESH_INTERVAL=300000
DATA_URL=https://vortex-ci-benchmark-results.s3.amazonaws.com/data.json.gz
COMMITS_URL=https://vortex-ci-benchmark-results.s3.amazonaws.com/commits.json
CACHE_DIR=/path/to/local/cache
```

`CACHE_DIR` is the most useful one during development. If it is unset, the server uses a temp
directory under `os.tmpdir()`.

## Pull The Data Locally

If you want a predictable local copy for exploration, populate a cache directory yourself and point
the server at it.

```bash
cd benchmarks-website
mkdir -p .cache/benchmarks

curl -L \
  https://vortex-ci-benchmark-results.s3.amazonaws.com/data.json.gz \
  -o .cache/benchmarks/data.json.gz

curl -L \
  https://vortex-ci-benchmark-results.s3.amazonaws.com/commits.json \
  -o .cache/benchmarks/commits.json
```

Then start the dev server against that cache:

```bash
cd benchmarks-website
CACHE_DIR="$PWD/.cache/benchmarks" npm run dev
```

On first startup, the server will use the cached files immediately and then asynchronously
revalidate them against S3.

## Explore The Cached Data Directly

Once `data.json.gz` and `commits.json` exist locally, you can query them with DuckDB without
running the website.

Example with the DuckDB CLI:

```sql
create view raw_commits as
select *
from read_json(
  '.cache/benchmarks/commits.json',
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

create view raw_benchmarks as
select *
from read_json(
  '.cache/benchmarks/data.json.gz',
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
```

Useful starter queries:

```sql
select count(*) as commit_count from raw_commits;

select count(*) as benchmark_count from raw_benchmarks;

select split_part(name, '/', 1) as prefix, count(*) as rows
from raw_benchmarks
group by 1
order by 2 desc
limit 20;

select
  coalesce(json_extract_string(commit, '$.id'), commit_id) as resolved_commit_id,
  count(*) as rows
from raw_benchmarks
group by 1
order by 2 desc
limit 20;
```

If you want the normalized relational model rather than the raw JSON views, follow the pipeline in
[SCHEMA.md](./SCHEMA.md) and [`store/sql.js`](./store/sql.js).

## Export The Full Bootstrap SQL

If you want the exact SQL that the server uses to create all config tables, raw views, normalized
tables, and derived projections, export it from the shared SQL builder:

```bash
cd benchmarks-website
npm run export-sql -- \
  --data-path "$PWD/.cache/benchmarks/data.json.gz" \
  --commits-path "$PWD/.cache/benchmarks/commits.json" \
  --output "$PWD/.cache/benchmarks/bootstrap.sql"
```

Then load it in DuckDB:

```bash
duckdb benchmark-explore.duckdb < .cache/benchmarks/bootstrap.sql
```

That creates the same tables and views the server uses, including:

- `query_suites`
- `valid_groups`
- `engine_renames`
- `raw_commits`
- `raw_benchmarks`
- `commit_dim`
- `benchmarks_base`
- `matched_suites`
- `classified_benchmarks`
- `benchmark_points`
- `active_commits`
- `benchmark_points_active`
- `chart_defs`
- `chart_latest_idx`
- `chart_latest_values`
- `chart_series_latest_values`

If you want a portable template instead of path-specific SQL:

```bash
cd benchmarks-website
npm run export-sql -- --placeholders --output bootstrap.template.sql
```

That emits a script using `__DATA_PATH__` and `__COMMITS_PATH__` placeholders.

## Notes

- The website only projects the subset of the raw benchmark JSON it needs for grouping, charting,
  and summaries.
- Benchmark names are part of the schema. Group, chart, and series identity are inferred from the
  `name`, `storage`, and `dataset` fields during refresh.
- The server returns `503` with `Retry-After` while the initial refresh is still loading.
