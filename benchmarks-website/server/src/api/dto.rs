// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wire-shape data transfer objects for the read API.
//!
//! These structs are the JSON the server emits on `/api/groups`,
//! `/api/group/{slug}`, `/api/chart/{slug}`, and `/health`. The shapes match
//! the contracts documented in `planning/02-contracts.md`; renaming or
//! reordering fields is a wire-compat break and must be coordinated with
//! the emitter and migrator.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value as JsonValue;

/// Default cap on the number of commits returned per chart when no `?n=` is
/// supplied. The HTML routes override this with their own per-page defaults
/// (see [`crate::html`]).
pub const DEFAULT_COMMIT_WINDOW: u32 = 100;

/// Canonical group ordering, ported from the v2 site's hard-coded list at
/// `origin/ct/vfvb:benchmarks-website/index.html`. Group names not in this
/// list sort after every listed name in alphabetical order. The order is
/// significant for the landing page render — every group is collapsed by
/// default, and only the first group's chart payloads are inlined into the
/// HTML so opening it skips a fetch round-trip.
pub const GROUP_ORDER: &[&str] = &[
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
    "TPC-DS (NVMe) (SF=10)",
];

/// Sort key for a group name against [`GROUP_ORDER`]. Names in the list sort
/// by position (0..GROUP_ORDER.len()); names not in the list sort after, by
/// the same primary index plus an alphabetical tiebreaker.
pub fn group_sort_key(name: &str) -> (usize, &str) {
    let pos = GROUP_ORDER
        .iter()
        .position(|&n| n == name)
        .unwrap_or(GROUP_ORDER.len());
    (pos, name)
}

/// Body of `GET /api/groups`: every group with its chart links and summary.
#[derive(Debug, Serialize)]
pub struct GroupsResponse {
    /// Every group surfaced by the discovery passes, in canonical order.
    pub groups: Vec<Group>,
}

/// One group: a display name, a slug for the group permalink, and the chart
/// links inside it. Optionally carries a v2-compatible rollup summary and a
/// short editorial description (rendered as a hover tooltip on the
/// disclosure title).
#[derive(Debug, Serialize)]
pub struct Group {
    /// Human-readable group label rendered in the disclosure header.
    pub name: String,
    /// Slug for `/group/{slug}`. Round-trips through [`crate::slug::GroupKey`].
    pub slug: String,
    /// Chart links inside the group, one per chart card.
    pub charts: Vec<ChartLink>,
    /// Optional v2-compatible rollup computed from the fact tables.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<Summary>,
    /// Short editorial description ported from v2's `BENCHMARK_DESCRIPTIONS` +
    /// `getBenchmarkDescription`. Rendered as a hover tooltip on the disclosure title; absent when
    /// no description exists for this group name (e.g. vector-search groups).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// All charts in one group, returned by `GET /api/group/{slug}`.
#[derive(Debug, Serialize)]
pub struct GroupChartsResponse {
    /// Group display name, matching the `name` field on [`Group`].
    pub name: String,
    /// Optional v2-compatible rollup computed from the fact tables.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<Summary>,
    /// Optional editorial description, mirroring [`Group::description`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Every chart inside the group, with full payload inlined.
    pub charts: Vec<NamedChartResponse>,
}

/// Server-computed group summary, matching the v2 metadata contract.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum Summary {
    /// Random-access format ranking for the latest populated random-access chart.
    #[serde(rename = "randomAccess")]
    RandomAccess {
        /// Header text for the summary card.
        title: &'static str,
        /// One row per format, sorted fastest-first.
        rankings: Vec<RandomAccessRanking>,
        /// Footer explainer rendered below the rankings.
        explanation: &'static str,
    },
    /// Compression/decompression speedup of Vortex over Parquet.
    #[serde(rename = "compression")]
    Compression {
        /// Header text for the summary card.
        title: &'static str,
        /// Geomean of `parquet/vortex` encode ratios; absent if no encode rows exist.
        #[serde(rename = "compressRatio", skip_serializing_if = "Option::is_none")]
        compress_ratio: Option<f64>,
        /// Geomean of `parquet/vortex` decode ratios; absent if no decode rows exist.
        #[serde(rename = "decompressRatio", skip_serializing_if = "Option::is_none")]
        decompress_ratio: Option<f64>,
        /// How many distinct datasets fed the geomean.
        #[serde(rename = "datasetCount")]
        dataset_count: usize,
        /// Footer explainer rendered below the values.
        explanation: &'static str,
    },
    /// Vortex-to-Parquet compressed size ratio distribution.
    #[serde(rename = "compressionSize")]
    CompressionSize {
        /// Header text for the summary card.
        title: &'static str,
        /// Smallest observed `vortex/parquet` ratio across datasets.
        #[serde(rename = "minRatio")]
        min_ratio: f64,
        /// Geomean across datasets.
        #[serde(rename = "meanRatio")]
        mean_ratio: f64,
        /// Largest observed ratio.
        #[serde(rename = "maxRatio")]
        max_ratio: f64,
        /// How many datasets contributed.
        #[serde(rename = "datasetCount")]
        dataset_count: usize,
        /// Footer explainer rendered below the values.
        explanation: &'static str,
    },
    /// Query-suite ranking by geomean ratio to the fastest engine per query.
    #[serde(rename = "queryBenchmark")]
    QueryBenchmark {
        /// Header text for the summary card.
        title: &'static str,
        /// One row per `engine:format`, sorted lowest-score-first.
        rankings: Vec<QueryRanking>,
        /// Footer explainer rendered below the rankings.
        explanation: &'static str,
    },
}

/// One random-access summary row.
#[derive(Debug, Serialize)]
pub struct RandomAccessRanking {
    /// Series name, normally the physical format.
    pub name: String,
    /// Latest measured time in nanoseconds.
    pub time: f64,
    /// Ratio to the fastest series in the same chart.
    pub ratio: f64,
}

/// One query benchmark summary row.
#[derive(Debug, Serialize)]
pub struct QueryRanking {
    /// Series name, normally `engine:format`.
    pub name: String,
    /// Geomean ratio to the fastest observed value per query.
    pub score: f64,
    /// Sum of latest runtimes for the queries this series has.
    #[serde(rename = "totalRuntime")]
    pub total_runtime: f64,
}

/// A single chart inside a [`GroupChartsResponse`]. `name` is the chart's
/// short label inside the group (e.g. `Q1`); `slug` round-trips through
/// `/api/chart/{slug}`.
#[derive(Debug, Serialize)]
pub struct NamedChartResponse {
    /// Chart label rendered in the chart-card title (e.g. `Q1`).
    pub name: String,
    /// Slug for `/chart/{slug}`. Round-trips through [`crate::slug::ChartKey`].
    pub slug: String,
    /// Inlined chart payload — same shape as `/api/chart/{slug}`.
    #[serde(flatten)]
    pub chart: ChartResponse,
}

/// One chart's short label inside a group (e.g. `Q1`) plus the slug that
/// resolves to its `/api/chart/{slug}` payload.
#[derive(Debug, Serialize)]
pub struct ChartLink {
    /// Chart label rendered in the chart-card title (e.g. `Q1`).
    pub name: String,
    /// Slug for `/chart/{slug}`. Round-trips through [`crate::slug::ChartKey`].
    pub slug: String,
}

/// Body of `GET /api/chart/{slug}`: every commit with data, every series'
/// values aligned to those commits, and per-series engine/format tags.
#[derive(Debug, Clone, Serialize)]
pub struct ChartResponse {
    /// Human-readable title for the chart (e.g. `tpch sf=1 Q1 [nvme]`).
    pub display_name: String,
    /// Structured taxonomy describing what the y-axis values *are* (time in
    /// nanoseconds, bytes, etc.). The client uses this together with the
    /// magnitude of the loaded values to pick a display unit (e.g. `ms` for
    /// time values around 1e6 ns) so the rendered axis stays readable. The
    /// taxonomy is small on purpose — see [`UnitKind`].
    pub unit_kind: UnitKind,
    /// Every commit that has at least one rendered data point, oldest first.
    pub commits: Vec<CommitPoint>,
    /// Per-series value arrays, indexed in lockstep with `commits`.
    pub series: serde_json::Map<String, JsonValue>,
    /// Per-series engine/format classification, used by the global filter
    /// bar to hide/show whole engines or formats across every chart at once.
    /// Keyed by series name; values are populated only for series whose name
    /// encodes an engine and/or format. Series without a classification (e.g.
    /// vector-search flavors) are simply absent from this map.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub series_meta: BTreeMap<String, SeriesTag>,
}

/// Structured y-axis unit taxonomy carried on every [`ChartResponse`]. The
/// client uses this — together with the magnitude of the values currently in
/// view — to pick a display unit (e.g. `ms` for `time_ns` values around
/// 1e6) so the rendered axis stays readable. Stored values on the wire are
/// always in the kind's *base* unit:
///
/// | Variant            | Base unit on the wire |
/// |--------------------|-----------------------|
/// | [`Self::TimeNs`]   | nanoseconds           |
/// | [`Self::Bytes`]    | bytes                 |
/// | [`Self::Ratio`]    | dimensionless ratio   |
/// | [`Self::Count`]    | dimensionless count   |
/// | [`Self::ThroughputMbS`] | megabytes per second |
///
/// Adding a variant is a wire-compat change — coordinate with the emitter,
/// migrator, and the client unit picker in `chart-init.js`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitKind {
    /// Times stored as integer nanoseconds. Client picks from `ns | µs | ms | s`.
    TimeNs,
    /// Sizes stored as integer bytes. Client picks from `B | KiB | MiB | GiB | TiB`
    /// (binary multiples) unless a series's `series_meta` overrides it.
    Bytes,
    /// Dimensionless ratio (e.g. compression-ratio derived charts). The
    /// client renders the values verbatim and uses no unit suffix.
    Ratio,
    /// Dimensionless integer count. Same client behaviour as [`Self::Ratio`].
    Count,
    /// Throughput in megabytes per second. The client renders the values
    /// verbatim with an `MB/s` suffix; no auto-scaling is applied today.
    #[serde(rename = "throughput_mb_s")]
    ThroughputMbS,
}

impl UnitKind {
    /// Short label used for the chart-page meta caption (`"unit: ns"`). The
    /// labels match the kind's base unit on the wire and stay stable across
    /// the client's display-unit picker.
    pub fn label(&self) -> &'static str {
        match self {
            Self::TimeNs => "ns",
            Self::Bytes => "bytes",
            Self::Ratio => "ratio",
            Self::Count => "count",
            Self::ThroughputMbS => "MB/s",
        }
    }
}

/// Engine/format tag for one series. Both fields are optional because not
/// every fact table records both dimensions: `query_measurements` carries
/// engine + format, while `compression_*` and `random_access_times` only
/// carry format. Vector-search series have neither and are omitted from the
/// map entirely.
#[derive(Debug, Default, Clone, Serialize)]
pub struct SeriesTag {
    /// Query engine name, e.g. `duckdb` or `datafusion`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    /// Physical format name, e.g. `vortex-file-compressed` or `parquet`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// Universe of engine + format chips the global filter bar can toggle.
/// Returned as a separate, cheap-to-compute summary so the landing page can
/// render the bar without iterating every chart payload.
#[derive(Debug, Default, Serialize)]
pub struct FilterUniverse {
    /// Distinct engine names observed in `query_measurements`.
    pub engines: Vec<String>,
    /// Distinct format names observed across every fact table that records
    /// one (excluding `vector_search_runs`).
    pub formats: Vec<String>,
}

/// One row of the `commits[]` array on a [`ChartResponse`]. Carries enough
/// metadata for the tooltip and the click-to-PR handler in `chart-init.js`.
#[derive(Debug, Clone, Serialize)]
pub struct CommitPoint {
    /// Full git SHA of the commit.
    pub sha: String,
    /// Commit timestamp as ISO-8601 / RFC 3339.
    pub timestamp: String,
    /// First-line commit message (or the full message if no newline).
    pub message: String,
    /// GitHub commit URL — used as the fallback when no `(#NNNN)` is present.
    pub url: String,
}

/// Body of `GET /health`: liveness probe plus a row-count rollup that's
/// useful for "did my ingest land?" smoke tests.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Always `"ok"` when the server is reachable.
    pub status: &'static str,
    /// Path to the DuckDB file the server is serving from.
    pub db_path: String,
    /// Schema version the server was compiled against.
    pub schema_version: i32,
    /// Most recent `commits.timestamp`, or `None` if the table is empty.
    pub latest_commit_timestamp: Option<String>,
    /// Per-fact-table row counts for smoke tests.
    pub row_counts: RowCounts,
}

/// Per-fact-table row counts surfaced by `/health`.
#[derive(Debug, Serialize)]
pub struct RowCounts {
    /// `commits` dim table.
    pub commits: i64,
    /// `query_measurements` fact table.
    pub query_measurements: i64,
    /// `compression_times` fact table.
    pub compression_times: i64,
    /// `compression_sizes` fact table.
    pub compression_sizes: i64,
    /// `random_access_times` fact table.
    pub random_access_times: i64,
    /// `vector_search_runs` fact table.
    pub vector_search_runs: i64,
}
