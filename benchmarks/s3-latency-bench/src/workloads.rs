// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The measured workloads.
//!
//! Microbenchmarks (`size_sweep`, `suffix`, `concurrency`) isolate one variable at a time so the
//! latency model can be fitted. The `scan` phase replays the request stream of Vortex scans:
//! HEAD, the 64 KiB footer suffix read, the spill-over footer read, then coalesced segment reads
//! issued with `vortex-io`'s object store concurrency, across many files at once, with scans that
//! overlap each other in time and in the bytes they read.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::future::join_all;
use futures::stream;
use object_store::GetRange;
use object_store::path::Path;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Serialize;

use crate::layout::FileLayout;
use crate::layout::coalesce;
use crate::recorder::Recorder;
use crate::recorder::Tag;

/// `vortex_io::object_store::DEFAULT_CONCURRENCY`.
pub const VORTEX_OBJECT_STORE_CONCURRENCY: usize = 192;
/// `vortex_io::CoalesceConfig::object_storage()`: 1 MiB distance, 16 MiB max.
pub const VORTEX_COALESCE: (u64, u64) = (1 << 20, 16 << 20);

pub struct Dataset {
    pub prefix: Path,
    pub files: Vec<FileLayout>,
}

impl Dataset {
    pub fn path(&self, file: usize) -> Path {
        self.prefix.clone().join(self.files[file].name.as_str())
    }
}

pub struct Ctx {
    pub rec: Arc<Recorder>,
    pub ds: Arc<Dataset>,
    /// Multiplies repetitions and file counts; 1.0 is the CI configuration.
    pub scale: f64,
    pub seed: u64,
}

impl Ctx {
    fn reps(&self, n: usize) -> usize {
        ((n as f64 * self.scale).round() as usize).max(1)
    }

    fn rng(&self, salt: u64) -> StdRng {
        StdRng::seed_from_u64(self.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt)
    }

    fn file_count(&self, n: usize) -> usize {
        self.reps(n).min(self.ds.files.len())
    }
}

fn tag(phase: &'static str, scenario: impl Into<String>, kind: &'static str) -> Tag {
    Tag {
        phase,
        scenario: scenario.into(),
        kind,
        segments: 1,
        concurrency: 1,
        ..Default::default()
    }
}

fn random_range(rng: &mut StdRng, size: u64, len: u64) -> Range<u64> {
    let len = len.min(size);
    let start = rng.random_range(0..=size - len);
    start..start + len
}

/// First requests of the process: DNS, TCP and TLS setup show up here.
pub async fn cold(ctx: &Ctx) {
    let mut rng = ctx.rng(1);
    for _ in 0..8 {
        let file = rng.random_range(0..ctx.ds.files.len());
        let path = ctx.ds.path(file);
        let size = ctx.ds.files[file].size;
        let mut t = tag("cold", "open", "head");
        t.file = file as u32;
        ctx.rec.head(&path, t.clone()).await;
        let (initial, _) = ctx.ds.files[file].open_reads();
        t.kind = "footer";
        t.useful_bytes = size - initial.start;
        ctx.rec.get(&path, GetRange::Bounded(initial), t).await;
    }
}

/// Serial reads of many sizes at random offsets: the clean latency-vs-bytes curve.
pub async fn size_sweep(ctx: &Ctx) {
    let mut rng = ctx.rng(2);
    let max_len = ctx.ds.files.iter().map(|f| f.size).min().unwrap_or(0);
    let mut sizes: Vec<u64> = Vec::new();
    for exp in [
        0u32, 6, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    ] {
        let len = 1u64 << exp;
        if len <= max_len {
            sizes.extend(std::iter::repeat_n(len, ctx.reps(20)));
        }
    }
    // Log-uniform sizes fill in the curve between the powers of two.
    let log_max = (max_len.min(16 << 20) as f64).log2();
    for _ in 0..ctx.reps(400) {
        sizes.push(2f64.powf(rng.random_range(0.0..log_max)) as u64);
    }
    sizes.shuffle(&mut rng);
    for len in sizes {
        let file = rng.random_range(0..ctx.ds.files.len());
        let range = random_range(&mut rng, ctx.ds.files[file].size, len.max(1));
        let mut t = tag("size_sweep", "serial", "get");
        t.file = file as u32;
        t.useful_bytes = range.end - range.start;
        ctx.rec
            .get(&ctx.ds.path(file), GetRange::Bounded(range), t)
            .await;
    }
}

/// Suffix (`bytes=-N`) reads against the equivalent bounded tail read, plus HEAD.
pub async fn suffix(ctx: &Ctx) {
    let mut rng = ctx.rng(3);
    let mut ops: Vec<(&'static str, u64)> = Vec::new();
    for len in [8u64, 1 << 10, 8 << 10, 65535, 256 << 10, 1 << 20, 4 << 20] {
        for _ in 0..ctx.reps(15) {
            ops.push(("suffix", len));
            ops.push(("tail_bounded", len));
        }
    }
    ops.extend(std::iter::repeat_n(("head", 0), ctx.reps(40)));
    ops.shuffle(&mut rng);
    for (kind, len) in ops {
        let file = rng.random_range(0..ctx.ds.files.len());
        let path = ctx.ds.path(file);
        let size = ctx.ds.files[file].size;
        let mut t = tag("suffix", format!("len_{len}"), kind);
        t.file = file as u32;
        t.useful_bytes = len;
        match kind {
            "head" => {
                ctx.rec.head(&path, t).await;
            }
            "suffix" => {
                ctx.rec.get(&path, GetRange::Suffix(len), t).await;
            }
            _ => {
                ctx.rec
                    .get(&path, GetRange::Bounded(size - len..size), t)
                    .await;
            }
        }
    }
}

/// Fixed-size reads at increasing concurrency: how latency and throughput respond to load.
pub async fn concurrency(ctx: &Ctx) {
    let mut rng = ctx.rng(4);
    let budget = (512u64 << 20) as f64 * ctx.scale;
    for len in [4u64 << 10, 64 << 10, 1 << 20, 8 << 20] {
        for conc in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
            let by_budget = (budget / len as f64) as usize;
            let n = by_budget.clamp(conc * 2, (conc * 8).max(ctx.reps(64)));
            if n as u64 * len > 2 << 30 {
                continue;
            }
            let reads: Vec<(usize, Range<u64>)> = (0..n)
                .map(|_| {
                    let file = rng.random_range(0..ctx.ds.files.len());
                    (file, random_range(&mut rng, ctx.ds.files[file].size, len))
                })
                .collect();
            stream::iter(reads)
                .map(|(file, range)| {
                    let mut t = tag("concurrency", format!("len_{len}_c{conc}"), "get");
                    t.file = file as u32;
                    t.concurrency = conc as u32;
                    t.useful_bytes = len;
                    let path = ctx.ds.path(file);
                    async move { ctx.rec.get(&path, GetRange::Bounded(range), t).await }
                })
                .buffer_unordered(conc)
                .collect::<Vec<_>>()
                .await;
        }
    }
}

#[derive(Clone, Debug)]
pub enum Plan {
    /// Read every column of every row group.
    Full,
    /// Read a subset of columns of every row group.
    Projection(Vec<u32>),
    /// Prune with the filter column's zone map, read the filter column, then read the projection
    /// (which includes the filter column again) for the row groups that pass.
    FilterProject {
        filter: u32,
        projection: Vec<u32>,
        zone_pass: f64,
        selectivity: f64,
    },
    /// Open the file and read every column of one row group.
    PointLookup,
}

#[derive(Clone, Debug)]
pub struct ScanSpec {
    pub scenario: String,
    pub files: Vec<usize>,
    pub plan: Plan,
    pub file_concurrency: usize,
    pub per_file_concurrency: usize,
    pub coalesce: Option<(u64, u64)>,
    /// Row groups registered with the segment source at a time (a scan split).
    pub split_row_groups: u32,
    pub start_delay: Duration,
}

#[derive(Debug, Serialize)]
pub struct ScanSummary {
    pub scenario: String,
    pub scan_id: u32,
    pub files: usize,
    pub file_concurrency: usize,
    pub per_file_concurrency: usize,
    pub coalesce: bool,
    pub start_us: u64,
    pub end_us: u64,
    pub bytes: u64,
}

/// Coalesce one split's worth of segment ranges at a time, as the segment source sees them.
fn split_reads(
    layout: &FileLayout,
    columns: &[u32],
    row_groups: &[u32],
    split: u32,
    config: Option<(u64, u64)>,
) -> Vec<crate::layout::CoalescedRead> {
    row_groups
        .chunks(split.max(1) as usize)
        .flat_map(|rgs| {
            let ranges = rgs
                .iter()
                .flat_map(|rg| columns.iter().flat_map(|c| layout.data_segments(*c, *rg)))
                .map(|s| s.range())
                .collect();
            coalesce(ranges, config)
        })
        .collect()
}

async fn read_all(
    ctx: &Ctx,
    path: &Path,
    reads: Vec<crate::layout::CoalescedRead>,
    concurrency: usize,
    base: &Tag,
    kind: &'static str,
) -> u64 {
    stream::iter(reads)
        .map(|read| {
            let mut t = base.clone();
            t.kind = kind;
            t.segments = read.segments;
            t.useful_bytes = read.useful_bytes;
            async move { ctx.rec.get(path, GetRange::Bounded(read.range), t).await }
        })
        .buffer_unordered(concurrency.max(1))
        .fold(0, |acc, b| async move { acc + b })
        .await
}

async fn scan_file(ctx: &Ctx, spec: &ScanSpec, scan_id: u32, file: usize, rng_seed: u64) -> u64 {
    let layout = &ctx.ds.files[file];
    let path = ctx.ds.path(file);
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut base = tag("scan", spec.scenario.clone(), "head");
    base.scan_id = scan_id;
    base.file = file as u32;
    base.concurrency = spec.per_file_concurrency as u32;

    ctx.rec.head(&path, base.clone()).await;
    let (initial, rest) = layout.open_reads();
    let mut t = base.clone();
    t.kind = "footer";
    t.useful_bytes = initial.end - initial.start;
    let mut bytes = ctx.rec.get(&path, GetRange::Bounded(initial), t).await;
    if let Some(rest) = rest {
        let mut t = base.clone();
        t.kind = "footer_rest";
        t.useful_bytes = rest.end - rest.start;
        bytes += ctx.rec.get(&path, GetRange::Bounded(rest), t).await;
    }

    let all_rgs: Vec<u32> = (0..layout.row_groups).collect();
    let all_cols: Vec<u32> = (0..layout.columns).collect();
    let conc = spec.per_file_concurrency;
    let split = spec.split_row_groups;
    bytes += match &spec.plan {
        Plan::Full => {
            let reads = split_reads(layout, &all_cols, &all_rgs, split, spec.coalesce);
            read_all(ctx, &path, reads, conc, &base, "segment").await
        }
        Plan::Projection(cols) => {
            let cols: Vec<u32> = cols.iter().map(|c| c % layout.columns).collect();
            let reads = split_reads(layout, &cols, &all_rgs, split, spec.coalesce);
            read_all(ctx, &path, reads, conc, &base, "segment").await
        }
        Plan::FilterProject {
            filter,
            projection,
            zone_pass,
            selectivity,
        } => {
            let filter = filter % layout.columns;
            let mut n = 0;
            if let Some(zm) = layout.zone_map(filter) {
                let read = crate::layout::CoalescedRead {
                    range: zm.range(),
                    segments: 1,
                    useful_bytes: zm.len,
                };
                n += read_all(ctx, &path, vec![read], 1, &base, "zonemap").await;
            }
            let kept: Vec<u32> = all_rgs
                .iter()
                .copied()
                .filter(|_| rng.random_bool(*zone_pass))
                .collect();
            let reads = split_reads(layout, &[filter], &kept, split, spec.coalesce);
            n += read_all(ctx, &path, reads, conc, &base, "filter").await;
            let selected: Vec<u32> = kept
                .into_iter()
                .filter(|_| rng.random_bool(*selectivity))
                .collect();
            let mut cols: Vec<u32> = projection.iter().map(|c| c % layout.columns).collect();
            cols.push(filter);
            cols.sort_unstable();
            cols.dedup();
            let reads = split_reads(layout, &cols, &selected, split, spec.coalesce);
            n + read_all(ctx, &path, reads, conc, &base, "project").await
        }
        Plan::PointLookup => {
            let rg = rng.random_range(0..layout.row_groups);
            let reads = split_reads(layout, &all_cols, &[rg], 1, spec.coalesce);
            read_all(ctx, &path, reads, conc, &base, "segment").await
        }
    };
    bytes
}

pub async fn run_scan(ctx: &Ctx, spec: ScanSpec, scan_id: u32) -> ScanSummary {
    tokio::time::sleep(spec.start_delay).await;
    let start_us = ctx.rec.elapsed_us();
    let seed = ctx.seed ^ (u64::from(scan_id) << 32);
    let bytes = stream::iter(spec.files.iter().enumerate())
        .map(|(i, &file)| scan_file(ctx, &spec, scan_id, file, seed ^ i as u64))
        .buffer_unordered(spec.file_concurrency.max(1))
        .fold(0, |acc, b| async move { acc + b })
        .await;
    ScanSummary {
        scenario: spec.scenario.clone(),
        scan_id,
        files: spec.files.len(),
        file_concurrency: spec.file_concurrency,
        per_file_concurrency: spec.per_file_concurrency,
        coalesce: spec.coalesce.is_some(),
        start_us,
        end_us: ctx.rec.elapsed_us(),
        bytes,
    }
}

/// Every scan scenario, run one group at a time. Scans within a group run concurrently.
pub async fn scans(ctx: &Ctx) -> Vec<ScanSummary> {
    let mut rng = ctx.rng(5);
    let nfiles = ctx.ds.files.len();
    let mut all: Vec<usize> = (0..nfiles).collect();
    all.shuffle(&mut rng);
    let take = |n: usize| all[..ctx.file_count(n)].to_vec();
    let three_cols = |rng: &mut StdRng| -> Vec<u32> {
        let mut cols: Vec<u32> = (0..16).collect();
        cols.shuffle(rng);
        cols.truncate(3);
        cols
    };

    let spec = |scenario: &str, files: Vec<usize>, plan: Plan, fc: usize| ScanSpec {
        scenario: scenario.to_string(),
        files,
        plan,
        file_concurrency: fc,
        per_file_concurrency: VORTEX_OBJECT_STORE_CONCURRENCY,
        coalesce: Some(VORTEX_COALESCE),
        split_row_groups: 8,
        start_delay: Duration::ZERO,
    };

    let proj = Plan::Projection(three_cols(&mut rng));
    let mut groups: Vec<Vec<ScanSpec>> = vec![
        vec![spec("full_fc1", take(4), Plan::Full, 1)],
        vec![spec("full_fc8", take(24), Plan::Full, 8)],
        vec![spec("full_fc32", take(100), Plan::Full, 32)],
        vec![spec("proj3_fc16", take(100), proj.clone(), 16)],
        vec![ScanSpec {
            coalesce: None,
            ..spec("proj3_fc16_nocoalesce", take(100), proj.clone(), 16)
        }],
        vec![ScanSpec {
            per_file_concurrency: 8,
            ..spec("proj3_fc16_pfc8", take(100), proj, 16)
        }],
        vec![spec(
            "filter_project_fc16",
            take(100),
            Plan::FilterProject {
                filter: rng.random_range(0..16),
                projection: three_cols(&mut rng),
                zone_pass: 0.5,
                selectivity: 0.2,
            },
            16,
        )],
    ];

    // Four scans staggered by 250ms over overlapping windows of files, each with its own
    // projection: overlapping in time, and in bytes wherever projections share columns.
    let window = ctx.file_count(40);
    let stride = (nfiles.saturating_sub(window) / 3).max(1);
    groups.push(
        (0..4)
            .map(|i| {
                let files: Vec<usize> = (0..window)
                    .map(|j| all[(i * stride + j) % nfiles])
                    .collect();
                ScanSpec {
                    start_delay: Duration::from_millis(250 * i as u64),
                    ..spec(
                        "concurrent_scans",
                        files,
                        Plan::Projection(three_cols(&mut rng)),
                        8,
                    )
                }
            })
            .collect(),
    );

    // Repeated random-access lookups: each reopens a file (no footer cache).
    let lookups: Vec<usize> = (0..ctx.reps(400))
        .map(|_| rng.random_range(0..nfiles))
        .collect();
    groups.push(vec![spec(
        "point_lookup_c32",
        lookups,
        Plan::PointLookup,
        32,
    )]);

    let mut summaries = Vec::new();
    let mut scan_id = 0u32;
    for group in groups {
        let futures = group.into_iter().map(|s| {
            scan_id += 1;
            run_scan(ctx, s, scan_id)
        });
        let futures: Vec<_> = futures.collect();
        summaries.extend(join_all(futures).await);
    }
    summaries
}
