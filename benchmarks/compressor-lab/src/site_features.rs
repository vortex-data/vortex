// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tiered features of one compression site, for any dtype, with what each feature group cost.
//!
//! A site is an array the compressor chooses a scheme for: the chunk itself, or a child such as
//! FoR offsets or dict codes. Features are grouped by cost tier:
//!
//! | Tier | Meaning |
//! |---|---|
//! | 0 | context, known before touching the data |
//! | 1 | exactly what `vortex-compressor`'s stats produce (`IntegerStats`, `FloatStats`, ...) |
//! | 2 | one hash pass (distinct values, top-k, entropy) |
//! | 3 | one extra scan without hashing |
//! | 4 | arithmetic on tiers 1–3 (closed-form size estimates) |
//! | 5 | a trial compression of the compressor's sample with each eligible scheme |
//!
//! Tiers 2 and 3 are computed twice: on the whole site (`name`) and on the sample the compressor
//! would estimate on (`s.name`), so the analysis can see what sampling loses. Every group is timed;
//! the time is recorded per group in [`SiteFeatures::costs`]. Tier 2–3 times are measured on an
//! Arrow copy of the site, so they are an estimate of a native implementation's cost, not that cost.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_schema::DataType;
use parking_lot::Mutex;
use serde::Deserialize;
use serde::Serialize;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_arrow::ExecuteArrowExt;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::sample_aligned_chunks;
use vortex_compressor::scheme::ChildDecision;
use vortex_compressor::scheme::ChildRequest;
use vortex_compressor::scheme::CompressionPolicy;
use vortex_compressor::scheme::CompressorContext;
use vortex_compressor::scheme::SchemeId;
use vortex_compressor::scheme::SchemeSelection;
use vortex_compressor::scheme::SiteOutcome;
use vortex_compressor::scheme::SiteRequest;
use vortex_compressor::stats::DecimalStats;
use vortex_compressor::stats::FloatStats;
use vortex_compressor::stats::GenerateStatsOptions;
use vortex_compressor::stats::IntegerStats;
use vortex_compressor::stats::StringStats;

/// The version of the site features. Bump it when a feature is added, removed or changed.
pub const SITE_FEATURES_VERSION: u32 = 1;

/// Rows per aligned sample chunk, as the compressor samples since #11452.
const SAMPLE_CHUNK: usize = 1024;

/// One feature value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    /// The name. Sample-based copies of tier 2–3 features are prefixed `s.`.
    pub name: String,
    /// The cost tier, 0–5.
    pub tier: u8,
    /// The value; `None` where undefined (e.g. on an all-null site).
    pub value: Option<f64>,
}

/// What computing one feature group cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupCost {
    /// The group, e.g. `stats`, `distinct`, `scan`, `s.scan`, `trial`.
    pub group: String,
    /// The tier of the group's features.
    pub tier: u8,
    /// Wall time, ns.
    pub ns: u64,
}

/// Where a site sits in the cascade.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiteContext {
    /// The cascade path from the root: `scheme#child` per level. Empty at the root.
    pub path: Vec<String>,
    /// Cascade levels still allowed below this site.
    pub budget_left: usize,
    /// The schemes eligible at this site, by name.
    pub eligible: Vec<String>,
}

impl SiteContext {
    /// The root of a chunk, with nothing known about eligible schemes.
    pub fn root() -> Self {
        Self {
            path: Vec::new(),
            budget_left: vortex_compressor::scheme::MAX_CASCADE,
            eligible: Vec::new(),
        }
    }

    fn from_ctx(ctx: &CompressorContext, eligible: &[SchemeId]) -> Self {
        let path = ctx
            .cascade_history()
            .iter()
            .map(|(scheme, child)| format!("{scheme}#{child}"))
            .collect::<Vec<_>>();
        Self {
            budget_left: vortex_compressor::scheme::MAX_CASCADE.saturating_sub(path.len()),
            path,
            eligible: eligible.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// The features of one site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiteFeatures {
    /// [`SITE_FEATURES_VERSION`] when computed.
    pub version: u32,
    /// The dtype family: `int`, `float`, `decimal`, `string`, `bool` or `other`.
    pub family: String,
    /// The features, in a fixed order per family.
    pub features: Vec<Feature>,
    /// Cost per group.
    pub costs: Vec<GroupCost>,
}

impl SiteFeatures {
    /// The value of a feature by name.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.features
            .iter()
            .find(|f| f.name == name)
            .and_then(|f| f.value)
    }
}

/// Collects features and costs while they are computed.
struct Out {
    features: Vec<Feature>,
    costs: Vec<GroupCost>,
}

impl Out {
    fn push(&mut self, prefix: &str, tier: u8, name: &str, value: Option<f64>) {
        self.features.push(Feature {
            name: format!("{prefix}{name}"),
            tier,
            value: value.filter(|v| v.is_finite()),
        });
    }

    fn group<T>(&mut self, group: String, tier: u8, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let out = f();
        self.costs.push(GroupCost {
            group,
            tier,
            ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        });
        out
    }
}

/// The dtype family of an array.
pub fn family(dtype: &DType) -> &'static str {
    match dtype {
        DType::Primitive(p, _) if p.is_int() => "int",
        DType::Primitive(..) => "float",
        DType::Decimal(..) => "decimal",
        DType::Utf8(_) | DType::Binary(_) => "string",
        DType::Bool(_) => "bool",
        _ => "other",
    }
}

/// Computes every feature of a site.
///
/// `compressor` enables tier 5: the compressor's sample is compressed once per eligible scheme in
/// `site.eligible`, with that scheme forced at its root. Without it, or with no eligible schemes,
/// tier 5 is empty.
pub fn compute(
    array: &ArrayRef,
    site: &SiteContext,
    compressor: Option<&BtrBlocksCompressor>,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<SiteFeatures> {
    let canonical = array.clone().execute::<Canonical>(ctx)?.into_array();
    let fam = family(canonical.dtype());
    let len = canonical.len();
    let mut out = Out {
        features: Vec::new(),
        costs: Vec::new(),
    };

    // Tier 0: context.
    let (bits, signed) = match canonical.dtype() {
        DType::Primitive(p, _) => (p.bit_width() as f64, p.is_signed_int() || p.is_float()),
        _ => (f64::NAN, false),
    };
    out.push("", 0, "ptype_bits", Some(bits));
    out.push("", 0, "signed", Some(f64::from(u8::from(signed))));
    out.push(
        "",
        0,
        "nullable",
        Some(f64::from(u8::from(canonical.dtype().is_nullable()))),
    );
    out.push("", 0, "log_len", Some((len.max(1) as f64).log2()));
    out.push("", 0, "depth", Some(site.path.len() as f64));
    out.push("", 0, "budget_left", Some(site.budget_left as f64));
    out.push("", 0, "eligible_count", Some(site.eligible.len() as f64));
    if let DType::Decimal(d, _) = canonical.dtype() {
        out.push("", 0, "precision", Some(f64::from(d.precision())));
        out.push("", 0, "scale", Some(f64::from(d.scale())));
    }

    // Tier 1: the compressor's own stats.
    tier1(&canonical, fam, len, &mut out, ctx)?;

    // Tiers 2–3, on the whole site and on the sample.
    let sample = sample(&canonical);
    let whole_arrow = canonical.clone().execute_arrow(None, ctx)?;
    let sample_arrow = match &sample {
        Some(s) => s.clone().execute_arrow(None, ctx)?,
        None => Arc::clone(&whole_arrow),
    };
    for (prefix, arrow) in [("", &whole_arrow), ("s.", &sample_arrow)] {
        values_tiers(arrow, fam, prefix, &mut out)?;
    }

    // Tier 4: closed-form size estimates from the whole-site features.
    if fam == "int" || fam == "decimal" {
        derived(&mut out, len, bits);
    }

    // Tier 5: a trial compression of the sample per eligible scheme.
    if let Some(compressor) = compressor
        && !site.eligible.is_empty()
    {
        let target = sample.unwrap_or_else(|| canonical.clone());
        let canonical_bytes = target.nbytes().max(1) as f64;
        for scheme in &site.eligible {
            let value = out.group(format!("trial.{scheme}"), 5, || {
                // A forced scheme can panic on data it was not meant for; that is a miss.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    trial(compressor, &target, scheme, ctx)
                }))
                .ok()
                .and_then(Result::ok)
                .flatten()
                .map(|bytes| (bytes.max(1) as f64 / canonical_bytes).log2())
            });
            out.push("", 5, &format!("trial.{scheme}"), value);
        }
    }

    Ok(SiteFeatures {
        version: SITE_FEATURES_VERSION,
        family: fam.to_string(),
        features: out.features,
        costs: out.costs,
    })
}

/// The compressor's sample of a site: aligned 1,024-row chunks covering about 1%, or `None` when
/// the site is small enough that the compressor would look at all of it.
fn sample(array: &ArrayRef) -> Option<ArrayRef> {
    let count = (array.len() / 100).div_ceil(SAMPLE_CHUNK).max(1);
    sample_aligned_chunks(array, SAMPLE_CHUNK, u32::try_from(count).ok()?)
}

fn tier1(
    canonical: &ArrayRef,
    fam: &str,
    len: usize,
    out: &mut Out,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<()> {
    let frac = |n: u32| (len > 0).then(|| f64::from(n) / len as f64);
    let distinct_opts = GenerateStatsOptions {
        count_distinct_values: true,
    };
    match fam {
        "int" => {
            let primitive = canonical.clone().execute::<PrimitiveArray>(ctx)?;
            let stats = out.group("stats".into(), 1, || {
                IntegerStats::generate(&primitive, ctx)
            });
            let erased = stats.erased();
            out.push("", 1, "null_frac", frac(stats.null_count()));
            out.push(
                "",
                1,
                "avg_run_length",
                Some(f64::from(stats.average_run_length())),
            );
            out.push(
                "",
                1,
                "bits_for",
                Some(f64::from(
                    u64::BITS - erased.max_minus_min().leading_zeros(),
                )),
            );
            out.push("", 1, "max_ilog2", erased.max_ilog2().map(f64::from));
            out.push(
                "",
                1,
                "min_is_negative",
                Some(f64::from(u8::from(erased.min_is_negative()))),
            );
            out.push(
                "",
                1,
                "min_is_zero",
                Some(f64::from(u8::from(erased.min_is_zero()))),
            );
            let stats = out.group("stats.distinct".into(), 2, || {
                IntegerStats::generate_opts(&primitive, distinct_opts, ctx)
            });
            let values = f64::from(stats.value_count().max(1));
            out.push(
                "",
                2,
                "stats.distinct_frac",
                stats.distinct_count().map(|d| f64::from(d) / values),
            );
            out.push(
                "",
                2,
                "stats.top1_frac",
                stats
                    .most_frequent_value_and_count()
                    .map(|(_, c)| f64::from(c) / values),
            );
        }
        "float" => {
            let primitive = canonical.clone().execute::<PrimitiveArray>(ctx)?;
            let stats = out.group("stats".into(), 1, || FloatStats::generate(&primitive, ctx));
            out.push("", 1, "null_frac", frac(stats.null_count()));
            out.push(
                "",
                1,
                "avg_run_length",
                Some(f64::from(stats.average_run_length())),
            );
            let stats = out.group("stats.distinct".into(), 2, || {
                FloatStats::generate_opts(&primitive, distinct_opts, ctx)
            });
            let values = f64::from(stats.value_count().max(1));
            out.push(
                "",
                2,
                "stats.distinct_frac",
                stats.distinct_count().map(|d| f64::from(d) / values),
            );
        }
        "decimal" => {
            let decimal = canonical.clone().execute::<DecimalArray>(ctx)?;
            let stats = out.group("stats".into(), 1, || {
                DecimalStats::generate_opts(&decimal, distinct_opts, ctx)
            });
            out.push("", 1, "null_frac", frac(stats.null_count()));
            out.push(
                "",
                1,
                "avg_run_length",
                stats.average_run_length().map(f64::from),
            );
            let values = f64::from(stats.value_count().max(1));
            out.push(
                "",
                1,
                "stats.distinct_frac",
                stats.distinct_count().map(|d| f64::from(d) / values),
            );
        }
        "string" => {
            let strings = canonical.clone().execute::<VarBinViewArray>(ctx)?;
            let stats = out.group("stats".into(), 1, || {
                StringStats::generate_opts(&strings, distinct_opts, ctx)
            });
            out.push("", 1, "null_frac", frac(stats.null_count()));
            let values = f64::from(stats.value_count().max(1));
            out.push(
                "",
                1,
                "stats.est_distinct_frac",
                stats
                    .estimated_distinct_count()
                    .map(|d| f64::from(d) / values),
            );
        }
        _ => {
            let nulls = canonical.invalid_count(ctx).unwrap_or(0);
            out.push(
                "",
                1,
                "null_frac",
                (len > 0).then(|| nulls as f64 / len as f64),
            );
        }
    }
    Ok(())
}

/// Tiers 2–3 over an Arrow copy of the site.
fn values_tiers(
    arrow: &ArrowArrayRef,
    fam: &str,
    prefix: &str,
    out: &mut Out,
) -> anyhow::Result<()> {
    // Tier 2 (one hash pass) and tier 3 (one scan) are timed as separate groups.
    let (distinct, scan) = match fam {
        "int" | "decimal" => {
            let values = int_values(arrow).context("integer values")?;
            (
                out.group(format!("{prefix}distinct"), 2, || int_distinct(&values)),
                out.group(format!("{prefix}scan"), 3, || int_scan(&values)),
            )
        }
        "float" => {
            let (values, mantissa) = float_values(arrow).context("float values")?;
            (
                out.group(format!("{prefix}distinct"), 2, || {
                    distinct_features(
                        &sorted_counts(values.iter().map(|v| v.to_bits())),
                        values.len(),
                    )
                }),
                out.group(format!("{prefix}scan"), 3, || float_scan(&values, mantissa)),
            )
        }
        "string" => {
            let values = byte_values(arrow).context("string values")?;
            (
                out.group(format!("{prefix}distinct"), 2, || {
                    let mut d =
                        distinct_features(&sorted_counts(values.iter().copied()), values.len());
                    let lens = sorted_counts(values.iter().map(|v| v.len())).len();
                    d.push((
                        "len_distinct_frac",
                        2,
                        (!values.is_empty()).then(|| lens as f64 / values.len() as f64),
                    ));
                    d
                }),
                out.group(format!("{prefix}scan"), 3, || string_scan(&values)),
            )
        }
        "bool" => {
            let values: Vec<bool> = arrow.as_boolean().iter().flatten().collect();
            (
                Vec::new(),
                out.group(format!("{prefix}scan"), 3, || bool_features(&values)),
            )
        }
        _ => (Vec::new(), Vec::new()),
    };
    for (name, tier, value) in distinct.into_iter().chain(scan) {
        out.push(prefix, tier, name, value);
    }
    Ok(())
}

type Named = Vec<(&'static str, u8, Option<f64>)>;

fn bits(value: u128) -> f64 {
    f64::from(u128::BITS - value.leading_zeros())
}

/// Valid integer values (decimals: unscaled), widened to i128.
fn int_values(arrow: &ArrowArrayRef) -> Option<Vec<i128>> {
    fn collect<T: Into<i128> + Copy>(iter: impl Iterator<Item = Option<T>>) -> Vec<i128> {
        iter.flatten().map(Into::into).collect()
    }
    Some(match arrow.data_type() {
        DataType::Int8 => collect(arrow.as_primitive::<Int8Type>().iter()),
        DataType::Int16 => collect(arrow.as_primitive::<Int16Type>().iter()),
        DataType::Int32 => collect(arrow.as_primitive::<Int32Type>().iter()),
        DataType::Int64 => collect(arrow.as_primitive::<Int64Type>().iter()),
        DataType::UInt8 => collect(arrow.as_primitive::<UInt8Type>().iter()),
        DataType::UInt16 => collect(arrow.as_primitive::<UInt16Type>().iter()),
        DataType::UInt32 => collect(arrow.as_primitive::<UInt32Type>().iter()),
        DataType::UInt64 => collect(arrow.as_primitive::<UInt64Type>().iter()),
        DataType::Decimal32(..) => collect(arrow.as_primitive::<Decimal32Type>().iter()),
        DataType::Decimal64(..) => collect(arrow.as_primitive::<Decimal64Type>().iter()),
        DataType::Decimal128(..) => collect(arrow.as_primitive::<Decimal128Type>().iter()),
        _ => return None,
    })
}

/// Frequencies, sorted descending.
fn sorted_counts<K: std::hash::Hash + Eq>(values: impl Iterator<Item = K>) -> Vec<u32> {
    let mut counts: HashMap<K, u32> = HashMap::default();
    for v in values {
        *counts.entry(v).or_default() += 1;
    }
    let mut freqs: Vec<u32> = counts.into_values().collect();
    freqs.sort_unstable_by(|a, b| b.cmp(a));
    freqs
}

/// Tier 2 features from value frequencies.
fn distinct_features(freqs: &[u32], n: usize) -> Named {
    if n == 0 {
        return vec![
            ("distinct_frac", 2, None),
            ("log2_distinct", 2, None),
            ("top1_frac", 2, None),
            ("top4_cover", 2, None),
            ("top16_cover", 2, None),
            ("top256_cover", 2, None),
            ("entropy_bits", 2, None),
        ];
    }
    let nf = n as f64;
    let cover = |k: usize| freqs.iter().take(k).map(|&c| f64::from(c)).sum::<f64>() / nf;
    let entropy = freqs
        .iter()
        .map(|&c| {
            let p = f64::from(c) / nf;
            -p * p.log2()
        })
        .sum::<f64>();
    vec![
        ("distinct_frac", 2, Some(freqs.len() as f64 / nf)),
        ("log2_distinct", 2, Some((freqs.len() as f64).log2())),
        ("top1_frac", 2, Some(cover(1))),
        ("top4_cover", 2, Some(cover(4))),
        ("top16_cover", 2, Some(cover(16))),
        ("top256_cover", 2, Some(cover(256))),
        ("entropy_bits", 2, Some(entropy)),
    ]
}

fn int_distinct(values: &[i128]) -> Named {
    distinct_features(&sorted_counts(values.iter().copied()), values.len())
}

fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Tier 3 integer features: one scan.
fn int_scan(values: &[i128]) -> Named {
    let n = values.len();
    let names = [
        "min",
        "max",
        "negative_frac",
        "bits_bp",
        "bits_zz",
        "run_frac",
        "longest_run_frac",
        "sorted_frac",
        "strict_sorted_frac",
        "bits_delta",
        "delta_distinct_frac",
        "step_mode_frac",
        "bits_delta2",
        "bits_p50",
        "bits_p90",
        "bits_p99",
        "above_p90_frac",
        "trailing_zeros",
        "gcd_bits",
        "affine_residual_bits",
    ];
    if n == 0 {
        return names.iter().map(|&name| (name, 3, None)).collect();
    }
    let nf = n as f64;
    let (mut min, mut max) = (values[0], values[0]);
    let (mut runs, mut longest, mut current) = (1usize, 1usize, 1usize);
    let (mut sorted, mut strict, mut negative) = (0usize, 0usize, 0usize);
    let (mut dmin, mut dmax) = (i128::MAX, i128::MIN);
    let (mut d2min, mut d2max) = (i128::MAX, i128::MIN);
    let mut trailing = u32::MAX;
    let mut deltas = Vec::with_capacity(n.saturating_sub(1));
    for (i, &v) in values.iter().enumerate() {
        min = min.min(v);
        max = max.max(v);
        negative += usize::from(v < 0);
        if v != 0 {
            trailing = trailing.min(v.trailing_zeros());
        }
        if i > 0 {
            let prev = values[i - 1];
            if prev == v {
                current += 1;
            } else {
                runs += 1;
                current = 1;
            }
            longest = longest.max(current);
            sorted += usize::from(prev <= v);
            strict += usize::from(prev < v);
            let d = v.wrapping_sub(prev);
            dmin = dmin.min(d);
            dmax = dmax.max(d);
            if let Some(&pd) = deltas.last() {
                let d2: i128 = d.wrapping_sub(pd);
                d2min = d2min.min(d2);
                d2max = d2max.max(d2);
            }
            deltas.push(d);
        }
    }
    let mut width_hist = [0usize; 129];
    let mut g = 0u128;
    for &v in values {
        let offset = v.wrapping_sub(min).unsigned_abs();
        width_hist[bits(offset) as usize] += 1;
        g = gcd(g, offset);
    }
    let pct = |p: f64| {
        let target = ((n - 1) as f64 * p) as usize;
        let mut seen = 0;
        for (width, &count) in width_hist.iter().enumerate() {
            seen += count;
            if seen > target {
                return width;
            }
        }
        128
    };
    let p90 = pct(0.9);
    let above_p90 = width_hist[p90 + 1..].iter().sum::<usize>();
    let pairs = (n - 1).max(1) as f64;
    let (bits_delta, delta_distinct, step_mode) = if n > 1 {
        let freqs = sorted_counts(deltas.iter().copied());
        (
            Some(bits(dmax.wrapping_sub(dmin).unsigned_abs())),
            Some(freqs.len() as f64 / pairs),
            Some(f64::from(freqs[0]) / pairs),
        )
    } else {
        (Some(0.0), Some(0.0), Some(1.0))
    };
    let bits_delta2 = (n > 2).then(|| bits(d2max.wrapping_sub(d2min).unsigned_abs()));
    // The line through the first and last values; residuals are what Affine would leave.
    let affine = if n > 1 {
        let slope = (values[n - 1] - values[0]) as f64 / (n - 1) as f64;
        let worst = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (v as f64 - (values[0] as f64 + slope * i as f64)).abs())
            .fold(0.0f64, f64::max);
        Some(if worst < 0.5 {
            0.0
        } else {
            worst.log2().ceil() + 2.0
        })
    } else {
        Some(0.0)
    };
    let bits_zz = bits(min.unsigned_abs().max(max.unsigned_abs())) + 1.0;
    let bits_bp = (min >= 0).then(|| bits(max.unsigned_abs()));
    let values: [Option<f64>; 20] = [
        Some(min as f64),
        Some(max as f64),
        Some(negative as f64 / nf),
        bits_bp,
        Some(bits_zz),
        Some(runs as f64 / nf),
        Some(longest as f64 / nf),
        Some(if n > 1 { sorted as f64 / pairs } else { 1.0 }),
        Some(if n > 1 { strict as f64 / pairs } else { 1.0 }),
        bits_delta,
        delta_distinct,
        step_mode,
        bits_delta2,
        Some(pct(0.5) as f64),
        Some(p90 as f64),
        Some(pct(0.99) as f64),
        Some(above_p90 as f64 / nf),
        Some(if trailing == u32::MAX {
            0.0
        } else {
            f64::from(trailing)
        }),
        Some(bits(g)),
        affine,
    ];
    names
        .iter()
        .zip(values)
        .map(|(&name, value)| (name, 3, value))
        .collect()
}

/// Valid float values widened to f64, and the native mantissa width.
fn float_values(arrow: &ArrowArrayRef) -> Option<(Vec<f64>, u32)> {
    Some(match arrow.data_type() {
        DataType::Float16 => (
            arrow
                .as_primitive::<Float16Type>()
                .iter()
                .flatten()
                .map(f64::from)
                .collect(),
            10,
        ),
        DataType::Float32 => (
            arrow
                .as_primitive::<Float32Type>()
                .iter()
                .flatten()
                .map(f64::from)
                .collect(),
            23,
        ),
        DataType::Float64 => (
            arrow
                .as_primitive::<Float64Type>()
                .iter()
                .flatten()
                .collect(),
            52,
        ),
        _ => return None,
    })
}

/// The fewest decimal digits after the point that reproduce `v`, up to 18.
fn decimal_digits(v: f64) -> u32 {
    for d in 0..=17 {
        let scale = 10f64.powi(d);
        if ((v * scale).round() / scale) == v {
            return d as u32;
        }
    }
    18
}

fn float_scan(values: &[f64], mantissa: u32) -> Named {
    let n = values.len();
    let mut out = Named::new();
    let names = [
        "nan_frac",
        "inf_frac",
        "neg_zero_frac",
        "negative_frac",
        "exponent_range",
        "mantissa_trailing_zeros",
        "integer_frac",
        "decimal_digits_p50",
        "decimal_digits_p99",
        "run_frac",
        "sorted_frac",
        "xor_leading_zeros",
        "xor_trailing_zeros",
    ];
    if n == 0 {
        out.extend(names.iter().map(|&name| (name, 3, None)));
        return out;
    }
    let nf = n as f64;
    let (mut nan, mut inf, mut neg_zero, mut negative, mut integer) = (0, 0, 0, 0, 0);
    let (mut emin, mut emax) = (i32::MAX, i32::MIN);
    let (mut tz, mut runs, mut sorted) = (0f64, 1usize, 0usize);
    let (mut xlz, mut xtz) = (0f64, 0f64);
    let mut digits = Vec::with_capacity(n);
    let pad = 52 - mantissa;
    for (i, &v) in values.iter().enumerate() {
        nan += usize::from(v.is_nan());
        inf += usize::from(v.is_infinite());
        neg_zero += usize::from(v == 0.0 && v.is_sign_negative());
        negative += usize::from(v < 0.0);
        if v.is_finite() {
            integer += usize::from(v.fract() == 0.0);
            digits.push(decimal_digits(v));
            if v != 0.0 {
                let exponent = ((v.to_bits() >> 52) & 0x7ff) as i32;
                emin = emin.min(exponent);
                emax = emax.max(exponent);
            }
        }
        let mantissa_bits = v.to_bits() & ((1u64 << 52) - 1);
        tz += f64::from(mantissa_bits.trailing_zeros().min(52).saturating_sub(pad));
        if i > 0 {
            let prev = values[i - 1];
            if prev.to_bits() != v.to_bits() {
                runs += 1;
            }
            sorted += usize::from(prev <= v);
            let xor = prev.to_bits() ^ v.to_bits();
            xlz += f64::from(xor.leading_zeros());
            xtz += f64::from(xor.trailing_zeros().min(64));
        }
    }
    digits.sort_unstable();
    let pct = |p: f64| {
        (!digits.is_empty()).then(|| f64::from(digits[((digits.len() - 1) as f64 * p) as usize]))
    };
    let pairs = (n - 1).max(1) as f64;
    let values: [Option<f64>; 13] = [
        Some(nan as f64 / nf),
        Some(inf as f64 / nf),
        Some(neg_zero as f64 / nf),
        Some(negative as f64 / nf),
        (emax >= emin).then(|| f64::from(emax - emin)),
        Some(tz / nf),
        Some(integer as f64 / nf),
        pct(0.5),
        pct(0.99),
        Some(runs as f64 / nf),
        Some(if n > 1 { sorted as f64 / pairs } else { 1.0 }),
        (n > 1).then(|| xlz / pairs),
        (n > 1).then(|| xtz / pairs),
    ];
    out.extend(names.iter().zip(values).map(|(&name, v)| (name, 3, v)));
    out
}

/// Valid string or binary values.
fn byte_values(arrow: &ArrowArrayRef) -> Option<Vec<&[u8]>> {
    Some(match arrow.data_type() {
        DataType::Utf8View => arrow
            .as_string_view()
            .iter()
            .flatten()
            .map(str::as_bytes)
            .collect(),
        DataType::BinaryView => arrow.as_binary_view().iter().flatten().collect(),
        DataType::Utf8 => arrow
            .as_string::<i32>()
            .iter()
            .flatten()
            .map(str::as_bytes)
            .collect(),
        DataType::LargeUtf8 => arrow
            .as_string::<i64>()
            .iter()
            .flatten()
            .map(str::as_bytes)
            .collect(),
        DataType::Binary => arrow.as_binary::<i32>().iter().flatten().collect(),
        DataType::LargeBinary => arrow.as_binary::<i64>().iter().flatten().collect(),
        _ => return None,
    })
}

fn string_scan(values: &[&[u8]]) -> Named {
    let n = values.len();
    let mut out = Named::new();
    let names = [
        "len_mean",
        "len_p50",
        "len_p99",
        "len_max",
        "inlined_frac",
        "prefix_shared_mean",
        "common_prefix_len",
        "byte_entropy",
        "ascii_frac",
        "digit_frac",
        "int_parse_frac",
        "run_frac",
        "sorted_frac",
    ];
    if n == 0 {
        out.extend(names.iter().map(|&name| (name, 3, None)));
        return out;
    }
    let nf = n as f64;
    let mut lens: Vec<usize> = values.iter().map(|v| v.len()).collect();
    let total: usize = lens.iter().sum();
    lens.sort_unstable();
    let pct = |p: f64| lens[((n - 1) as f64 * p) as usize] as f64;
    let mut byte_hist = [0u64; 256];
    let (mut shared, mut runs, mut sorted, mut parses) = (0usize, 1usize, 0usize, 0usize);
    let mut common = values[0].len();
    for (i, v) in values.iter().enumerate() {
        for &b in *v {
            byte_hist[b as usize] += 1;
        }
        parses += usize::from(
            !v.is_empty()
                && std::str::from_utf8(v).is_ok_and(|s| s.parse::<i64>().is_ok()),
        );
        common = common.min(
            v.iter()
                .zip(values[0].iter())
                .take_while(|(a, b)| a == b)
                .count(),
        );
        if i > 0 {
            let prev = values[i - 1];
            shared += prev
                .iter()
                .zip(v.iter())
                .take_while(|(a, b)| a == b)
                .count();
            runs += usize::from(prev != *v);
            sorted += usize::from(prev <= *v);
        }
    }
    let bytes = total.max(1) as f64;
    let entropy = byte_hist
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / bytes;
            -p * p.log2()
        })
        .sum::<f64>();
    let ascii = byte_hist[..128].iter().sum::<u64>() as f64 / bytes;
    let digits = byte_hist[b'0' as usize..=b'9' as usize].iter().sum::<u64>() as f64 / bytes;
    let pairs = (n - 1).max(1) as f64;
    let values: [Option<f64>; 13] = [
        Some(total as f64 / nf),
        Some(pct(0.5)),
        Some(pct(0.99)),
        Some(lens[n - 1] as f64),
        Some(lens.iter().filter(|&&l| l <= 12).count() as f64 / nf),
        Some(if n > 1 { shared as f64 / pairs } else { 0.0 }),
        Some(common as f64),
        Some(entropy),
        Some(ascii),
        Some(digits),
        Some(parses as f64 / nf),
        Some(runs as f64 / nf),
        Some(if n > 1 { sorted as f64 / pairs } else { 1.0 }),
    ];
    out.extend(names.iter().zip(values).map(|(&name, v)| (name, 3, v)));
    out
}

fn bool_features(values: &[bool]) -> Named {
    let n = values.len();
    if n == 0 {
        return vec![
            ("true_frac", 3, None),
            ("run_frac", 3, None),
            ("longest_run_frac", 3, None),
        ];
    }
    let (mut runs, mut longest, mut current) = (1usize, 1usize, 1usize);
    for w in values.windows(2) {
        if w[0] == w[1] {
            current += 1;
        } else {
            runs += 1;
            current = 1;
        }
        longest = longest.max(current);
    }
    let nf = n as f64;
    vec![
        (
            "true_frac",
            3,
            Some(values.iter().filter(|&&b| b).count() as f64 / nf),
        ),
        ("run_frac", 3, Some(runs as f64 / nf)),
        ("longest_run_frac", 3, Some(longest as f64 / nf)),
    ]
}

/// Tier 4: log2 of each closed-form size estimate over the canonical size, from whole-site
/// features. The same formulas as `features::with_derived`, on the new names.
fn derived(out: &mut Out, len: usize, width: f64) {
    let get = |name: &str| {
        out.features
            .iter()
            .find(|f| f.name == name)
            .and_then(|f| f.value)
    };
    let width = if width.is_finite() { width } else { 128.0 };
    let (Some(null_frac), Some(bits_zz), Some(run_frac), Some(top1)) = (
        get("null_frac"),
        get("bits_zz"),
        get("run_frac"),
        get("top1_frac"),
    ) else {
        return;
    };
    let len = len.max(1) as f64;
    let n = len * (1.0 - null_frac);
    let pos = len.log2().max(1.0);
    // Decimals have no `bits_for` stat: derive it from the scanned range.
    let bits_for = get("bits_for")
        .or_else(|| Some((get("max")? - get("min")? + 1.0).max(1.0).log2().ceil()))
        .unwrap_or(width);
    let bits_bp = get("bits_bp").unwrap_or(width);
    let distinct = get("distinct_frac").unwrap_or(1.0) * n;
    let runs = (run_frac * n).max(1.0);
    let patches = (n * (1.0 - top1)).max(1.0);
    let estimates = [
        ("est_bp", n * bits_bp),
        ("est_for", n * bits_for),
        ("est_zz", n * bits_zz),
        ("est_delta", n * get("bits_delta").unwrap_or(width)),
        ("est_dict", distinct * width + n * distinct.max(2.0).log2()),
        ("est_runend", runs * (bits_for + pos)),
        ("est_sparse", patches * (bits_for + pos)),
        (
            "est_p90_patched",
            n * get("bits_p90").unwrap_or(width)
                + n * get("above_p90_frac").unwrap_or(0.0) * (width + pos),
        ),
    ];
    for (name, est) in estimates {
        out.push("", 4, name, Some((est.max(1.0) / (len * width)).log2()));
    }
}

/// Compresses `array` with `scheme` forced at the root and the compressor's choice below.
/// `None` when the scheme was not applied (ineligible, failed, or did not shrink the array).
fn trial(
    compressor: &BtrBlocksCompressor,
    array: &ArrayRef,
    scheme: &str,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<Option<u64>> {
    let policy = Arc::new(ForceRoot {
        scheme: scheme.to_string(),
        applied: Mutex::new(None),
    });
    let forced: CascadingCompressor = compressor
        .0
        .clone()
        .with_policy(Arc::<ForceRoot>::clone(&policy));
    let encoded = forced.compress(array, ctx)?;
    let applied = policy.applied.lock().clone();
    Ok((applied.as_deref() == Some(scheme)).then(|| encoded.nbytes()))
}

/// Forces one scheme at the root and records what the root actually got.
#[derive(Debug)]
struct ForceRoot {
    scheme: String,
    applied: Mutex<Option<String>>,
}

impl CompressionPolicy for ForceRoot {
    fn select_scheme(&self, request: &SiteRequest<'_>) -> SchemeSelection {
        if request.ctx.cascade_history().is_empty()
            && let Some(id) = request
                .eligible
                .iter()
                .find(|s| s.to_string() == self.scheme)
        {
            return SchemeSelection::Scheme(*id);
        }
        SchemeSelection::Estimate
    }

    fn observe(&self, outcome: &SiteOutcome<'_>) {
        if outcome.ctx.cascade_history().is_empty() {
            *self.applied.lock() = outcome.scheme.map(|s| s.to_string());
        }
    }
}

/// One site seen during a compression.
#[derive(Debug, Clone)]
pub struct RecordedSite {
    /// The canonical array at the site.
    pub array: ArrayRef,
    /// Where it sits.
    pub site: SiteContext,
    /// The scheme the compressor applied, or `None` for canonical.
    pub applied: Option<String>,
}

/// A policy wrapper that records every site the compressor is asked about, and otherwise defers
/// to the wrapped policy (or the compressor's defaults).
///
/// Wrap the policy a search or plan uses, compress, then compute [`compute`] for each recorded
/// site outside the compression, so feature costs do not pollute compression timings.
#[derive(Debug, Default)]
pub struct SiteRecorder {
    inner: Option<Arc<dyn CompressionPolicy>>,
    sites: Mutex<Vec<RecordedSite>>,
}

impl SiteRecorder {
    /// Records sites, deferring every decision to `inner`, or to the compressor when `None`.
    pub fn new(inner: Option<Arc<dyn CompressionPolicy>>) -> Self {
        Self {
            inner,
            sites: Mutex::default(),
        }
    }

    /// The sites seen so far, in the order the compressor asked about them.
    pub fn sites(&self) -> Vec<RecordedSite> {
        self.sites.lock().clone()
    }
}

impl CompressionPolicy for SiteRecorder {
    fn decide_child(&self, request: &ChildRequest<'_>) -> ChildDecision {
        match &self.inner {
            Some(p) => p.decide_child(request),
            None => request.requested,
        }
    }

    fn select_scheme(&self, request: &SiteRequest<'_>) -> SchemeSelection {
        self.sites.lock().push(RecordedSite {
            array: request.array.clone(),
            site: SiteContext::from_ctx(request.ctx, request.eligible),
            applied: None,
        });
        match &self.inner {
            Some(p) => p.select_scheme(request),
            None => SchemeSelection::Estimate,
        }
    }

    fn rank(&self, candidate: &ArrayRef, ctx: &CompressorContext) -> f64 {
        match &self.inner {
            Some(p) => p.rank(candidate, ctx),
            None => candidate.nbytes() as f64,
        }
    }

    fn observe(&self, outcome: &SiteOutcome<'_>) {
        let path = SiteContext::from_ctx(outcome.ctx, &[]).path;
        if let Some(site) = self
            .sites
            .lock()
            .iter_mut()
            .rev()
            .find(|s| s.site.path == path && s.applied.is_none())
        {
            site.applied = Some(
                outcome
                    .scheme
                    .map_or_else(|| "canonical".to_string(), |s| s.to_string()),
            );
        }
        if let Some(p) = &self.inner {
            p.observe(outcome);
        }
    }

    fn observe_candidate(&self, candidate: &SiteOutcome<'_>) {
        if let Some(p) = &self.inner {
            p.observe_candidate(candidate);
        }
    }
}

/// The tiered features of one compression site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiteRow {
    /// The content hash of the site's canonical array; equal arrays share a row.
    pub hash: String,
    /// Where the site was first seen.
    pub site: SiteContext,
    /// The scheme production applied there, or `canonical`.
    pub applied: Option<String>,
    /// The features.
    pub features: SiteFeatures,
}

/// Compresses `array` with the production compressor, recording every site it chose a scheme
/// for, then computes each distinct site's features outside the compression, root first.
///
/// Sites are deduplicated by the content hash of their canonical array.
pub fn production_sites(
    array: &ArrayRef,
    session: &vortex::session::VortexSession,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<Vec<SiteRow>> {
    let production = BtrBlocksCompressor::from_session(session);
    let recorder = Arc::new(SiteRecorder::new(None));
    BtrBlocksCompressor(
        production
            .0
            .clone()
            .with_policy(Arc::<SiteRecorder>::clone(&recorder)),
    )
    .compress(array, ctx)?;
    let mut sites = recorder.sites();
    if sites.is_empty() {
        // Constant, empty and all-null arrays never reach the policy; keep the root.
        sites.push(RecordedSite {
            array: array.clone(),
            site: SiteContext::root(),
            applied: None,
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut rows = Vec::new();
    for site in sites {
        let canonical = site.array.clone().execute::<Canonical>(ctx)?.into_array();
        let hash = content_hash(&canonical, session)?;
        if !seen.insert(hash.clone()) {
            continue;
        }
        let features = compute(&canonical, &site.site, Some(&production), ctx)?;
        rows.push(SiteRow {
            hash,
            site: site.site,
            applied: site.applied,
            features,
        });
    }
    Ok(rows)
}

/// A digest of an array's serialized form, for any dtype.
pub fn content_hash(
    array: &ArrayRef,
    session: &vortex::session::VortexSession,
) -> anyhow::Result<String> {
    let buffers = array.serialize(
        &vortex_array::ArrayContext::empty(),
        session,
        &vortex_array::serde::SerializeOptions::default(),
    )?;
    let mut bytes = Vec::with_capacity(buffers.iter().map(|b| b.len()).sum());
    for buffer in &buffers {
        bytes.extend_from_slice(buffer.as_ref());
    }
    Ok(crate::key::bytes_digest(&bytes))
}

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex::session::VortexSession;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;

    use super::*;
    use crate::measure;

    fn session() -> VortexSession {
        VortexSession::default()
    }

    #[test]
    fn integer_features_cover_every_tier() -> anyhow::Result<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let array = PrimitiveArray::from_iter((0..10_000i64).map(|i| i / 10 * 3)).into_array();
        let rows = production_sites(&array, &session, &mut ctx)?;
        let root = &rows[0].features;
        assert_eq!(root.family, "int");
        for tier in 0..=5u8 {
            assert!(
                root.features.iter().any(|f| f.tier == tier),
                "no tier {tier} feature"
            );
        }
        assert_eq!(root.get("run_frac"), Some(0.1));
        assert_eq!(root.get("sorted_frac"), Some(1.0));
        assert_eq!(root.get("gcd_bits"), Some(2.0)); // gcd 3
        assert!(root.get("s.distinct_frac").is_some());
        assert!(root.costs.iter().any(|c| c.group == "scan" && c.tier == 3));
        Ok(())
    }

    #[test]
    fn features_for_floats_strings_and_bools() -> anyhow::Result<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let floats =
            PrimitiveArray::from_iter((0..4096).map(|i| f64::from(i % 100) * 0.25)).into_array();
        let float_features = compute(&floats, &SiteContext::root(), None, &mut ctx)?;
        assert_eq!(float_features.family, "float");
        assert_eq!(float_features.get("decimal_digits_p99"), Some(2.0));
        assert_eq!(float_features.get("distinct_frac"), Some(100.0 / 4096.0));

        let strings =
            VarBinViewArray::from_iter_str((0..4096).map(|i| format!("id-{}", i % 7))).into_array();
        let string_features = compute(&strings, &SiteContext::root(), None, &mut ctx)?;
        assert_eq!(string_features.family, "string");
        assert_eq!(string_features.get("common_prefix_len"), Some(3.0));
        assert_eq!(string_features.get("inlined_frac"), Some(1.0));

        let bools = BoolArray::from_iter((0..4096).map(|i| i % 4 == 0)).into_array();
        let b = compute(&bools, &SiteContext::root(), None, &mut ctx)?;
        assert_eq!(b.get("true_frac"), Some(0.25));
        Ok(())
    }

    #[test]
    fn measures_check_pushdown_against_canonical() -> anyhow::Result<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let array = PrimitiveArray::from_iter((0..65_536u32).map(|i| i % 1000)).into_array();
        let encoded = BtrBlocksCompressor::from_session(&session).compress(&array, &mut ctx)?;
        let workload = measure::Workload::standard(&array, 1, &mut ctx)?;
        let config = measure::TimingConfig {
            max_samples: 5,
            ..Default::default()
        };
        let out = measure::measure(
            &array,
            &[("production".into(), encoded)],
            &workload,
            &config,
            &mut ctx,
        )?;
        assert_eq!(out.len(), 2);
        for candidate in &out {
            assert_eq!(candidate.ops.len(), workload.ops.len());
            for op in &candidate.ops {
                assert!(
                    op.correct,
                    "{} {} incorrect: {:?}",
                    candidate.name, op.op, op.error
                );
                assert!(!op.samples_ns.is_empty());
            }
        }
        Ok(())
    }
}
