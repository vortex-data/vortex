// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Features of an integer compression site, from statistics the compressor already holds.

use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexExpect;

use crate::scheme::CompressionPolicy;
use crate::scheme::CompressorContext;
use crate::scheme::SchemeId;
use crate::scheme::SchemeSelection;
use crate::scheme::SiteOutcome;
use crate::scheme::SiteRequest;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::stats::IntegerStats;

/// Values read for the sampled distinct-value features.
const DISTINCT_SAMPLE: usize = 1024;

/// The number of integer site features.
pub const N_INT_FEATURES: usize = 15;

/// The names of the integer site features, in the order [`int_features`] returns them.
///
/// The training script reads these names, so renaming or reordering one needs a new model.
pub const INT_FEATURE_NAMES: [&str; N_INT_FEATURES] = [
    "log2_len",
    "ptype_bits",
    "signed",
    "null_frac",
    "bits_bp",
    "bits_for",
    "for_gain",
    "min_is_negative",
    "log2_avg_run",
    "distinct_frac",
    "bits_distinct",
    "top_frac",
    "depth",
    "parent",
    "child_index",
];

/// Parent schemes, by name, for the `parent` feature. The root is 0 and an unlisted parent is
/// one past the last entry. Append only: an index is part of every trained model.
const PARENT_SCHEMES: &[&str] = &[
    "vortex.int.for",
    "vortex.int.bitpacking",
    "vortex.int.dict",
    "vortex.int.rle",
    "vortex.int.runend",
    "vortex.int.sparse",
    "vortex.int.zigzag",
    "vortex.int.delta",
    "vortex.int.sequence",
    "vortex.float.dict",
    "vortex.float.alp",
    "vortex.float.alprd",
    "vortex.float.rle",
    "vortex.float.sparse",
    "vortex.string.dict",
    "vortex.string.fsst",
    "vortex.binary.dict",
    "vortex.decimal.byte_parts",
    "vortex.decimal.dict",
    "vortex.ext.temporal",
    "vortex.bool.runend",
    "vortex.bool.sparse",
];

/// Number of bits needed to hold `value`.
fn bits(value: u64) -> f64 {
    f64::from(u64::BITS - value.leading_zeros())
}

/// The integer features of a site, from its statistics and position in the cascade.
///
/// Every value is cheap. `stats` need no distinct counts: that hash pass costs more than most
/// schemes do. Distinct-value features come from `sampled`, the distinct fraction and the top
/// value's frequency over an evenly strided sample (see [`sampled_distinct`]).
pub fn int_features(
    stats: &IntegerStats,
    len: usize,
    ptype_bits: u32,
    signed: bool,
    sampled: (f64, f64),
    ctx: &CompressorContext,
) -> [f64; N_INT_FEATURES] {
    let erased = stats.erased();
    let len_f = len.max(1) as f64;
    let values = f64::from(stats.value_count().max(1));
    let bits_bp = erased.max_ilog2().map_or(0.0, |b| f64::from(b + 1));
    let bits_bp = if erased.min_is_negative() {
        f64::from(ptype_bits)
    } else {
        bits_bp
    };
    let bits_for = bits(erased.max_minus_min());
    let (distinct_frac, top_frac) = sampled;
    let bits_distinct = (distinct_frac * values).max(1.0).log2().ceil();
    let history = ctx.cascade_history();
    let (parent, child_index) = match history.last() {
        None => (0.0, -1.0),
        Some((scheme, index)) => {
            let position = PARENT_SCHEMES
                .iter()
                .position(|name| *name == scheme.name)
                .unwrap_or(PARENT_SCHEMES.len());
            ((position + 1) as f64, *index as f64)
        }
    };
    [
        len_f.log2(),
        f64::from(ptype_bits),
        if signed { 1.0 } else { 0.0 },
        f64::from(stats.null_count()) / len_f,
        bits_bp,
        bits_for,
        bits_bp - bits_for,
        if erased.min_is_negative() { 1.0 } else { 0.0 },
        f64::from(stats.average_run_length().max(1)).log2(),
        distinct_frac,
        bits_distinct,
        top_frac,
        history.len() as f64,
        parent,
        child_index,
    ]
}

/// The integer features of the site in `request`, or `None` when the array is not an integer.
pub(super) fn site_int_features(
    request: &SiteRequest<'_>,
    data: &ArrayAndStats,
    exec_ctx: &mut ExecutionCtx,
) -> Option<[f64; N_INT_FEATURES]> {
    let DType::Primitive(ptype, _) = request.array.dtype() else {
        return None;
    };
    if !ptype.is_int() {
        return None;
    }
    let stats = cheap_stats(request, data, exec_ctx);
    let primitive = request
        .array
        .as_opt::<Primitive>()
        .vortex_expect("integer sites are canonical primitives");
    let sampled = match_each_integer_ptype!(primitive.ptype(), |T| {
        sampled_distinct(primitive.as_slice::<T>())
    });
    Some(int_features(
        &stats,
        request.array.len(),
        (ptype.byte_width() * 8) as u32,
        ptype.is_signed_int(),
        sampled,
        request.ctx,
    ))
}

/// Integer statistics without distinct counts, cached apart from the schemes' own.
struct CheapIntStats(Arc<IntegerStats>);

/// The site's integer statistics without distinct counts.
///
/// When no eligible scheme counts distinct values these are the schemes' own statistics, so
/// they are computed once. Otherwise they are a separate pass without the hash table, which is
/// far cheaper than the one the schemes would run.
fn cheap_stats(
    request: &SiteRequest<'_>,
    data: &ArrayAndStats,
    exec_ctx: &mut ExecutionCtx,
) -> Arc<IntegerStats> {
    if !request.ctx.merged_stats_options().count_distinct_values {
        return data.integer_stats(exec_ctx);
    }
    let array = request.array.clone();
    let cached = data.get_or_insert_with::<CheapIntStats>(|| {
        let primitive = array
            .as_opt::<Primitive>()
            .vortex_expect("integer sites are canonical primitives")
            .into_owned();
        CheapIntStats(Arc::new(IntegerStats::generate_opts(
            &primitive,
            GenerateStatsOptions::default(),
            exec_ctx,
        )))
    });
    Arc::clone(&cached.0)
}

/// The distinct fraction and the most frequent value's frequency over up to
/// [`DISTINCT_SAMPLE`] evenly strided values, null slots included.
pub fn sampled_distinct<T: NativePType + Ord>(values: &[T]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let step = (values.len() / DISTINCT_SAMPLE).max(1);
    let mut sample: Vec<T> = values
        .iter()
        .step_by(step)
        .take(DISTINCT_SAMPLE)
        .copied()
        .collect();
    sample.sort_unstable();
    let mut distinct = 1usize;
    let mut run = 1usize;
    let mut top = 1usize;
    for pair in sample.windows(2) {
        if pair[0] == pair[1] {
            run += 1;
        } else {
            distinct += 1;
            run = 1;
        }
        top = top.max(run);
    }
    let n = sample.len() as f64;
    (distinct as f64 / n, top as f64 / n)
}

/// One integer site seen by [`RecordFeatures`].
#[derive(Debug, Clone)]
pub struct SiteFeatures {
    /// The site's cascade history as `scheme#child` steps joined by `/`, empty at the root.
    pub path: String,
    /// The site's cascade history.
    pub history: Vec<(SchemeId, usize)>,
    /// The site's features, in [`INT_FEATURE_NAMES`] order.
    pub features: [f64; N_INT_FEATURES],
    /// The eligible schemes.
    pub eligible: Vec<SchemeId>,
    /// The scheme the compressor applied, or `None` when the site stayed canonical.
    pub applied: Option<SchemeId>,
}

/// Renders a cascade history as `scheme#child` steps joined by `/`.
pub fn path_string(ctx: &CompressorContext) -> String {
    ctx.cascade_history()
        .iter()
        .map(|(scheme, index)| format!("{}#{index}", scheme.name))
        .collect::<Vec<_>>()
        .join("/")
}

/// A policy that records the features of every integer site and changes nothing.
///
/// The compressor lab uses it to gather training rows with the exact code the selector runs.
#[derive(Debug, Default)]
pub struct RecordFeatures {
    /// The sites recorded so far, in the order the compressor chose them.
    sites: Mutex<Vec<SiteFeatures>>,
}

impl RecordFeatures {
    /// A recorder with nothing recorded.
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes the sites recorded so far.
    pub fn take(&self) -> Vec<SiteFeatures> {
        std::mem::take(&mut self.sites.lock())
    }
}

impl CompressionPolicy for RecordFeatures {
    fn select_scheme_with_stats(
        &self,
        request: &SiteRequest<'_>,
        data: &ArrayAndStats,
        exec_ctx: &mut ExecutionCtx,
    ) -> SchemeSelection {
        if let Some(features) = site_int_features(request, data, exec_ctx) {
            self.sites.lock().push(SiteFeatures {
                path: path_string(request.ctx),
                history: request.ctx.cascade_history().to_vec(),
                features,
                eligible: request.eligible.to_vec(),
                applied: None,
            });
        }
        SchemeSelection::Estimate
    }

    fn observe(&self, outcome: &SiteOutcome<'_>) {
        let history = outcome.ctx.cascade_history();
        let mut sites = self.sites.lock();
        if let Some(site) = sites.iter_mut().rev().find(|s| s.history == history) {
            site.applied = outcome.scheme;
        }
    }
}
