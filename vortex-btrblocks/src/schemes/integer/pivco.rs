// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-reset `Pow2-Bin + PIVCO` wavelet-tree integer leaf compression scheme (`vortex.int.pivco`).

use num_traits::AsPrimitive;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_compressor::stats::IntegerErasedStats as ErasedStats;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::FoRReferences;
use vortex_fastlanes::for_v1_id;
use vortex_mask::Mask;
use vortex_pivco::DEFAULT_BLOCK_SHIFT;
use vortex_pivco::EBITS_TABLE;
use vortex_pivco::MAX_BLOCK_VALUES;
use vortex_pivco::MAX_SYMS;
use vortex_pivco::Pivco;
use vortex_pivco::PivcoOptions;
use vortex_pivco::PivcoTreeMode;
use vortex_pivco::block_header_stride;
use vortex_pivco::estimate_compressed_bits;
use vortex_pivco::pivco_id;
use vortex_pivco::pow2_sym_and_ebits;

use super::FoRScheme;
use super::IntDictScheme;
use super::RunEndScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::boolean::BoolRunEndScheme;

const MIN_PIVCO_LEN: usize = 1024;
const SAMPLE_BLOCKS: usize = 4;
const SAMPLE_BLOCK_VALUES: usize = 4096;

const MIN_ABSOLUTE_BITS_SAVED: f64 = 0.75;
const MIN_NARROW_BITS_SAVED: f64 = 0.35;
const MAX_NARROW_REL_RATIO: f64 = 0.80;
/// Maximum `bpv` penalty allowed when auto-promoting default `PHA` to `OrderPreserving` (`OP-PHA`)
/// for free `<` / `<=` / `>` / `>=` bitmap predicate pushdown.
const OP_PHA_PROMOTE_MAX_PENALTY_BPV: f64 = 0.02;
/// Minimum improvement over unbiased `Pivco` required to wrap with `FoR(Pivco)`.
const FOR_PIVCO_MIN_GAIN_BPV: f64 = 0.05;
const MIN_RATIO_DENOMINATOR_BPV: f64 = 0.05;

#[derive(Clone, Copy)]
struct CachedPivcoChoice {
    opts: PivcoOptions,
    use_for: bool,
}

/// Companion scheme with `FoR(Pivco)` disabled, embedded inside [`PivcoScheme`] so
/// [`Scheme::refine`] can drop `fastlanes.for` for any explicit or default [`PivcoOptions`]
/// without global state or allocation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct PivcoSchemeNoFor {
    options: PivcoOptions,
    allow_op_promote: bool,
}

/// Block-reset `Pow2-Bin + PIVCO` wavelet-tree leaf scheme (`vortex.int.pivco`).
///
/// When constructed with [`PivcoScheme::new`], [`PivcoScheme::ph`], [`PivcoScheme::pha`], or
/// [`PivcoScheme::op_pha`], the scheme strictly honors the explicit [`PivcoOptions`] (including its
/// `tree_mode`). When constructed via [`PivcoScheme::default`] or [`PivcoScheme::auto`], it uses
/// `PHA` with automatic block-shift selection and promotes `PHA` to `OrderPreserving` (`OP-PHA`)
/// whenever the order-preserving tree is within `0.02 bpv` of Huffman.
///
/// When `fastlanes.for` is permitted, `min != 0`, and `max - min` fits non-negatively in the
/// column's integer type, `PivcoScheme` also evaluates `FoR(Pivco)` (single-reference
/// `fastlanes.for` over a `vortex.pivco` child) and emits it whenever biasing beats both unbiased
/// `Pivco` and `FoR(BitPacked)`. [`Scheme::refine`] drops `fastlanes.for` when not allowed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PivcoScheme {
    no_for: PivcoSchemeNoFor,
    allow_for: bool,
}

impl Default for PivcoScheme {
    fn default() -> Self {
        Self::auto()
    }
}

impl PivcoScheme {
    /// Creates a [`PivcoScheme`] using adaptive `PHA` with automatic block-shift selection and
    /// automatic `OP-PHA` promotion when order preservation costs at most `0.02 bpv`.
    pub const fn auto() -> Self {
        Self {
            no_for: PivcoSchemeNoFor {
                options: PivcoOptions::pha().with_block_shift(0),
                allow_op_promote: true,
            },
            allow_for: true,
        }
    }

    /// Creates a [`PivcoScheme`] configured with explicit [`PivcoOptions`].
    pub const fn new(options: PivcoOptions) -> Self {
        Self {
            no_for: PivcoSchemeNoFor {
                options,
                allow_op_promote: false,
            },
            allow_for: true,
        }
    }

    /// Creates a [`PivcoScheme`] configured for raw wavelet-tree bitmaps (`PH` mode, auto block
    /// shift).
    pub const fn ph() -> Self {
        Self::new(PivcoOptions::ph().with_block_shift(0))
    }

    /// Creates a [`PivcoScheme`] configured for adaptive skewed-bitmap coding (`PHA` mode, auto
    /// block shift).
    pub const fn pha() -> Self {
        Self::new(PivcoOptions::pha().with_block_shift(0))
    }

    /// Creates a [`PivcoScheme`] configured for order-preserving trees (`OP-PHA` mode, auto block
    /// shift).
    pub const fn op_pha() -> Self {
        Self::new(PivcoOptions::op_pha().with_block_shift(0))
    }

    /// Returns the configured [`PivcoOptions`].
    pub const fn options(&self) -> PivcoOptions {
        self.no_for.options
    }
}

fn can_for_encode_nonneg(
    stats: &ErasedStats,
    ptype: PType,
    compress_ctx: &CompressorContext,
) -> bool {
    if compress_ctx.finished_cascading() {
        return false;
    }
    if ptype.is_signed_int() {
        let signed_max = (1u64 << (ptype.bit_width() - 1)) - 1;
        if stats.max_minus_min() > signed_max {
            return false;
        }
    }
    let for_id = FoRScheme::v1().id();
    let dict_id = IntDictScheme.id();
    !compress_ctx
        .cascade_history()
        .iter()
        .any(|&(ancestor, _)| ancestor == for_id || ancestor == dict_id)
}

fn is_for_pivco_eligible(
    stats: &ErasedStats,
    ptype: PType,
    compress_ctx: &CompressorContext,
) -> bool {
    if stats.min_is_zero() || stats.max_minus_min() == 0 {
        return false;
    }
    can_for_encode_nonneg(stats, ptype, compress_ctx)
}

fn is_worthwhile(bp_bits: f64, pivco_bits: f64) -> bool {
    let saved = bp_bits - pivco_bits;
    saved >= MIN_ABSOLUTE_BITS_SAVED
        || (saved >= MIN_NARROW_BITS_SAVED && pivco_bits <= MAX_NARROW_REL_RATIO * bp_bits)
}

fn min_overhead_bpv(total_len: usize, base_opts: PivcoOptions) -> f64 {
    let max_block_len = if base_opts.block_shift == 0 {
        1usize << 12
    } else {
        1usize << base_opts.block_shift
    };
    let min_blocks = total_len.div_ceil(max_block_len);
    let header_stride = block_header_stride(false);
    let min_overhead_bytes = min_blocks * (header_stride + 1) + header_stride + 18;
    (min_overhead_bytes * 8) as f64 / (total_len as f64)
}

fn pivco_ancestor_exclusions() -> Vec<AncestorExclusion> {
    vec![
        AncestorExclusion {
            ancestor: RunEndScheme.id(),
            children: ChildSelection::One(1),
        },
        AncestorExclusion {
            ancestor: BoolRunEndScheme.id(),
            children: ChildSelection::One(0),
        },
    ]
}

fn pivco_expected_compression_ratio(
    base_opts: PivcoOptions,
    allow_op_promote: bool,
    allow_for: bool,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> CompressionEstimate {
    let total_len = data.array_len();
    if total_len < MIN_PIVCO_LEN {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }
    let stats = data.integer_stats(exec_ctx);
    if stats.value_count() == 0 {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }
    let erased = stats.erased();
    let min_is_negative = erased.min_is_negative();
    let ptype = data.array_as_primitive().ptype();
    let for_eligible = allow_for && is_for_pivco_eligible(erased, ptype, &compress_ctx);
    if min_is_negative && !for_eligible {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }
    let bias_min = for_eligible.then(|| erased.min_i128());

    CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
        move |_compressor, data, best_so_far, compress_ctx, exec_ctx| {
            if matches!(best_so_far, Some(EstimateScore::ZeroBytes)) {
                return Ok(EstimateVerdict::Skip);
            }

            let full_width = ptype.bit_width() as f64;
            let threshold = best_so_far.and_then(EstimateScore::finite_ratio);
            let min_oh_bpv = min_overhead_bpv(total_len, base_opts);
            let max_possible_ratio = full_width / min_oh_bpv.max(MIN_RATIO_DENOMINATOR_BPV);
            if threshold.is_some_and(|t| max_possible_ratio <= t) {
                return Ok(EstimateVerdict::Skip);
            }

            let primitive = data.array_as_primitive();
            let dict_available = !compress_ctx.finished_cascading()
                && !compress_ctx
                    .cascade_history()
                    .iter()
                    .any(|&(ancestor, _)| ancestor == IntDictScheme.id());

            let Some(samples) =
                collect_dual_sampled_histograms(primitive, !min_is_negative, bias_min, exec_ctx)?
            else {
                return Ok(EstimateVerdict::Skip);
            };

            let mut chosen: Option<(f64, CachedPivcoChoice)> = None;
            if let Some(unbiased_hist) = samples.unbiased.as_ref()
                && let Some((bp_bits, pivco_bits, resolved_opts)) = evaluate_candidate(
                    ptype,
                    unbiased_hist,
                    samples.sample_len,
                    samples.null_count,
                    total_len,
                    dict_available,
                    base_opts,
                    allow_op_promote,
                    min_oh_bpv,
                    full_width,
                    threshold,
                )?
                && is_worthwhile(bp_bits, pivco_bits)
            {
                chosen = Some((
                    pivco_bits,
                    CachedPivcoChoice {
                        opts: resolved_opts,
                        use_for: false,
                    },
                ));
            }
            if let Some(biased_hist) = samples.biased.as_ref()
                && let Some((for_bp_bits, for_pivco_bits, for_opts)) = evaluate_candidate(
                    ptype,
                    biased_hist,
                    samples.sample_len,
                    samples.null_count,
                    total_len,
                    dict_available,
                    base_opts,
                    allow_op_promote,
                    min_oh_bpv,
                    full_width,
                    threshold,
                )?
                && is_worthwhile(for_bp_bits, for_pivco_bits)
            {
                let beats_unbiased = match chosen {
                    None => true,
                    Some((unbiased_bits, _)) => {
                        for_pivco_bits + FOR_PIVCO_MIN_GAIN_BPV < unbiased_bits
                    }
                };
                if beats_unbiased {
                    chosen = Some((
                        for_pivco_bits,
                        CachedPivcoChoice {
                            opts: for_opts,
                            use_for: true,
                        },
                    ));
                }
            }

            let Some((best_bits, choice)) = chosen else {
                return Ok(EstimateVerdict::Skip);
            };

            let ratio = full_width / best_bits.max(MIN_RATIO_DENOMINATOR_BPV);
            if threshold.is_some_and(|t| ratio <= t) {
                return Ok(EstimateVerdict::Skip);
            }

            drop(data.get_or_insert_with(|| choice));
            Ok(EstimateVerdict::Ratio(ratio))
        },
    )))
}

fn pivco_compress(
    base_opts: PivcoOptions,
    allow_op_promote: bool,
    allow_for: bool,
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let primitive_array = compressor
        .with_compressed_validity(data.array(), exec_ctx)?
        .execute::<PrimitiveArray>(exec_ctx)?;
    if primitive_array.is_empty() {
        return Ok(primitive_array.into_array());
    }

    let choice = if let Some(cached) = data.get::<CachedPivcoChoice>() {
        *cached
    } else {
        let stats = data.integer_stats(exec_ctx);
        if stats.value_count() == 0 {
            return Ok(primitive_array.into_array());
        }
        let erased = stats.erased();
        let min_is_negative = erased.min_is_negative();
        let ptype = primitive_array.ptype();
        let can_use_for = allow_for && can_for_encode_nonneg(erased, ptype, &compress_ctx);
        if min_is_negative && !can_use_for {
            // Negative inputs cannot be encoded directly by `vortex.pivco` without `FoR`.
            return Ok(primitive_array.into_array());
        }

        let bias_min = if allow_for && is_for_pivco_eligible(erased, ptype, &compress_ctx) {
            Some(erased.min_i128())
        } else if min_is_negative && can_use_for {
            // Constant negative array (`span == 0`) with `allow_for: true`: bias to 0 via `FoR`.
            Some(erased.min_i128())
        } else {
            None
        };

        let computed = choose_uncached_pivco(
            primitive_array.as_view(),
            !min_is_negative,
            bias_min,
            base_opts,
            allow_op_promote,
            exec_ctx,
        )?;
        *data.get_or_insert_with(|| computed)
    };

    if choice.use_for {
        let for_array = FoR::encode(primitive_array, exec_ctx)?;
        let biased = for_array
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;
        let pivco_child = Pivco::encode_with_options(&biased, choice.opts, exec_ctx)?.into_array();
        let FoRReferences::Global(reference) = for_array.references() else {
            unreachable!("FoR::encode always produces a global reference")
        };
        let wrapped = FoR::try_new(pivco_child, reference.clone())?;
        wrapped
            .as_ref()
            .statistics()
            .inherit_from(for_array.as_ref().statistics());
        return Ok(wrapped.into_array());
    }

    Ok(Pivco::encode_with_options(&primitive_array, choice.opts, exec_ctx)?.into_array())
}

impl Scheme for PivcoScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.pivco"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        if self.allow_for {
            vec![pivco_id(), for_v1_id()]
        } else {
            vec![pivco_id()]
        }
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        if self.allow_for && !allowed(&for_v1_id()) {
            &self.no_for
        } else {
            self
        }
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        pivco_ancestor_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        pivco_expected_compression_ratio(
            self.no_for.options,
            self.no_for.allow_op_promote,
            self.allow_for,
            data,
            compress_ctx,
            exec_ctx,
        )
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        pivco_compress(
            self.no_for.options,
            self.no_for.allow_op_promote,
            self.allow_for,
            compressor,
            data,
            compress_ctx,
            exec_ctx,
        )
    }
}

impl Scheme for PivcoSchemeNoFor {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.pivco"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![pivco_id()]
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        pivco_ancestor_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        pivco_expected_compression_ratio(
            self.options,
            self.allow_op_promote,
            false,
            data,
            compress_ctx,
            exec_ctx,
        )
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        pivco_compress(
            self.options,
            self.allow_op_promote,
            false,
            compressor,
            data,
            compress_ctx,
            exec_ctx,
        )
    }
}

fn choose_uncached_pivco(
    primitive: ArrayView<'_, Primitive>,
    want_unbiased: bool,
    bias_min: Option<i128>,
    base_opts: PivcoOptions,
    allow_op_promote: bool,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<CachedPivcoChoice> {
    let ptype = primitive.ptype();
    let total_len = primitive.len();
    let min_oh_bpv = min_overhead_bpv(total_len, base_opts);
    let full_width = ptype.bit_width() as f64;

    if let Some(samples) =
        collect_dual_sampled_histograms(primitive, want_unbiased, bias_min, exec_ctx)?
    {
        let unbiased = match samples.unbiased.as_ref() {
            Some(hist) => evaluate_candidate(
                ptype,
                hist,
                samples.sample_len,
                samples.null_count,
                total_len,
                false,
                base_opts,
                allow_op_promote,
                min_oh_bpv,
                full_width,
                None,
            )?,
            None => None,
        };
        let biased = match samples.biased.as_ref() {
            Some(hist) => evaluate_candidate(
                ptype,
                hist,
                samples.sample_len,
                samples.null_count,
                total_len,
                false,
                base_opts,
                allow_op_promote,
                min_oh_bpv,
                full_width,
                None,
            )?,
            None => None,
        };

        if let Some((_, for_bits, for_opts)) = biased {
            let prefer_for = match unbiased {
                None => true,
                Some((_, u_bits, _)) => for_bits + FOR_PIVCO_MIN_GAIN_BPV < u_bits,
            };
            if prefer_for {
                return Ok(CachedPivcoChoice {
                    opts: for_opts,
                    use_for: true,
                });
            }
        }
        if let Some((_, _, resolved)) = unbiased {
            return Ok(CachedPivcoChoice {
                opts: resolved,
                use_for: false,
            });
        }
    }

    let mut fallback_opts = base_opts;
    if fallback_opts.block_shift == 0 {
        fallback_opts.block_shift = DEFAULT_BLOCK_SHIFT;
    }
    Ok(CachedPivcoChoice {
        opts: fallback_opts,
        use_for: !want_unbiased && bias_min.is_some(),
    })
}

struct CandidateHist {
    sym_counts: [u64; MAX_SYMS],
    bw_hist: [usize; 65],
}

impl CandidateHist {
    fn new() -> Self {
        Self {
            sym_counts: [0u64; MAX_SYMS],
            bw_hist: [0usize; 65],
        }
    }

    #[inline]
    fn record(&mut self, u: u64) {
        self.bw_hist[(u64::BITS - u.leading_zeros()) as usize] += 1;
        self.sym_counts[pow2_sym_and_ebits(u).0] += 1;
    }
}

struct DualSampledHistograms {
    unbiased: Option<CandidateHist>,
    biased: Option<CandidateHist>,
    sample_len: usize,
    null_count: usize,
}

#[inline]
fn unbiased_to_u64<T>(v: T, is_signed: bool) -> Option<u64>
where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    if is_signed {
        let s = AsPrimitive::<i64>::as_(v);
        (s >= 0).then_some(s as u64)
    } else {
        Some(AsPrimitive::<u64>::as_(v))
    }
}

#[inline]
fn biased_to_u64<T>(v: T, is_signed: bool, min_val: i128) -> Option<u64>
where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let val_i128 = if is_signed {
        i128::from(AsPrimitive::<i64>::as_(v))
    } else {
        i128::from(AsPrimitive::<u64>::as_(v))
    };
    let diff = val_i128.checked_sub(min_val)?;
    u64::try_from(diff).ok()
}

/// Collects unbiased and/or `FoR`-biased sample histograms in a single pass over at most
/// `SAMPLE_BLOCKS * SAMPLE_BLOCK_VALUES` rows, slicing validity per sampled block so full-array
/// validity is never materialized.
fn collect_dual_sampled_histograms(
    primitive: ArrayView<'_, Primitive>,
    want_unbiased: bool,
    bias_min: Option<i128>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<DualSampledHistograms>> {
    let total_len = primitive.len();
    if total_len == 0 || (!want_unbiased && bias_min.is_none()) {
        return Ok(None);
    }
    let validity = primitive.validity()?;
    if matches!(validity, Validity::AllInvalid) {
        return Ok(None);
    }

    let ptype = primitive.ptype();
    match_each_integer_ptype!(ptype, |T| {
        collect_dual_sampled_typed::<T>(
            primitive.as_slice::<T>(),
            ptype.is_signed_int(),
            want_unbiased,
            bias_min,
            &validity,
            total_len,
            exec_ctx,
        )
    })
}

fn accumulate_slice<T>(
    slice: &[T],
    is_signed: bool,
    unbiased: &mut Option<CandidateHist>,
    biased: &mut Option<(i128, CandidateHist)>,
) where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    match (unbiased.as_mut(), biased.as_mut()) {
        (Some(u_hist), Some((min_val, b_hist))) => {
            let min_val = *min_val;
            let mut u_ok = true;
            let mut b_ok = true;
            for &v in slice {
                if u_ok {
                    if let Some(u) = unbiased_to_u64(v, is_signed) {
                        u_hist.record(u);
                    } else {
                        u_ok = false;
                    }
                }
                if b_ok {
                    if let Some(b) = biased_to_u64(v, is_signed, min_val) {
                        b_hist.record(b);
                    } else {
                        b_ok = false;
                    }
                }
                if !u_ok && !b_ok {
                    break;
                }
            }
            if !u_ok {
                *unbiased = None;
            }
            if !b_ok {
                *biased = None;
            }
        }
        (Some(u_hist), None) => {
            for &v in slice {
                if let Some(u) = unbiased_to_u64(v, is_signed) {
                    u_hist.record(u);
                } else {
                    *unbiased = None;
                    break;
                }
            }
        }
        (None, Some((min_val, b_hist))) => {
            let min_val = *min_val;
            for &v in slice {
                if let Some(b) = biased_to_u64(v, is_signed, min_val) {
                    b_hist.record(b);
                } else {
                    *biased = None;
                    break;
                }
            }
        }
        (None, None) => {}
    }
}

fn collect_dual_sampled_typed<T>(
    values: &[T],
    is_signed: bool,
    want_unbiased: bool,
    bias_min: Option<i128>,
    validity: &Validity,
    total_len: usize,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<DualSampledHistograms>>
where
    T: NativePType + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let num_blocks = total_len.div_ceil(SAMPLE_BLOCK_VALUES);
    let sampled_blocks = num_blocks.min(SAMPLE_BLOCKS);
    let mut unbiased = want_unbiased.then(CandidateHist::new);
    let mut biased = bias_min.map(|min_val| (min_val, CandidateHist::new()));
    let mut sample_len = 0usize;
    let mut valid_count = 0usize;

    for i in 0..sampled_blocks {
        if unbiased.is_none() && biased.is_none() {
            return Ok(None);
        }
        let blk = if num_blocks <= SAMPLE_BLOCKS {
            i
        } else {
            i * num_blocks / SAMPLE_BLOCKS
        };
        let start = blk * SAMPLE_BLOCK_VALUES;
        let end = (start + SAMPLE_BLOCK_VALUES).min(total_len);
        let block_len = end - start;
        sample_len += block_len;

        match validity {
            Validity::NonNullable | Validity::AllValid => {
                accumulate_slice(&values[start..end], is_signed, &mut unbiased, &mut biased);
                valid_count += block_len;
            }
            Validity::AllInvalid => return Ok(None),
            Validity::Array(_) => {
                let block_mask = validity
                    .slice(start..end)?
                    .execute_mask(block_len, exec_ctx)?;
                match block_mask {
                    Mask::AllTrue(_) => {
                        accumulate_slice(
                            &values[start..end],
                            is_signed,
                            &mut unbiased,
                            &mut biased,
                        );
                        valid_count += block_len;
                    }
                    Mask::AllFalse(_) => {}
                    Mask::Values(mv) => {
                        for (s, e) in mv.bit_buffer().set_slices() {
                            accumulate_slice(
                                &values[start + s..start + e],
                                is_signed,
                                &mut unbiased,
                                &mut biased,
                            );
                            valid_count += e - s;
                        }
                    }
                }
            }
        }
    }

    if valid_count == 0 || sample_len == 0 {
        return Ok(None);
    }
    let null_count = sample_len - valid_count;
    if let Some(u_hist) = unbiased.as_mut() {
        u_hist.bw_hist[0] += null_count;
    }
    let biased = biased.map(|(_, mut b_hist)| {
        b_hist.bw_hist[0] += null_count;
        b_hist
    });
    if unbiased.is_none() && biased.is_none() {
        return Ok(None);
    }

    Ok(Some(DualSampledHistograms {
        unbiased,
        biased,
        sample_len,
        null_count,
    }))
}

fn compute_bp_bits(ptype: PType, bw_hist: &[usize; 65], sample_len: usize) -> f64 {
    let max_bw = ptype.bit_width();
    let actual_max_bw = (0..=max_bw).rev().find(|&bw| bw_hist[bw] > 0).unwrap_or(0);
    let bytes_per_exception = ptype.byte_width() + 4;
    let non_zeros = sample_len - bw_hist[0];
    let bw0_bits = if non_zeros == 0 {
        0
    } else {
        non_zeros * (actual_max_bw + 24) + sample_len.div_ceil(1024) * 64
    };
    let mut num_packed = bw_hist[0];
    let mut best_bits = bw0_bits.min(sample_len * max_bw);
    for (bw, &freq) in bw_hist[1..=max_bw].iter().enumerate() {
        let bw = bw + 1;
        let packed_bytes = (bw * sample_len).div_ceil(8);
        num_packed += freq;
        let exception_bytes = (sample_len - num_packed) * bytes_per_exception;
        best_bits = best_bits.min((packed_bytes + exception_bytes) * 8);
    }
    best_bits as f64 / sample_len as f64
}

#[expect(clippy::too_many_arguments)]
fn evaluate_candidate(
    ptype: PType,
    hist: &CandidateHist,
    sample_len: usize,
    null_count: usize,
    total_len: usize,
    dict_available: bool,
    mut opts: PivcoOptions,
    allow_op_promote: bool,
    min_oh_bpv: f64,
    full_width: f64,
    threshold: Option<f64>,
) -> VortexResult<Option<(f64, f64, PivcoOptions)>> {
    let mut sym_counts = hist.sym_counts;
    let mut fill_sym = 0usize;
    let mut best_exact = 0u64;
    for (sym, &cnt) in sym_counts[..32].iter().enumerate() {
        if cnt > best_exact {
            best_exact = cnt;
            fill_sym = sym;
        }
    }
    sym_counts[fill_sym] += null_count as u64;

    let mut active_syms = 0usize;
    let mut any_extra_bits = false;
    let mut raw_extra_bits_sum = 0.0f64;
    for (sym, &cnt) in sym_counts.iter().enumerate() {
        if cnt > 0 {
            active_syms += 1;
            let extra = EBITS_TABLE[sym];
            if extra > 0 {
                any_extra_bits = true;
                raw_extra_bits_sum += (cnt as f64) * f64::from(extra);
            }
        }
    }

    // Raw extra bits are incompressible, so bail before tree construction when even 0-bit tree
    // bitmaps plus minimum overhead cannot beat `best_so_far`.
    let raw_extra_bpv = raw_extra_bits_sum / (sample_len as f64);
    let min_candidate_bpv = (raw_extra_bpv + min_oh_bpv).max(MIN_RATIO_DENOMINATOR_BPV);
    if threshold.is_some_and(|t| full_width / min_candidate_bpv <= t) {
        return Ok(None);
    }

    let block_len = if opts.block_shift == 0 {
        1usize << DEFAULT_BLOCK_SHIFT
    } else {
        (1usize << opts.block_shift).min(MAX_BLOCK_VALUES)
    };
    let (bits, is_op) = estimate_compressed_bits(&sym_counts, sample_len, block_len, opts)?;
    let mut content_bpv = (bits as f64) / (sample_len as f64);

    // In default auto mode, promote `PHA` -> `OP-PHA` for free `< / <= / > / >=` bitmap predicate
    // pushdown whenever `OrderPreserving` is within `<= 0.02 bpv` of `Huffman`.
    if allow_op_promote && !is_op {
        let mut op_opts = opts;
        op_opts.tree_mode = PivcoTreeMode::OrderPreserving;
        let (op_bits, _) = estimate_compressed_bits(&sym_counts, sample_len, block_len, op_opts)?;
        let op_bpv = (op_bits as f64) / (sample_len as f64);
        if op_bpv <= content_bpv + OP_PHA_PROMOTE_MAX_PENALTY_BPV {
            opts.tree_mode = PivcoTreeMode::OrderPreserving;
            content_bpv = op_bpv;
        }
    }

    if opts.block_shift == 0 {
        // `PIVCO-PH` pays ~12 B/block (`0.09 bpv` at 1024 rows) and `PIVCO-PHA` pays at most 32 B
        // per skewed node (`~0.08 bpv` at 4096 rows).
        opts.block_shift = if !opts.pha_enabled || content_bpv >= 4.0 {
            11 // 2,048 rows/block: fast random access + L1 residency
        } else {
            12 // 4,096 rows/block: full `PHA` skewed-bitmap amortization
        };
    }

    let num_blocks = total_len.div_ceil(1usize << opts.block_shift);
    let header_stride = block_header_stride(any_extra_bits);
    let per_block_overhead =
        header_stride + active_syms.min(16) + if any_extra_bits { 4 } else { 0 };
    let overhead_bytes = num_blocks * per_block_overhead + header_stride + active_syms * 2 + 16;
    let overhead_bpv = (overhead_bytes * 8) as f64 / (total_len as f64);
    let pivco_bits = content_bpv + overhead_bpv;

    let bp_bits = compute_bp_bits(ptype, &hist.bw_hist, sample_len);
    let capped_bp_bits = if !any_extra_bits && dict_available && active_syms >= 2 {
        let dict_bw = f64::from(usize::BITS - (active_syms - 1).leading_zeros());
        let dict_table_bpv = (active_syms * ptype.byte_width() * 8) as f64 / (total_len as f64);
        bp_bits.min(dict_bw + dict_table_bpv)
    } else {
        bp_bits
    };

    Ok(Some((capped_bp_bits, pivco_bits, opts)))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_array::VTable;
    use vortex_array::arrays::Constant;
    use vortex_fastlanes::for_v1_id;
    use vortex_pivco::PivcoOptions;
    use vortex_pivco::pivco_id;

    use super::PivcoScheme;
    use crate::Scheme;
    use crate::SchemeExt;

    #[rstest]
    #[case::auto(PivcoScheme::auto())]
    #[case::ph(PivcoScheme::ph())]
    #[case::pha(PivcoScheme::pha())]
    #[case::op_pha(PivcoScheme::op_pha())]
    #[case::explicit_custom(PivcoScheme::new(PivcoOptions::pha().with_block_shift(11)))]
    fn refine_drops_for_when_disallowed_and_never_declares_constant(
        #[case] scheme: PivcoScheme,
        #[values(false, true)] allow_for: bool,
    ) {
        let allowed = |id: &ArrayId| *id == pivco_id() || (allow_for && *id == for_v1_id());
        let refined = scheme.refine(&allowed);
        assert_eq!(refined.id(), scheme.id());

        let expected = if allow_for {
            vec![pivco_id(), for_v1_id()]
        } else {
            vec![pivco_id()]
        };
        assert_eq!(refined.produced_encodings(), expected);
        assert!(!refined.produced_encodings().contains(&Constant.id()));
    }
}
