// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! 32-lane block-reset `Pow2-Bin + Raw-Bits` rANS integer leaf compression scheme.

use num_traits::AsPrimitive;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_rans::BLOCK_VALUES;
use vortex_rans::EBITS_TABLE;
use vortex_rans::MAX_SYMS;
use vortex_rans::Rans;
use vortex_rans::pow2_sym_and_ebits;
use vortex_rans::rans_id;
use vortex_rans::zigzag_encode;

use super::IntDictScheme;
use super::RunEndScheme;
use super::ZigZagScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::boolean::BoolRunEndScheme;

/// Minimum array length to evaluate [`RansScheme`].
const MIN_RANS_LEN: usize = 1024;

/// Maximum number of evenly spaced `4,096`-value blocks sampled by the in-memory estimator.
const SAMPLE_BLOCKS: usize = 4;

/// Unconditional bits-per-value savings threshold over `BitPacked` (`>= 0.75` bpv).
const MIN_ABSOLUTE_BITS_SAVED: f64 = 0.75;

/// Narrow-skewed bits-per-value savings threshold (`>= 0.35` bpv with `<= 0.80 * bp_bits`).
const MIN_NARROW_BITS_SAVED: f64 = 0.35;

/// Relative size ceiling paired with [`MIN_NARROW_BITS_SAVED`].
const MAX_NARROW_REL_RATIO: f64 = 0.80;

/// Per-block fixed byte cost (`32 * 4` state bytes + `16` header bytes = `144` bytes per `4,096`
/// values, or `~0.28125` bits/value on full blocks).
const PER_BLOCK_FIXED_BYTES: usize = 144;

/// 32-lane `Pow2-Bin + Raw-Bits` rANS leaf scheme (`vortex.int.rans`) for skewed integer streams.
///
/// Competes as a leaf alongside `BitPackingScheme` at the bottom of BtrBlocks cascades, including
/// underneath `Dict.codes`, `ALP`, `FoR`, `ZigZag`, `Affine`, `Delta`, and `DecimalByteParts`.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct RansScheme;

impl Scheme for RansScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.rans"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![rans_id(), Constant.id()]
    }

    /// Exclude `RunEnd.ends` and `BoolRunEnd.ends` (binary-searched via `scalar_at`), while
    /// allowing `Dict.codes`, `ALP`, `FoR.v2`, `ZigZag`, `Affine`, `Delta`, and `DecimalByteParts`.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
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

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.is_sample() || data.array_len() < MIN_RANS_LEN {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, best_so_far, compress_ctx, exec_ctx| {
                if matches!(best_so_far, Some(EstimateScore::ZeroBytes)) {
                    return Ok(EstimateVerdict::Skip);
                }

                let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
                let zigzag_available = zigzag_is_available(compressor, &compress_ctx);
                let dict_available = dict_is_available(compressor, &compress_ctx);
                let Some((bp_bits, rans_bits)) = estimate_bp_vs_rans_bits(
                    &primitive,
                    zigzag_available,
                    dict_available,
                    exec_ctx,
                )?
                else {
                    return Ok(EstimateVerdict::Skip);
                };

                let saved = bp_bits - rans_bits;
                let worthwhile = saved >= MIN_ABSOLUTE_BITS_SAVED
                    || (saved >= MIN_NARROW_BITS_SAVED
                        && rans_bits <= MAX_NARROW_REL_RATIO * bp_bits);
                if !worthwhile {
                    return Ok(EstimateVerdict::Skip);
                }

                let full_width = primitive.ptype().bit_width() as f64;
                Ok(EstimateVerdict::Ratio(full_width / rans_bits.max(0.05)))
            },
        )))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive_array = compressor
            .with_compressed_validity(data.array(), exec_ctx)?
            .execute::<PrimitiveArray>(exec_ctx)?;
        Ok(Rans::encode(&primitive_array, exec_ctx)?.into_array())
    }
}

fn zigzag_is_available(compressor: &CascadingCompressor, compress_ctx: &CompressorContext) -> bool {
    let zz_id = ZigZagScheme.id();
    compressor.has_scheme(zz_id)
        && !compress_ctx
            .cascade_history()
            .iter()
            .any(|&(id, _)| id == zz_id)
}

fn dict_is_available(compressor: &CascadingCompressor, compress_ctx: &CompressorContext) -> bool {
    let dict_id = IntDictScheme.id();
    compressor.has_scheme(dict_id)
        && !compress_ctx.finished_cascading()
        && !compress_ctx
            .cascade_history()
            .iter()
            .any(|&(id, _)| id == dict_id)
}

/// Samples up to `4` evenly spaced `4,096`-value blocks (`<= 16,384` elements) and computes
/// `(bp_bits, rans_bits)` in bits/value.
fn estimate_bp_vs_rans_bits(
    array: &PrimitiveArray,
    zigzag_available: bool,
    dict_available: bool,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(f64, f64)>> {
    let total_len = array.len();
    if total_len == 0 {
        return Ok(None);
    }
    let mask = array.validity()?.execute_mask(total_len, exec_ctx)?;
    if matches!(mask, Mask::AllFalse(_)) {
        return Ok(None);
    }

    let ptype = array.ptype();
    Ok(match_each_integer_ptype!(ptype, |T| {
        estimate_typed::<T>(
            array.as_slice::<T>(),
            ptype,
            &mask,
            zigzag_available,
            dict_available,
            total_len,
        )
    }))
}

fn estimate_typed<T>(
    values: &[T],
    ptype: PType,
    mask: &Mask,
    zigzag_available: bool,
    dict_available: bool,
    total_len: usize,
) -> Option<(f64, f64)>
where
    T: NativePType + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let num_blocks = total_len.div_ceil(BLOCK_VALUES);
    let is_signed = ptype.is_signed_int();
    let has_negative = is_signed && sampled_has_negative(values, mask, num_blocks, total_len);

    let sample =
        collect_sampled_histograms(values, is_signed, has_negative, mask, num_blocks, total_len)?;

    let bp_bits = compute_bp_bits(
        ptype,
        &sample.bw_hist,
        sample.sample_len,
        has_negative,
        zigzag_available,
    );
    let (rans_bits, bp_bits) = compute_rans_and_capped_bp_bits(
        ptype,
        &sample.sym_counts,
        sample.sample_len,
        sample.null_count,
        num_blocks,
        total_len,
        dict_available,
        bp_bits,
    );

    Some((bp_bits, rans_bits))
}

#[inline]
fn sampled_block_idx(idx: usize, num_blocks: usize) -> usize {
    if num_blocks <= SAMPLE_BLOCKS {
        idx
    } else {
        idx * num_blocks / SAMPLE_BLOCKS
    }
}

fn sampled_has_negative<T>(values: &[T], mask: &Mask, num_blocks: usize, total_len: usize) -> bool
where
    T: Copy + AsPrimitive<i64>,
{
    let sampled_blocks = num_blocks.min(SAMPLE_BLOCKS);
    for i in 0..sampled_blocks {
        let start = sampled_block_idx(i, num_blocks) * BLOCK_VALUES;
        let end = (start + BLOCK_VALUES).min(total_len);
        match mask {
            Mask::AllTrue(_) => {
                if values[start..end]
                    .iter()
                    .any(|&v| AsPrimitive::<i64>::as_(v) < 0)
                {
                    return true;
                }
            }
            Mask::Values(mv) => {
                let bits = mv.bit_buffer();
                for (rel, &v) in values[start..end].iter().enumerate() {
                    if bits.value(start + rel) && AsPrimitive::<i64>::as_(v) < 0 {
                        return true;
                    }
                }
            }
            Mask::AllFalse(_) => return false,
        }
    }
    false
}

struct SampledHistograms {
    sym_counts: [usize; MAX_SYMS],
    bw_hist: [usize; 65],
    sample_len: usize,
    null_count: usize,
}

fn collect_sampled_histograms<T>(
    values: &[T],
    is_signed: bool,
    has_negative: bool,
    mask: &Mask,
    num_blocks: usize,
    total_len: usize,
) -> Option<SampledHistograms>
where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let sampled_blocks = num_blocks.min(SAMPLE_BLOCKS);
    let map_val = |v: T| -> u64 {
        if has_negative {
            zigzag_encode(AsPrimitive::<i64>::as_(v))
        } else if is_signed {
            AsPrimitive::<i64>::as_(v) as u64
        } else {
            AsPrimitive::<u64>::as_(v)
        }
    };

    let mut sym_counts = [0usize; MAX_SYMS];
    let mut bw_hist = [0usize; 65];
    let mut sample_len = 0usize;
    let mut valid_count = 0usize;

    for i in 0..sampled_blocks {
        let start = sampled_block_idx(i, num_blocks) * BLOCK_VALUES;
        let end = (start + BLOCK_VALUES).min(total_len);
        sample_len += end - start;

        match mask {
            Mask::AllTrue(_) => {
                for &v in &values[start..end] {
                    let u = map_val(v);
                    bw_hist[(u64::BITS - u.leading_zeros()) as usize] += 1;
                    sym_counts[pow2_sym_and_ebits(u).0] += 1;
                }
                valid_count += end - start;
            }
            Mask::Values(mv) => {
                let bits = mv.bit_buffer();
                for (rel, &v) in values[start..end].iter().enumerate() {
                    if bits.value(start + rel) {
                        let u = map_val(v);
                        bw_hist[(u64::BITS - u.leading_zeros()) as usize] += 1;
                        sym_counts[pow2_sym_and_ebits(u).0] += 1;
                        valid_count += 1;
                    }
                }
            }
            Mask::AllFalse(_) => return None,
        }
    }

    if valid_count == 0 || sample_len == 0 {
        return None;
    }
    let null_count = sample_len - valid_count;
    bw_hist[0] += null_count;

    Some(SampledHistograms {
        sym_counts,
        bw_hist,
        sample_len,
        null_count,
    })
}

fn compute_bp_bits(
    ptype: PType,
    bw_hist: &[usize; 65],
    sample_len: usize,
    has_negative: bool,
    zigzag_available: bool,
) -> f64 {
    let full_width = ptype.bit_width() as f64;
    if has_negative && !zigzag_available {
        return full_width;
    }

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
fn compute_rans_and_capped_bp_bits(
    ptype: PType,
    raw_sym_counts: &[usize; MAX_SYMS],
    sample_len: usize,
    null_count: usize,
    num_blocks: usize,
    total_len: usize,
    dict_available: bool,
    bp_bits: f64,
) -> (f64, f64) {
    let mut sym_counts = *raw_sym_counts;
    let mut fill_sym = 0usize;
    let mut best_exact = 0usize;
    for (sym, &cnt) in sym_counts[..32].iter().enumerate() {
        if cnt > best_exact {
            best_exact = cnt;
            fill_sym = sym;
        }
    }
    sym_counts[fill_sym] += null_count;

    let n_f64 = sample_len as f64;
    let mut entropy_and_ebits = 0.0f64;
    let mut active_syms = 0usize;
    let mut any_extra_bits = false;
    for (sym, &cnt) in sym_counts.iter().enumerate() {
        if cnt > 0 {
            active_syms += 1;
            let c = cnt as f64;
            let sym_bits = (n_f64 / c).log2();
            let extra = EBITS_TABLE[sym];
            if extra > 0 {
                any_extra_bits = true;
            }
            entropy_and_ebits += c * (sym_bits + f64::from(extra));
        }
    }

    let capped_bp_bits = if !any_extra_bits && dict_available && active_syms >= 2 {
        let dict_bw = f64::from(usize::BITS - (active_syms - 1).leading_zeros());
        let dict_table_bpv = (active_syms * ptype.byte_width() * 8) as f64 / total_len as f64;
        bp_bits.min(dict_bw + dict_table_bpv)
    } else {
        bp_bits
    };

    let overhead_bytes = num_blocks * PER_BLOCK_FIXED_BYTES + active_syms * 3 + 16;
    let overhead_bpv = (overhead_bytes * 8) as f64 / total_len as f64;
    let rans_bits = (entropy_and_ebits / n_f64) + overhead_bpv;

    (rans_bits, capped_bp_bits)
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::RunEnd;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::arrays::runend::RunEndArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::BitBuffer;
    use vortex_edition::EDITION_DECLARATIONS;
    use vortex_edition::EDITION_FAMILIES;
    use vortex_edition::EditionSession;
    use vortex_edition::EditionSessionExt;
    use vortex_edition::declarations::core::CORE_2026_08_3;
    use vortex_edition::declarations::unstable::UNSTABLE;
    use vortex_error::VortexResult;
    use vortex_rans::Rans;
    use vortex_session::VortexSession;

    use super::RansScheme;
    use crate::BtrBlocksCompressorBuilder;
    use crate::CompressionSession;
    use crate::SESSION;
    use crate::SchemeExt;
    use crate::schemes::integer::DeltaScheme;

    fn find_rans(array: &ArrayRef) -> Option<ArrayRef> {
        array.depth_first_traversal().find(|node| node.is::<Rans>())
    }

    fn session_with_editions(enable_unstable: bool) -> VortexSession {
        let session = VortexSession::empty()
            .with::<EditionSession>()
            .with::<CompressionSession>();
        for family in EDITION_FAMILIES {
            session
                .editions()
                .declare_family(family)
                .expect("edition family");
        }
        for declaration in EDITION_DECLARATIONS {
            session
                .register_edition(declaration)
                .expect("edition declaration");
        }
        session
            .enable_edition(CORE_2026_08_3)
            .expect("core edition");
        if enable_unstable {
            session.enable_edition(UNSTABLE).expect("unstable edition");
        }
        vortex_alp::initialize(&session);
        vortex_fastlanes::initialize(&session);
        vortex_rans::initialize(&session);
        vortex_sequence::initialize(&session);
        vortex_sparse::initialize(&session);
        vortex_zigzag::initialize(&session);
        session
    }

    #[test]
    fn selects_rans_on_statpopgen_gt_and_pl_and_beats_bitpacked() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let with_rans = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();
        let without_rans = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .exclude_schemes([RansScheme.id()])
            .build();

        let mut rng = StdRng::seed_from_u64(11892);
        let len = 65_536usize;

        // 1. `statpopgen.GT`-shaped `{0, 1, 2}` u64 with ~21% nulls & ~78% of valid calls == 0
        //    (Hardy-Weinberg MAF ~0.11: 21% null, 61.6% 0, 16.4% 1, 1.0% 2).
        let gt = PrimitiveArray::from_option_iter((0..len).map(|_| {
            let d = rng.random_range(0..1000u32);
            if d < 210 {
                None
            } else if d < 826 {
                Some(0u64)
            } else if d < 990 {
                Some(1u64)
            } else {
                Some(2u64)
            }
        }))
        .into_array();

        let gt_rans = with_rans.compress(&gt, &mut ctx)?;
        let gt_bp = without_rans.compress(&gt, &mut ctx)?;
        assert_arrays_eq!(&gt_rans, &gt, &mut ctx);
        assert!(
            gt_rans.is::<Rans>(),
            "GT should select vortex.rans directly, got: {}",
            gt_rans.display_tree()
        );
        assert!(
            gt_rans.nbytes() < gt_bp.nbytes(),
            "GT rANS ({}) must beat BitPacked ({})",
            gt_rans.nbytes(),
            gt_bp.nbytes()
        );
        let validity_bytes = (len / 8) as u64;
        assert!(
            (gt_rans.nbytes() - validity_bytes) * 100 < (gt_bp.nbytes() - validity_bytes) * 75,
            "GT rANS data ({}) must beat BitPacked data ({}) by >25%",
            gt_rans.nbytes() - validity_bytes,
            gt_bp.nbytes() - validity_bytes
        );

        // Ultra-sparse 0.5% `{1, 2}` (where BitPacked(bw=0) + compressed patches is below the
        // 0.28 bpv rANS block-state floor) must yield cleanly to BitPacked without regression.
        let ultra_sparse_gt = PrimitiveArray::from_option_iter((0..len).map(|_| {
            let d = rng.random_range(0..1000u32);
            if d < 210 {
                None
            } else if d < 995 {
                Some(0u64)
            } else {
                Some(1u64)
            }
        }))
        .into_array();
        let ultra_comp = with_rans.compress(&ultra_sparse_gt, &mut ctx)?;
        let ultra_bp = without_rans.compress(&ultra_sparse_gt, &mut ctx)?;
        assert_arrays_eq!(&ultra_comp, &ultra_sparse_gt, &mut ctx);
        assert!(
            ultra_comp.nbytes() <= ultra_bp.nbytes(),
            "Ultra-sparse GT ({}) must never regress vs BitPacked ({})",
            ultra_comp.nbytes(),
            ultra_bp.nbytes()
        );

        // 2. `statpopgen.PL`-shaped `[0, ~60, ~900]` i32 triples
        let pl = PrimitiveArray::from_iter((0..len).map(|i| match i % 3 {
            0 => 0i32,
            1 => 45 + rng.random_range(0..35i32),
            _ => 750 + rng.random_range(0..350i32),
        }))
        .into_array();

        let pl_rans = with_rans.compress(&pl, &mut ctx)?;
        let pl_bp = without_rans.compress(&pl, &mut ctx)?;
        assert_arrays_eq!(&pl_rans, &pl, &mut ctx);
        assert!(
            find_rans(&pl_rans).is_some(),
            "PL should select vortex.rans, got: {}",
            pl_rans.display_tree()
        );
        assert!(
            pl_rans.nbytes() * 100 < pl_bp.nbytes() * 78,
            "PL rANS ({}) must beat baseline ({}) by >22%",
            pl_rans.nbytes(),
            pl_bp.nbytes()
        );

        Ok(())
    }

    #[test]
    fn cascades_under_zipfian_dict_codes_and_skips_uniform_and_runend_ends() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .exclude_schemes([DeltaScheme::default().id()])
            .build();

        // 1. Zipfian low-cardinality dictionary column: `Dict.codes` should compress with `Rans`.
        let mut rng = StdRng::seed_from_u64(99);
        let dict_vals: Vec<i64> = (0..64).map(|k| 10_000_000 + k * 9973).collect();
        let zipf_input = PrimitiveArray::from_iter((0..32_768).map(|_| {
            let u = rng.random_range(0..1000u32);
            let code = if u < 750 {
                0
            } else if u < 880 {
                1
            } else if u < 940 {
                2
            } else {
                rng.random_range(3..64usize)
            };
            dict_vals[code]
        }))
        .into_array();
        let zipf_comp = compressor.compress(&zipf_input, &mut ctx)?;
        assert_arrays_eq!(&zipf_comp, &zipf_input, &mut ctx);
        if let Some(dict) = zipf_comp.as_opt::<Dict>() {
            assert!(
                find_rans(dict.codes()).is_some(),
                "Zipfian Dict.codes should select vortex.rans: {}",
                dict.codes().display_tree()
            );
        } else {
            assert!(find_rans(&zipf_comp).is_some());
        }

        // 2. Uniform 8-bit integers must NOT pick rANS over BitPacked.
        let uniform =
            PrimitiveArray::from_iter((0..32_768u32).map(|i| (i.wrapping_mul(2654435761)) & 0xFF))
                .into_array();
        let uniform_comp = compressor.compress(&uniform, &mut ctx)?;
        assert_arrays_eq!(&uniform_comp, &uniform, &mut ctx);
        assert!(
            find_rans(&uniform_comp).is_none(),
            "Uniform integers must stay on BitPacked, got: {}",
            uniform_comp.display_tree()
        );

        // 3. RunEnd.ends and BoolRunEnd.ends must never use rANS.
        let mut runs = Vec::with_capacity(40_000);
        for r in 0..2_000i32 {
            runs.extend(std::iter::repeat_n(r % 17, 20));
        }
        let run_input = PrimitiveArray::from_iter(runs).into_array();
        let run_comp = compressor.compress(&run_input, &mut ctx)?;
        assert_arrays_eq!(&run_comp, &run_input, &mut ctx);
        if let Some(re) = run_comp.as_opt::<RunEnd>() {
            assert!(
                find_rans(re.ends()).is_none(),
                "RunEnd.ends must exclude RansScheme: {}",
                re.ends().display_tree()
            );
        }

        let mut bool_bits = Vec::with_capacity(200_000);
        for r in 0..1_000 {
            bool_bits.extend(std::iter::repeat_n(r % 2 == 0, 200));
        }
        let bool_input = BoolArray::from(BitBuffer::from_iter(bool_bits)).into_array();
        let bool_comp = compressor.compress(&bool_input, &mut ctx)?;
        assert_arrays_eq!(&bool_comp, &bool_input, &mut ctx);
        if let Some(re) = bool_comp.as_opt::<RunEnd>() {
            assert!(
                find_rans(re.ends()).is_none(),
                "BoolRunEnd.ends must exclude RansScheme: {}",
                re.ends().display_tree()
            );
        }

        Ok(())
    }

    #[test]
    fn edition_gating_and_cuda_exclusion() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(55);
        let pl = PrimitiveArray::from_iter((0..20_000).map(|i| match i % 3 {
            0 => 0i32,
            1 => 50 + rng.random_range(0..20i32),
            _ => 800 + rng.random_range(0..200i32),
        }))
        .into_array();

        let core_session = session_with_editions(false);
        let mut core_ctx = core_session.create_execution_ctx();
        let core_comp = BtrBlocksCompressorBuilder::from_session(&core_session)
            .build()
            .compress(&pl, &mut core_ctx)?;
        assert_arrays_eq!(&core_comp, &pl, &mut core_ctx);
        assert!(
            find_rans(&core_comp).is_none(),
            "CORE edition must not emit vortex.rans: {}",
            core_comp.display_tree()
        );

        let unstable_session = session_with_editions(true);
        let mut unstable_ctx = unstable_session.create_execution_ctx();
        let unstable_comp = BtrBlocksCompressorBuilder::from_session(&unstable_session)
            .build()
            .compress(&pl, &mut unstable_ctx)?;
        assert_arrays_eq!(&unstable_comp, &pl, &mut unstable_ctx);
        assert!(
            find_rans(&unstable_comp).is_some(),
            "UNSTABLE edition should emit vortex.rans: {}",
            unstable_comp.display_tree()
        );

        let cuda_comp = BtrBlocksCompressorBuilder::from_session(&unstable_session)
            .only_cuda_compatible()
            .build()
            .compress(&pl, &mut unstable_ctx)?;
        assert_arrays_eq!(&cuda_comp, &pl, &mut unstable_ctx);
        assert!(
            find_rans(&cuda_comp).is_none(),
            "CUDA mode must exclude RansScheme even under UNSTABLE: {}",
            cuda_comp.display_tree()
        );

        Ok(())
    }
}
