// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Affine integer encoding: per-chunk reference, common divisor and linear trend.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::DecimalDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::Affine;
use vortex_fastlanes::AffineArray;
use vortex_fastlanes::AffineArrayExt;
use vortex_fastlanes::AffineArraySlotsExt;
use vortex_fastlanes::AffineModeEstimates;
use vortex_fastlanes::AffineOptions;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::affine_id;

use super::DeltaScheme;
use super::RunEndScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
use crate::schemes::boolean::BoolRunEndScheme;

/// The number of whole chunks the estimate fits the model to.
const SAMPLE_CHUNKS: usize = 16;

/// The bits per value a mode must save over per-chunk FoR before Affine is worth its extra
/// decode work, the threshold Pco uses for its common-divisor mode.
const MIN_BITS_SAVED_PER_VALUE: f64 = 0.5;

/// Affine encoding: `value = encoded * scale + reference + ((slope * j) >> shift)` per
/// 1024-element chunk.
///
/// The scheme pre-estimates all candidate modes (`FOR`, `SCALE`, `SLOPE`, `ALL`) directly in
/// memory from sampled residual bit-width histograms + outlier patch costs, rejecting early if no
/// non-`FOR` mode saves at least [`MIN_BITS_SAVED_PER_VALUE`] bits per value over per-chunk FoR.
/// Only the winning mode is cascaded through the compressor on the sample to measure its final
/// compression ratio.
///
/// Affine applies to every integer array the compressor reaches, including the integers that the
/// decimal scheme narrows decimals to and that ALP encodes floats as.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct AffineScheme;

impl Scheme for AffineScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.affine"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        // `Affine::encode` stores uniform per-chunk parameters as `ConstantArray`.
        vec![affine_id(), Constant.id()]
    }

    /// Children: encoded=0, references=1, scales=2, slopes=3.
    fn num_children(&self) -> usize {
        4
    }

    /// Exclude dictionary codes (which start at 0 with no trend) and `RunEnd.ends` /
    /// `BoolRunEnd.ends` (which are binary-searched via `scalar_at`, matching `FoR.v2`).
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        let mut exclusions: Vec<_> = [
            IntDictScheme.id(),
            FloatDictScheme.id(),
            StringDictScheme.id(),
            BinaryDictScheme.id(),
            DecimalDictScheme.id(),
            RunEndScheme.id(),
        ]
        .into_iter()
        .map(|ancestor| AncestorExclusion {
            ancestor,
            children: ChildSelection::One(1),
        })
        .collect();
        exclusions.push(AncestorExclusion {
            ancestor: BoolRunEndScheme.id(),
            children: ChildSelection::One(0),
        });
        exclusions
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // The residuals only shrink once a later layer packs them. Samples splice together short
        // runs from across the array, so per-chunk models fitted to them mean nothing.
        if compress_ctx.finished_cascading()
            || compress_ctx.is_sample()
            || data.array_len() < FL_CHUNK_SIZE
        {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let scheme = *self;
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, best_so_far, compress_ctx, exec_ctx| {
                if matches!(best_so_far, Some(EstimateScore::ZeroBytes)) {
                    return Ok(EstimateVerdict::Skip);
                }
                let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
                let Some(choice) =
                    choose_mode(&scheme, &primitive, compressor, &compress_ctx, exec_ctx)?
                else {
                    return Ok(EstimateVerdict::Skip);
                };
                if !choice.is_worthwhile() {
                    return Ok(EstimateVerdict::Skip);
                }
                let full_width = primitive.ptype().bit_width() as f64;
                Ok(EstimateVerdict::Ratio(full_width / choice.bits.max(1e-3)))
            },
        )))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let options =
            choose_mode_for_compress(self, &primitive, compressor, &compress_ctx, exec_ctx)?;
        let affine = Affine::encode(&primitive, options, exec_ctx)?;
        compress_children(self, &affine, compressor, &compress_ctx, exec_ctx)
    }
}

/// Cascade every non-constant child of `affine`.
fn compress_children(
    scheme: &AffineScheme,
    affine: &AffineArray,
    compressor: &CascadingCompressor,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut children = [
        affine.encoded().clone(),
        affine.references().clone(),
        affine.scales().clone(),
        affine.slopes().clone(),
    ];
    for (idx, child) in children.iter_mut().enumerate() {
        if child.as_constant().is_none() {
            *child = compressor.compress_child(child, compress_ctx, scheme.id(), idx, exec_ctx)?;
        }
    }
    let [encoded, references, scales, slopes] = children;
    Ok(Affine::try_new(
        encoded,
        references,
        scales,
        slopes,
        affine.offset(),
        affine.slope_shift(),
    )?
    .into_array())
}

/// The chosen non-`FOR` mode and its sampled bits per value alongside the per-chunk `FOR`
/// baseline.
#[derive(Debug, Clone, Copy)]
struct ModeChoice {
    /// The chosen mode (never [`AffineOptions::FOR`]).
    options: AffineOptions,
    /// Sampled cascaded bits per value of the chosen mode.
    bits: f64,
    /// Estimated bits per value of per-chunk FoR on the same sample.
    for_bits: f64,
}

impl ModeChoice {
    /// Whether the chosen mode is worth its decode work over per-chunk FoR.
    fn is_worthwhile(&self) -> bool {
        self.options != AffineOptions::FOR && self.for_bits - self.bits >= MIN_BITS_SAVED_PER_VALUE
    }
}

/// Pre-screened candidate modes from the fast in-memory residual estimator.
#[derive(Debug, Clone, Copy)]
struct PreEstimatePlan {
    est: AffineModeEstimates,
    scale_worthwhile: bool,
    best_sloped: Option<(AffineOptions, f64)>,
}

fn pre_estimate(
    array: &PrimitiveArray,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<PreEstimatePlan> {
    let est = Affine::estimate_modes(array, SAMPLE_CHUNKS, exec_ctx)?;
    let scale_worthwhile = est.for_bits - est.scale_bits >= MIN_BITS_SAVED_PER_VALUE;
    let (sloped_opts, sloped_bits) = if est.all_bits < est.slope_bits {
        (AffineOptions::ALL, est.all_bits)
    } else {
        (AffineOptions::SLOPE, est.slope_bits)
    };
    let best_sloped = (est.for_bits - sloped_bits >= MIN_BITS_SAVED_PER_VALUE)
        .then_some((sloped_opts, sloped_bits));
    Ok(PreEstimatePlan {
        est,
        scale_worthwhile,
        best_sloped,
    })
}

fn delta_allowed(
    compressor: &CascadingCompressor,
    compress_ctx: &CompressorContext,
) -> Option<SchemeId> {
    let delta_id = DeltaScheme::default().id();
    let active = compressor.has_scheme(delta_id)
        && !compress_ctx
            .cascade_history()
            .iter()
            .any(|&(id, _)| id == delta_id);
    active.then_some(delta_id)
}

fn measure_affine_bits(
    scheme: &AffineScheme,
    sample: &PrimitiveArray,
    options: AffineOptions,
    compressor: &CascadingCompressor,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<f64> {
    let affine = Affine::encode(sample, options, exec_ctx)?;
    let compressed = compress_children(scheme, &affine, compressor, compress_ctx, exec_ctx)?;
    // Floor at 4 scalars (one per child) so Affine never beats a pure 2-scalar `Sequence` when
    // all children collapse to `Constant`/`Sequence` with near-zero buffer bytes.
    let min_bytes = (4 * sample.ptype().byte_width()) as u64;
    Ok(compressed.nbytes().max(min_bytes) as f64 * 8.0 / sample.len() as f64)
}

fn measure_delta_bits(
    sample: &PrimitiveArray,
    compressor: &CascadingCompressor,
    compress_ctx: &CompressorContext,
    delta_id: SchemeId,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<f64> {
    let (bases, deltas) = vortex_fastlanes::delta_compress(sample, exec_ctx)?;
    let mut bytes = 0u64;
    for (idx, child) in [bases, deltas].into_iter().enumerate() {
        bytes += compressor
            .compress_child(&child.into_array(), compress_ctx, delta_id, idx, exec_ctx)?
            .nbytes();
    }
    Ok(bytes as f64 * 8.0 / sample.len() as f64)
}

/// Pre-estimate all modes in memory, reject early if estimated savings over per-chunk FoR are
/// below [`MIN_BITS_SAVED_PER_VALUE`], and cascade only the winning non-`FOR` mode on the sample.
///
/// A slope serves the same sorted and trending data as Delta, so when Delta is allowed, a sloped
/// mode is only kept if its cascaded sample size is no larger than Delta's on the same sample.
/// Otherwise `SCALE` is used if it independently saves at least [`MIN_BITS_SAVED_PER_VALUE`] bits
/// per value over per-chunk FoR.
fn choose_mode(
    scheme: &AffineScheme,
    array: &PrimitiveArray,
    compressor: &CascadingCompressor,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ModeChoice>> {
    let plan = pre_estimate(array, exec_ctx)?;
    if !plan.scale_worthwhile && plan.best_sloped.is_none() {
        return Ok(None);
    }

    let sample = sample_chunks(array, exec_ctx)?;
    let flat_bits = if plan.scale_worthwhile {
        plan.est.scale_bits
    } else {
        plan.est.for_bits
    };

    if let Some((sloped_opts, sloped_est)) = plan.best_sloped
        && sloped_est < flat_bits
    {
        let sloped_bits = measure_affine_bits(
            scheme,
            &sample,
            sloped_opts,
            compressor,
            compress_ctx,
            exec_ctx,
        )?;
        let slope_wins = match delta_allowed(compressor, compress_ctx) {
            None => true,
            Some(delta_id) => {
                sloped_bits
                    <= measure_delta_bits(&sample, compressor, compress_ctx, delta_id, exec_ctx)?
            }
        };
        if slope_wins {
            return Ok(Some(ModeChoice {
                options: sloped_opts,
                bits: sloped_bits,
                for_bits: plan.est.for_bits,
            }));
        }
    }

    if !plan.scale_worthwhile {
        return Ok(None);
    }

    let scale_bits = measure_affine_bits(
        scheme,
        &sample,
        AffineOptions::SCALE,
        compressor,
        compress_ctx,
        exec_ctx,
    )?;
    Ok(Some(ModeChoice {
        options: AffineOptions::SCALE,
        bits: scale_bits,
        for_bits: plan.est.for_bits,
    }))
}

/// Select the [`AffineOptions`] to encode with during [`AffineScheme::compress`], matching
/// [`choose_mode`] while avoiding redundant sample cascades whenever Delta arbitration is not
/// needed.
fn choose_mode_for_compress(
    scheme: &AffineScheme,
    array: &PrimitiveArray,
    compressor: &CascadingCompressor,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<AffineOptions> {
    let plan = pre_estimate(array, exec_ctx)?;
    let flat_bits = if plan.scale_worthwhile {
        plan.est.scale_bits
    } else {
        plan.est.for_bits
    };

    if let Some((sloped_opts, sloped_est)) = plan.best_sloped
        && sloped_est < flat_bits
    {
        let Some(delta_id) = delta_allowed(compressor, compress_ctx) else {
            return Ok(sloped_opts);
        };
        // Only arbitrate against Delta when both SCALE and a sloped mode are viable; if SCALE is
        // not viable, `choose_mode` already confirmed the sloped mode beat Delta.
        if !plan.scale_worthwhile {
            return Ok(sloped_opts);
        }
        let sample = sample_chunks(array, exec_ctx)?;
        let sloped_bits = measure_affine_bits(
            scheme,
            &sample,
            sloped_opts,
            compressor,
            compress_ctx,
            exec_ctx,
        )?;
        let delta_bits = measure_delta_bits(&sample, compressor, compress_ctx, delta_id, exec_ctx)?;
        if sloped_bits <= delta_bits {
            return Ok(sloped_opts);
        }
    }

    if plan.scale_worthwhile {
        Ok(AffineOptions::SCALE)
    } else if let Some((sloped_opts, _)) = plan.best_sloped {
        Ok(sloped_opts)
    } else {
        Ok(AffineOptions::SCALE)
    }
}

/// Up to [`SAMPLE_CHUNKS`] evenly spaced chunks, kept aligned to chunk boundaries so that each
/// sampled chunk is fitted exactly as in the full array.
fn sample_chunks(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let num_chunks = array.len().div_ceil(FL_CHUNK_SIZE);
    if num_chunks <= SAMPLE_CHUNKS {
        return Ok(array.clone());
    }
    let array = array.clone().into_array();
    let chunks = (0..SAMPLE_CHUNKS)
        .map(|i| {
            let chunk = i * num_chunks / SAMPLE_CHUNKS;
            let start = chunk * FL_CHUNK_SIZE;
            array.slice(start..(start + FL_CHUNK_SIZE).min(array.len()))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    ChunkedArray::try_new(chunks, array.dtype().clone())?
        .into_array()
        .execute::<PrimitiveArray>(ctx)
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::RunEnd;
    use vortex_array::arrays::runend::RunEndArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_edition::EDITION_DECLARATIONS;
    use vortex_edition::EDITION_FAMILIES;
    use vortex_edition::EditionSession;
    use vortex_edition::EditionSessionExt;
    use vortex_edition::declarations::core::CORE_2026_08_3;
    use vortex_edition::declarations::unstable::UNSTABLE;
    use vortex_error::VortexResult;
    use vortex_fastlanes::Affine;
    use vortex_fastlanes::AffineArraySlotsExt;
    use vortex_sequence::Sequence;
    use vortex_session::VortexSession;

    use crate::BtrBlocksCompressorBuilder;
    use crate::CompressionSession;
    use crate::SESSION;
    use crate::SchemeExt;
    use crate::schemes::integer::DeltaScheme;

    /// Compress with the default schemes (including Affine) and check the round trip.
    fn compress(values: impl IntoArray) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();
        let input = values.into_array();
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(compressed)
    }

    /// The first Affine array anywhere in the compressed tree.
    fn find_affine(compressed: &ArrayRef) -> Option<ArrayRef> {
        compressed
            .depth_first_traversal()
            .find(|array| array.is::<Affine>())
    }

    /// Prices on a 0.05 tick around a level, in cents, so every value is a multiple of 5 and has
    /// no trend for Delta to exploit.
    fn nickel_cents() -> impl Iterator<Item = i64> {
        let mut rng = StdRng::seed_from_u64(11);
        (0..20_000).map(move |_| 10_000 + 5 * rng.random_range(0..1000i64))
    }

    /// Decimals narrow to integers in the decimal scheme, which then cascade into Affine.
    #[test]
    fn divisor_under_decimal() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimals = DecimalArray::new(
            nickel_cents().collect::<Buffer<i64>>(),
            DecimalDType::new(12, 2),
            Validity::NonNullable,
            &mut ctx,
        );
        let compressed = compress(decimals)?;
        let affine = find_affine(&compressed).unwrap_or_else(|| {
            panic!(
                "Affine should encode the decimal's integers: {}",
                compressed.display_tree()
            )
        });
        let affine = affine.as_::<Affine>();
        assert!(!is_unit(affine.scales(), 1));
        Ok(())
    }

    /// ALP turns floats on a decimal grid into integers, which then cascade into Affine.
    #[test]
    fn divisor_under_alp() -> VortexResult<()> {
        let floats = PrimitiveArray::from_iter(nickel_cents().map(|cents| cents as f64 / 100.0));
        let compressed = compress(floats)?;
        let affine = find_affine(&compressed).unwrap_or_else(|| {
            panic!(
                "Affine should encode ALP's integers: {}",
                compressed.display_tree()
            )
        });
        let affine = affine.as_::<Affine>();
        assert!(!is_unit(affine.scales(), 1));
        Ok(())
    }

    fn is_unit(param: &ArrayRef, unit: i64) -> bool {
        param
            .as_constant()
            .and_then(|s| s.as_primitive().as_::<i64>())
            .is_some_and(|v| v == unit)
    }

    /// Prices on a 0.01 tick in units of 1e-8, which only a common divisor narrows.
    fn tick_prices() -> PrimitiveArray {
        let mut rng = StdRng::seed_from_u64(7);
        let mut price = 6_776_562i64;
        PrimitiveArray::from_iter((0..20_000).map(|_| {
            price += rng.random_range(-40..=40);
            price * 1_000_000
        }))
    }

    /// Nanosecond timestamps on a one-minute grid with gaps, which a slope and a divisor fit.
    fn gappy_grid() -> PrimitiveArray {
        PrimitiveArray::from_iter(
            (0..30_000i64)
                .filter(|i| i % 97 != 5)
                .map(|i| 1_700_000_000_000_000_000 + i * 60_000_000_000),
        )
    }

    #[test]
    fn divisor_on_tick_prices() -> VortexResult<()> {
        let compressed = compress(tick_prices())?;
        let affine = compressed
            .as_opt::<Affine>()
            .expect("Affine should win on tick prices");
        assert!(!is_unit(affine.scales(), 1));
        assert!(is_unit(affine.slopes(), 0));
        Ok(())
    }

    #[test]
    fn slope_and_divisor_on_gappy_grid() -> VortexResult<()> {
        let compressed = compress(gappy_grid())?;
        let affine = compressed
            .as_opt::<Affine>()
            .expect("Affine should win on a gappy grid");
        assert!(!is_unit(affine.slopes(), 0));
        assert!(compressed.nbytes() * 8 < 30_000);
        Ok(())
    }

    /// Random values and heavy-tailed sorted keys have no divisor or trend to exploit, and the
    /// latter suit Delta better than a slope.
    #[rstest]
    #[case::random(PrimitiveArray::from_iter(
        (0..20_000u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20)
    ))]
    #[case::heavy_tailed_keys({
        let mut rng = StdRng::seed_from_u64(3);
        let mut key = 0u64;
        PrimitiveArray::from_iter((0..20_000).map(|_| {
            key += if rng.random_ratio(1, 50) { rng.random_range(1..400_000) } else { rng.random_range(1..60) };
            key
        }))
    })]
    fn not_chosen(#[case] values: PrimitiveArray) -> VortexResult<()> {
        let compressed = compress(values)?;
        assert!(
            compressed.as_opt::<Affine>().is_none(),
            "{}",
            compressed.display_tree()
        );
        Ok(())
    }

    /// Affine must never wrap `RunEnd.ends`, even with Delta disabled, so binary-search
    /// `scalar_at` on run-end encoded columns stays fast.
    #[test]
    fn excluded_on_runend_ends() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = StdRng::seed_from_u64(19);
        let mut values = Vec::with_capacity(40_000);
        for run in 0..2_500i32 {
            let run_len = rng.random_range(12..=20);
            values.extend(std::iter::repeat_n(run * 7, run_len));
        }
        let input = PrimitiveArray::from_iter(values).into_array();
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .exclude_schemes([DeltaScheme::default().id()])
            .build();
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert_arrays_eq!(compressed, input, &mut ctx);
        let runend = compressed
            .as_opt::<RunEnd>()
            .expect("repeated runs should compress with RunEnd");
        assert!(
            find_affine(runend.ends()).is_none(),
            "Affine must not encode RunEnd.ends: {}",
            runend.ends().display_tree()
        );
        Ok(())
    }

    /// Affine must also stay off `BoolRunEnd.ends` (`child 0`) and never wrap a pure `Sequence`.
    #[test]
    fn excluded_on_bool_runend_ends_and_pure_sequence() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = StdRng::seed_from_u64(23);
        let mut bits = Vec::with_capacity(400_000);
        let mut current = false;
        for _ in 0..2_500 {
            let run_len = rng.random_range(120..=200);
            bits.extend(std::iter::repeat_n(current, run_len));
            current = !current;
        }
        let bool_input = BoolArray::from(BitBuffer::from_iter(bits)).into_array();
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .exclude_schemes([DeltaScheme::default().id()])
            .build();
        let compressed_bool = compressor.compress(&bool_input, &mut ctx)?;
        assert_arrays_eq!(compressed_bool, bool_input, &mut ctx);
        let runend = compressed_bool
            .as_opt::<RunEnd>()
            .expect("long bool runs should compress with RunEnd");
        assert!(
            find_affine(runend.ends()).is_none(),
            "Affine must not encode BoolRunEnd.ends: {}",
            runend.ends().display_tree()
        );

        let seq_input = PrimitiveArray::from_iter(1..=10_000i64).into_array();
        let compressed_seq = compressor.compress(&seq_input, &mut ctx)?;
        assert_arrays_eq!(compressed_seq, seq_input, &mut ctx);
        assert!(
            compressed_seq.is::<Sequence>(),
            "pure sequence should stay Sequence, got {}",
            compressed_seq.display_tree()
        );
        Ok(())
    }

    fn session_with_editions(enable_unstable: bool) -> VortexSession {
        let session = VortexSession::empty()
            .with::<EditionSession>()
            .with::<CompressionSession>();
        for family in EDITION_FAMILIES {
            session
                .editions()
                .declare_family(family)
                .expect("first-party edition family");
        }
        for declaration in EDITION_DECLARATIONS {
            session
                .register_edition(declaration)
                .expect("first-party edition");
        }
        session
            .enable_edition(CORE_2026_08_3)
            .expect("core edition is registered");
        if enable_unstable {
            session
                .enable_edition(UNSTABLE)
                .expect("unstable edition is registered");
        }
        vortex_fastlanes::initialize(&session);
        session
    }

    /// `CORE` excludes `fastlanes.affine`, `UNSTABLE` enables it by default, and CUDA mode
    /// excludes it even when `UNSTABLE` is enabled.
    #[test]
    fn edition_gating_and_cuda_exclusion() -> VortexResult<()> {
        let input = tick_prices().into_array();

        let core_session = session_with_editions(false);
        let mut core_ctx = core_session.create_execution_ctx();
        let core_compressed = BtrBlocksCompressorBuilder::from_session(&core_session)
            .build()
            .compress(&input, &mut core_ctx)?;
        assert_arrays_eq!(core_compressed, input, &mut core_ctx);
        assert!(
            find_affine(&core_compressed).is_none(),
            "CORE edition must not emit fastlanes.affine: {}",
            core_compressed.display_tree()
        );

        let unstable_session = session_with_editions(true);
        let mut unstable_ctx = unstable_session.create_execution_ctx();
        let unstable_compressed = BtrBlocksCompressorBuilder::from_session(&unstable_session)
            .build()
            .compress(&input, &mut unstable_ctx)?;
        assert_arrays_eq!(unstable_compressed, input, &mut unstable_ctx);
        assert!(
            unstable_compressed.is::<Affine>(),
            "UNSTABLE edition should emit fastlanes.affine by default: {}",
            unstable_compressed.display_tree()
        );

        let cuda_compressed = BtrBlocksCompressorBuilder::from_session(&unstable_session)
            .only_cuda_compatible()
            .build()
            .compress(&input, &mut unstable_ctx)?;
        assert_arrays_eq!(cuda_compressed, input, &mut unstable_ctx);
        assert!(
            find_affine(&cuda_compressed).is_none(),
            "CUDA mode must exclude AffineScheme even under UNSTABLE: {}",
            cuda_compressed.display_tree()
        );
        Ok(())
    }
}
