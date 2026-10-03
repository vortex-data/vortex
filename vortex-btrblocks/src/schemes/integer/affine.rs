// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Affine integer encoding: per-chunk reference, common divisor and linear trend.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::Affine;
use vortex_fastlanes::AffineArray;
use vortex_fastlanes::AffineArrayExt;
use vortex_fastlanes::AffineArraySlotsExt;
use vortex_fastlanes::AffineOptions;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::affine_id;

use super::DeltaScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// The modes [`AffineScheme::auto`] chooses between, in tie-break order.
const MODES: [AffineOptions; 4] = [
    AffineOptions::FOR,
    AffineOptions::SCALE,
    AffineOptions::SLOPE,
    AffineOptions::ALL,
];

/// The number of whole chunks the estimate fits the model to.
const SAMPLE_CHUNKS: usize = 16;

/// The bits per value a mode must save over per-chunk FoR before Affine is worth its extra
/// decode work, the threshold Pco uses for its common-divisor mode.
const MIN_BITS_SAVED_PER_VALUE: f64 = 0.5;

/// Affine encoding: `value = encoded * scale + reference + ((slope * j) >> shift)` per
/// 1024-element chunk.
///
/// [`AffineScheme::auto`] fits every mode to a sample of whole chunks, cascades each through the
/// compressor, and keeps the smallest (see [`choose_mode`]). It is only selected when that mode
/// saves at least [`MIN_BITS_SAVED_PER_VALUE`] bits per value over per-chunk FoR, so FoR keeps
/// winning where a divisor or a trend does not help. [`AffineScheme::fixed`] always encodes with
/// one mode, for benchmarking.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AffineScheme {
    name: &'static str,
    fixed: Option<AffineOptions>,
}

impl AffineScheme {
    /// A scheme that picks the mode per array from a sample.
    pub const fn auto() -> Self {
        Self {
            name: "vortex.int.affine",
            fixed: None,
        }
    }

    /// A scheme that always encodes with `options`, under a scheme name distinct per mode.
    pub const fn fixed(options: AffineOptions) -> Self {
        let name = match (options.scale, options.slope, options.least_squares) {
            (_, _, true) => "vortex.int.affine.leco",
            (false, false, _) => "vortex.int.affine.for",
            (true, false, _) => "vortex.int.affine.scale",
            (false, true, _) => "vortex.int.affine.slope",
            (true, true, _) => "vortex.int.affine.all",
        };
        Self {
            name,
            fixed: Some(options),
        }
    }
}

impl Default for AffineScheme {
    fn default() -> Self {
        Self::auto()
    }
}

impl Scheme for AffineScheme {
    fn scheme_name(&self) -> &'static str {
        self.name
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![affine_id()]
    }

    /// Children: encoded=0, references=1, scales=2, slopes=3.
    fn num_children(&self) -> usize {
        4
    }

    /// Dict codes start at 0 and have no trend, as for FoR.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        [
            IntDictScheme.id(),
            FloatDictScheme.id(),
            StringDictScheme.id(),
            BinaryDictScheme.id(),
        ]
        .into_iter()
        .map(|ancestor| AncestorExclusion {
            ancestor,
            children: ChildSelection::One(1),
        })
        .collect()
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
        if self.fixed.is_some() {
            return CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse);
        }
        let scheme = *self;
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, _best_so_far, compress_ctx, exec_ctx| {
                let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
                let choice = choose_mode(&scheme, &primitive, compressor, &compress_ctx, exec_ctx)?;
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
        let options = match self.fixed {
            Some(options) => options,
            None => choose_mode(self, &primitive, compressor, &compress_ctx, exec_ctx)?.options,
        };
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

/// The mode with the smallest sampled size, with the sampled bits per value of it and of
/// per-chunk FoR.
#[derive(Debug, Clone, Copy)]
pub struct ModeChoice {
    /// The chosen mode.
    pub options: AffineOptions,
    /// Sampled bits per value of the chosen mode.
    pub bits: f64,
    /// Sampled bits per value of per-chunk FoR.
    pub for_bits: f64,
}

impl ModeChoice {
    /// Whether the chosen mode is worth its decode work over per-chunk FoR.
    pub fn is_worthwhile(&self) -> bool {
        self.options != AffineOptions::FOR && self.for_bits - self.bits >= MIN_BITS_SAVED_PER_VALUE
    }
}

/// Fit every mode to evenly spaced whole chunks of `array`, cascade each through `compressor`,
/// and keep the smallest.
///
/// The sample keeps whole chunks, so each mode's per-chunk model and its cascaded children are
/// measured exactly as they would be on the full array. Ties go to the later, richer mode: a
/// common divisor that costs nothing extra still leaves smaller residuals.
///
/// A slope serves the same sorted and trending data as Delta, so when the compressor may use
/// Delta, a slope is only kept if it is no larger than Delta on the same sample. Otherwise the
/// best mode without a slope is chosen, which Delta's own estimate then competes with.
pub fn choose_mode(
    scheme: &AffineScheme,
    array: &PrimitiveArray,
    compressor: &CascadingCompressor,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ModeChoice> {
    let sample = sample_chunks(array, exec_ctx)?;
    let len = sample.len() as f64;
    let mut for_bits = f64::INFINITY;
    let mut flat = (AffineOptions::FOR, f64::INFINITY);
    let mut sloped = (AffineOptions::SLOPE, f64::INFINITY);
    for options in MODES {
        let affine = Affine::encode(&sample, options, exec_ctx)?;
        let compressed = compress_children(scheme, &affine, compressor, compress_ctx, exec_ctx)?;
        let bits = compressed.nbytes() as f64 * 8.0 / len;
        if options == AffineOptions::FOR {
            for_bits = bits;
        }
        let best = if options.slope { &mut sloped } else { &mut flat };
        if bits <= best.1 {
            *best = (options, bits);
        }
    }

    let delta_id = DeltaScheme::default().id();
    let slope_wins = sloped.1 < flat.1
        && (!compressor.has_scheme(delta_id) || {
            let (bases, deltas) = vortex_fastlanes::delta_compress(&sample, exec_ctx)?;
            let mut bytes = 0;
            for (idx, child) in [bases, deltas].into_iter().enumerate() {
                bytes += compressor
                    .compress_child(&child.into_array(), compress_ctx, delta_id, idx, exec_ctx)?
                    .nbytes();
            }
            sloped.1 <= bytes as f64 * 8.0 / len
        });
    let (options, bits) = if slope_wins { sloped } else { flat };
    Ok(ModeChoice {
        options,
        bits,
        for_bits,
    })
}

/// Up to [`SAMPLE_CHUNKS`] evenly spaced chunks, kept aligned to chunk boundaries so that each
/// sampled chunk is fitted exactly as in the full array.
pub(crate) fn sample_chunks(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
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
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_fastlanes::Affine;
    use vortex_fastlanes::AffineArraySlotsExt;

    use super::AffineScheme;
    use crate::BtrBlocksCompressorBuilder;
    use crate::SESSION;

    static AFFINE: AffineScheme = AffineScheme::auto();

    /// Compress with the default schemes plus Affine and check the round trip.
    fn compress(values: PrimitiveArray) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .with_new_scheme(&AFFINE)
            .build();
        let input = values.into_array();
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(compressed)
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
        assert!(compressed.as_opt::<Affine>().is_none(), "{}", compressed.display_tree());
        Ok(())
    }
}
