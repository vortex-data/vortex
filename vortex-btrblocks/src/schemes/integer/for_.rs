// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame of Reference integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::DecimalDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::sample_aligned_chunks;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::FoRReferences;
use vortex_fastlanes::for_v1_id;
use vortex_fastlanes::for_v2_id;

use super::BitPackingScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;
use crate::SchemeExt;

/// Number of aligned FastLanes chunks sampled when estimating V1 FoR compression.
const SAMPLE_CHUNKS: u32 = 4;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FoRSchemeMode {
    V1,
    V2,
}

/// The FoR scheme that only produces single-reference arrays.
pub(crate) static FOR_V1: FoRScheme = FoRScheme::v1();

/// The FoR scheme that may produce one reference per 1024-element chunk.
pub(crate) static FOR_V2: FoRScheme = FoRScheme::v2();

/// Frame of Reference encoding.
///
/// The v1 mode subtracts one reference, the array's minimum. The v2 mode subtracts one reference
/// per 1024-element chunk, the chunk's minimum. Arrays with a single reference, including v2
/// arrays whose references compress to a constant, serialize as `fastlanes.for`, while per-chunk
/// references serialize as `fastlanes.for.v2`.
///
/// The encoded child is always bit-packed by [`BitPackingScheme`] called directly, not chosen by
/// the compressor, so this scheme declares the child's serialized IDs as its own. Excluding the
/// BitPacking scheme does not stop FoR from bit-packing; disallowing `fastlanes.bitpacked` does.
///
/// The default uses v1. [`refine`](Scheme::refine) picks v2 references when `fastlanes.for.v2` is
/// allowed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FoRScheme {
    mode: FoRSchemeMode,
}

impl FoRScheme {
    /// Creates a FoR scheme configured for v1, which uses a single reference.
    pub const fn v1() -> Self {
        Self {
            mode: FoRSchemeMode::V1,
        }
    }

    /// Creates a FoR scheme configured for v2, which may use one reference per chunk.
    pub const fn v2() -> Self {
        Self {
            mode: FoRSchemeMode::V2,
        }
    }
}

impl Default for FoRScheme {
    fn default() -> Self {
        Self::v1()
    }
}

impl Scheme for FoRScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.for"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        let mut encodings = match self.mode {
            FoRSchemeMode::V1 => vec![for_v1_id()],
            FoRSchemeMode::V2 => vec![for_v1_id(), for_v2_id()],
        };
        encodings.extend(BitPackingScheme.produced_encodings());
        encodings
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        if allowed(&for_v2_id()) {
            &FOR_V2
        } else {
            &FOR_V1
        }
    }

    /// Children: references=0 in v2 mode. The encoded child is always bit-packed.
    fn num_children(&self) -> usize {
        match self.mode {
            FoRSchemeMode::V1 => 0,
            FoRSchemeMode::V2 => 1,
        }
    }

    /// Dict codes always start at 0, so FoR (which subtracts the min) is a no-op.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: BinaryDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: DecimalDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // V1 fuses single-reference FoR with BitPacking directly inside `encode`; V2 registers
        // per-chunk references as a child to compress.
        if self.mode == FoRSchemeMode::V2 && compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Fast-skip checks use global stats and are conservative for v2: arrays whose global
        // minimum is zero, or whose single-reference range width is no narrower than plain
        // BitPacking, are skipped before sampling even if per-chunk references would help.
        let stats = data.integer_stats(exec_ctx);

        // Only apply when the min is not already zero.
        if stats.erased().min_is_zero() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Difference between max and min.
        let for_bitwidth = match stats.erased().max_minus_min().checked_ilog2() {
            Some(l) => l + 1,
            // If max-min == 0, then we should be compressing this as a constant array.
            None => return CompressionEstimate::Verdict(EstimateVerdict::Skip),
        };

        // If BitPacking can be applied (only non-negative values) and FoR doesn't reduce bit width
        // compared to BitPacking, don't use FoR since it has a small amount of overhead (storing
        // the reference) for effectively no benefits.
        if let Some(max_log) = stats
            .erased()
            .max_ilog2()
            // Only skip FoR when min >= 0, otherwise BitPacking can't be applied without ZigZag.
            .filter(|_| !stats.erased().min_is_negative())
        {
            let bitpack_bitwidth = max_log + 1;
            if for_bitwidth >= bitpack_bitwidth {
                return CompressionEstimate::Verdict(EstimateVerdict::Skip);
            }
        }

        // Signed residuals wrap negative (which BitPacking rejects) when the range reaches the
        // sign bit.
        let ptype = data.array_as_primitive().ptype();
        if ptype.is_signed_int() && for_bitwidth as usize >= ptype.bit_width() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        match self.mode {
            // V2 uses per-chunk references, so sampling via `encode` directly models per-chunk
            // references, child compression, padding, and `MIN_BITS_SAVED`.
            FoRSchemeMode::V2 => CompressionEstimate::Deferred(DeferredEstimate::Sample),
            // V1 subtracts the full array's global minimum from every row. Biasing the sample
            // against `global_min` avoids overestimating compression when a sample misses a rare
            // low outlier, while still measuring patch-aware bit widths, `MIN_BITS_SAVED`, and
            // FastLanes block padding.
            FoRSchemeMode::V1 => {
                let global_min = stats.erased().min_scalar();
                let full_width = ptype.bit_width() as f64;
                CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                    move |_compressor, data, _best_so_far, ctx, exec_ctx| {
                        let sampled = if ctx.is_sample() {
                            None
                        } else {
                            sample_aligned_chunks(data.array(), FL_CHUNK_SIZE, SAMPLE_CHUNKS)
                        };
                        let sample_arr = sampled.unwrap_or_else(|| data.array().clone());
                        if sample_arr.all_invalid(exec_ctx)? {
                            return Ok(EstimateVerdict::Skip);
                        }

                        // Measure `nbytes` after materializing the sample into a contiguous
                        // `PrimitiveArray`; slices in a `ChunkedArray` still reference the full
                        // parent buffer.
                        let sample_primitive = sample_arr.execute::<PrimitiveArray>(exec_ctx)?;
                        let before_nbytes = sample_primitive.nbytes();
                        if before_nbytes == 0 {
                            return Ok(EstimateVerdict::Skip);
                        }
                        let for_array =
                            FoR::encode_with_reference(sample_primitive, global_min, exec_ctx)?;
                        let biased = for_array
                            .encoded()
                            .clone()
                            .execute::<PrimitiveArray>(exec_ctx)?;
                        let Some(packed) = BitPackingScheme.compress_array(&biased, exec_ctx)?
                        else {
                            return Ok(EstimateVerdict::Skip);
                        };

                        let after_nbytes = packed.nbytes();
                        let ratio = if after_nbytes == 0 {
                            // Every sampled valid value equaled `global_min` (`bit_width == 0`),
                            // while the full column is non-constant; fall back to the global-range
                            // ratio instead of disqualifying FoR with a zero-byte sample.
                            full_width / for_bitwidth as f64
                        } else {
                            before_nbytes as f64 / after_nbytes as f64
                        };
                        if ratio <= 1.0 {
                            return Ok(EstimateVerdict::Skip);
                        }
                        Ok(EstimateVerdict::Ratio(ratio))
                    },
                )))
            }
        }
    }

    /// Children: references=0 in v2 mode. The encoded child is bit-packed, and its children are
    /// the bit-packing scheme's.
    fn encode(
        &self,
        data: &ArrayAndStats,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let for_array = match self.mode {
            FoRSchemeMode::V1 => FoR::encode(primitive, exec_ctx)?,
            FoRSchemeMode::V2 => FoR::encode_chunked(primitive, exec_ctx)?,
        };
        let biased = for_array
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;

        // Immediately bitpack. If any other scheme was preferable, it would be chosen instead
        // of bitpacking.
        // NOTE: we could delegate in the future if we had another downstream codec that performs
        //  as well.
        let biased_data =
            ArrayAndStats::new(biased.into_array(), compress_ctx.merged_stats_options());
        let Some(packed) = BitPackingScheme.try_encode(&biased_data, exec_ctx)? else {
            // Unpacked residuals are no smaller than the input, so hand it back for the size
            // check to reject.
            return Ok(Encoded::leaf(data.array().clone()));
        };

        let for_stats = for_array.statistics().to_owned();
        let offset = for_array.offset();
        let (compress, reference) = match for_array.references() {
            FoRReferences::Global(reference) => (Vec::new(), Some(reference.clone())),
            FoRReferences::Blocked(references) => {
                (vec![Child::new(self, 0, references.clone())], None)
            }
        };
        Ok(
            packed.wrap(compress, Vec::new(), move |packed, children, _| {
                // TODO(connor): This should really be `new_unchecked`.
                let for_compressed = match reference {
                    Some(reference) => FoR::try_new(packed, reference)?,
                    None => {
                        let [references] = children.compressed()?;
                        FoR::try_new_chunked(packed, references, offset)?
                    }
                };
                for_compressed.as_ref().statistics().set_all(for_stats);
                Ok(for_compressed.into_array())
            }),
        )
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_compressor::scheme::CompressionEstimate;
    use vortex_compressor::scheme::EstimateVerdict;
    use vortex_error::VortexResult;
    use vortex_fastlanes::FoR;
    use vortex_fastlanes::for_v1_id;
    use vortex_fastlanes::for_v2_id;
    use vortex_sequence::Sequence;

    use super::BitPackingScheme;
    use super::FOR_V1;
    use super::FOR_V2;
    use crate::ArrayAndStats;
    use crate::CascadingCompressor;
    use crate::Child;
    use crate::CompressorContext;
    use crate::Encoded;
    use crate::SESSION;
    use crate::Scheme;
    use crate::SchemeExt;
    use crate::schemes::integer::IntDictScheme;
    use crate::schemes::integer::SequenceScheme;

    /// Pass-through test scheme that consumes one cascade level before delegating to child 0.
    #[derive(Debug)]
    struct CascadeBurner(&'static str);

    impl Scheme for CascadeBurner {
        fn scheme_name(&self) -> &'static str {
            self.0
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            canonical.dtype().is_int()
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            Vec::new()
        }

        fn num_children(&self) -> usize {
            1
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            if compress_ctx.finished_cascading() {
                CompressionEstimate::Verdict(EstimateVerdict::Skip)
            } else {
                CompressionEstimate::Verdict(EstimateVerdict::Ratio(1_000_000.0))
            }
        }

        fn encode(
            &self,
            data: &ArrayAndStats,
            _compress_ctx: &CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<Encoded> {
            Ok(Encoded::new(
                vec![Child::new(self, 0, data.array().clone())],
                Vec::new(),
                |children, _| {
                    let [inner] = children.compressed()?;
                    Ok(inner)
                },
            ))
        }

        #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unimplemented!()
        }
    }

    static BURN_1: CascadeBurner = CascadeBurner("vortex.test.for.burn.1");
    static BURN_2: CascadeBurner = CascadeBurner("vortex.test.for.burn.2");
    static BURN_3: CascadeBurner = CascadeBurner("vortex.test.for.burn.3");

    /// Both variants refine to v2 exactly when the v2 ID is allowed.
    #[rstest]
    #[case::neither(false, false, false)]
    #[case::v1(true, false, false)]
    #[case::v2_only(false, true, true)]
    #[case::both(true, true, true)]
    fn refine_picks_v2_when_v2_id_is_allowed(
        #[case] allow_v1: bool,
        #[case] allow_v2: bool,
        #[case] expect_v2: bool,
        #[values(&FOR_V1, &FOR_V2)] scheme: &'static dyn Scheme,
    ) {
        let allowed =
            |id: &ArrayId| (allow_v1 && *id == for_v1_id()) || (allow_v2 && *id == for_v2_id());
        let refined = scheme.refine(&allowed);
        assert_eq!(refined.id(), scheme.id());
        let mut expected = if expect_v2 {
            vec![for_v1_id(), for_v2_id()]
        } else {
            vec![for_v1_id()]
        };
        expected.extend(BitPackingScheme.produced_encodings());
        assert_eq!(refined.produced_encodings(), expected);
    }
    #[rstest]
    fn test_for_skips_when_saving_fewer_than_min_bits_saved(
        #[values(&FOR_V1, &FOR_V2)] for_scheme: &'static dyn Scheme,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 64 distinct u8 values split across 1..=32 and 97..=128: biased span is 127 (7 bits,
        // saving only 1 bit on u8) and half the values require bit 6, so BitPacking refuses to
        // pack (< MIN_BITS_SAVED). FoR must skip instead of outscoring IntDict (~1.12x vs 8/7)
        // and falling back to uncompressed data.
        let values: Vec<u8> = (0..4_096)
            .map(|i| {
                let idx = (i % 64) as u8;
                if idx < 32 { idx + 1 } else { idx + 65 }
            })
            .collect();
        let input = PrimitiveArray::from_iter(values).into_array();

        let compressor =
            CascadingCompressor::new(vec![for_scheme, &IntDictScheme, &BitPackingScheme]);
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            !compressed.is::<FoR>() && !compressed.is_canonical(),
            "expected IntDict rather than FoR/uncompressed, got {}",
            compressed.encoding_id()
        );
        assert!(compressed.nbytes() < input.nbytes());
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[rstest]
    fn test_for_selected_for_negative_values_with_sparse_outliers(
        #[values(&FOR_V1, &FOR_V2)] for_scheme: &'static dyn Scheme,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Mostly -100..=-85 with sparse outliers at 15: global range is 115 (7 bits, saving only
        // 1 bit on i8), but after subtracting -100 almost all residuals fit in 4 bits and the
        // outliers move to patches. BitPacking skips negative values, so FoR must still win.
        let values: Vec<i8> = (0..4_096)
            .map(|i| {
                if i % 512 == 0 {
                    15
                } else {
                    -100 + (i % 16) as i8
                }
            })
            .collect();
        let input = PrimitiveArray::from_iter(values).into_array();

        let compressor = CascadingCompressor::new(vec![for_scheme, &BitPackingScheme]);
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            compressed.is::<FoR>(),
            "expected FoR for negative i8 values with sparse outliers, got {}",
            compressed.encoding_id()
        );
        assert!(compressed.nbytes() < input.nbytes());
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_for_v1_does_not_overestimate_with_rare_low_sentinel() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 100,000 i32 rows mostly in 1_000_000..1_000_016 with 10 negative sentinel rows at -5.
        // A sample that misses the sentinels would see 4-bit residuals (~8x) if biased by the
        // sample's own minimum, whereas real V1 encode subtracts -5 and packs 20 bits (~1.6x).
        // Biasing the sample by the global minimum keeps IntDict ahead of FOR_V1.
        let values: Vec<i32> = (0..100_000i32)
            .map(|i| if i < 10 { -5 } else { 1_000_000 + (i % 16) })
            .collect();
        let input = PrimitiveArray::from_iter(values).into_array();

        let compressor = CascadingCompressor::new(vec![&FOR_V1, &IntDictScheme, &BitPackingScheme]);
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            !compressed.is::<FoR>() && !compressed.is_canonical(),
            "expected IntDict rather than FoR for rare low sentinel column, got {}",
            compressed.encoding_id()
        );
        assert!(compressed.nbytes() < input.nbytes());
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[rstest]
    fn test_for_prices_fastlanes_padding_on_short_arrays(
        #[values(&FOR_V1, &FOR_V2)] for_scheme: &'static dyn Scheme,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 10 u64 values take 80 bytes uncompressed, whereas a 1024-padded 4-bit FastLanes block
        // takes 512 + 8 bytes. Sequence must win instead of FoR outscoring it and failing size.
        let input = PrimitiveArray::from_iter(1000u64..1010).into_array();

        let compressor = CascadingCompressor::new(vec![for_scheme, &SequenceScheme]);
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            compressed.is::<Sequence>(),
            "expected Sequence, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_for_v1_succeeds_at_exhausted_cascade_depth() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input =
            PrimitiveArray::from_iter((0..2_048u64).map(|i| 1_000_000 + (i % 16))).into_array();

        // Three burners exhaust `MAX_CASCADE` (3), leaving `allowed_cascading == 0` for FoR.
        // Use `CascadingCompressor::new` directly so `FOR_V1` is not refined to `FOR_V2`.
        let v1_compressor = CascadingCompressor::new(vec![&BURN_1, &BURN_2, &BURN_3, &FOR_V1]);
        let v1_compressed = v1_compressor.compress(&input, &mut ctx)?;
        assert!(
            v1_compressed.is::<FoR>(),
            "expected FoR v1 at exhausted cascade depth, got {}",
            v1_compressed.encoding_id()
        );
        assert!(v1_compressed.nbytes() < input.nbytes() / 4);
        assert_arrays_eq!(v1_compressed, input, &mut ctx);

        // V2 still skips at exhausted cascade depth because its per-chunk references child needs
        // cascading.
        let v2_compressor = CascadingCompressor::new(vec![&BURN_1, &BURN_2, &BURN_3, &FOR_V2]);
        let v2_compressed = v2_compressor.compress(&input, &mut ctx)?;
        assert!(
            !v2_compressed.is::<FoR>(),
            "expected FoR v2 to skip at exhausted cascade depth"
        );
        assert_arrays_eq!(v2_compressed, input, &mut ctx);
        Ok(())
    }
}
