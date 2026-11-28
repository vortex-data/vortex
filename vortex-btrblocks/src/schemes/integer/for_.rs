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
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::FoRReferences;
use vortex_fastlanes::bitpacked_v2_id;
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FoRSchemeMode {
    V1,
    V2,
}

/// The FoR scheme that only produces single-reference arrays.
pub(crate) static FOR_V1: FoRScheme = FoRScheme::v1();

/// The FoR scheme that may produce one reference per 1024-element chunk.
pub(crate) static FOR_V2: FoRScheme = FoRScheme::v2();

/// [`FOR_V1`] with a child that may give each 1024-element block its own bit width.
static FOR_V1_BLOCKED: FoRScheme = FoRScheme {
    mode: FoRSchemeMode::V1,
    bitpacking: BitPackingScheme::v2(),
};

/// [`FOR_V2`] with a child that may give each 1024-element block its own bit width.
static FOR_V2_BLOCKED: FoRScheme = FoRScheme {
    mode: FoRSchemeMode::V2,
    bitpacking: BitPackingScheme::v2(),
};

/// Frame of Reference encoding.
///
/// The v1 mode subtracts one reference, the array's minimum. The v2 mode subtracts one reference
/// per 1024-element chunk, the chunk's minimum. Arrays with a single reference, including v2
/// arrays whose references compress to a constant, serialize as `fastlanes.for`, while per-chunk
/// references serialize as `fastlanes.for.v2`.
///
/// The encoded child is always bit-packed by a [`BitPackingScheme`] called directly, not chosen by
/// the compressor, so this scheme declares the child's serialized IDs as its own. Excluding the
/// BitPacking scheme does not stop FoR from bit-packing; disallowing `fastlanes.bitpacked` does.
///
/// The default uses v1 for both. [`refine`](Scheme::refine) picks v2 references when
/// `fastlanes.for.v2` is allowed, and a v2 child when `fastlanes.bitpacked.v2` is allowed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FoRScheme {
    mode: FoRSchemeMode,
    bitpacking: BitPackingScheme,
}

impl FoRScheme {
    /// Creates a FoR scheme configured for v1, which uses a single reference.
    pub const fn v1() -> Self {
        Self {
            mode: FoRSchemeMode::V1,
            bitpacking: BitPackingScheme::v1(),
        }
    }

    /// Creates a FoR scheme configured for v2, which may use one reference per chunk.
    pub const fn v2() -> Self {
        Self {
            mode: FoRSchemeMode::V2,
            bitpacking: BitPackingScheme::v1(),
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
        encodings.extend(self.bitpacking.produced_encodings());
        encodings
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        match (allowed(&for_v2_id()), allowed(&bitpacked_v2_id())) {
            (false, false) => &FOR_V1,
            (false, true) => &FOR_V1_BLOCKED,
            (true, false) => &FOR_V2,
            (true, true) => &FOR_V2_BLOCKED,
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
        // V1 fuses single-reference FoR with BitPacking directly inside `encode` and needs no
        // further cascade step; V2 registers per-chunk references as a child to compress.
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

        // Defer to sampling so the estimate reflects what `encode` actually produces: patch-aware
        // bit widths (`find_best_bit_width`), `BitPackingScheme`'s `MIN_BITS_SAVED` threshold,
        // FastLanes block padding, and V2 per-chunk references/widths.
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
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
        let Some(packed) = self.bitpacking.try_encode(&biased_data, exec_ctx)? else {
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
        // The bit-packed children are compressed in our own context, not a leaf one, so a v2
        // child's block offsets can cascade.
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
    use vortex_fastlanes::bitpacked_v2_id;
    use vortex_fastlanes::for_v1_id;
    use vortex_fastlanes::for_v2_id;
    use vortex_sequence::Sequence;

    use super::BitPackingScheme;
    use super::FOR_V1;
    use super::FOR_V2;
    use crate::ArrayAndStats;
    use crate::BtrBlocksCompressorBuilder;
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
        expected.extend(BitPackingScheme::v1().produced_encodings());
        assert_eq!(refined.produced_encodings(), expected);
    }

    /// Both variants give the child per-block bit widths exactly when `fastlanes.bitpacked.v2` is
    /// allowed.
    #[rstest]
    fn refine_picks_blocked_child_when_bitpacked_v2_is_allowed(
        #[values(false, true)] allow_v2: bool,
        #[values(&FOR_V1, &FOR_V2)] scheme: &'static dyn Scheme,
    ) {
        let refined = scheme.refine(&|id: &ArrayId| allow_v2 && *id == bitpacked_v2_id());
        assert_eq!(
            refined.produced_encodings().contains(&bitpacked_v2_id()),
            allow_v2
        );
    }

    #[test]
    fn test_for_skips_when_saving_fewer_than_min_bits_saved() -> VortexResult<()> {
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

        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&FOR_V1)
            .with_new_scheme(&IntDictScheme)
            .with_new_scheme(&super::super::BITPACKING_V1)
            .build();
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

    #[test]
    fn test_for_selected_for_negative_values_with_sparse_outliers() -> VortexResult<()> {
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

        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&FOR_V1)
            .with_new_scheme(&super::super::BITPACKING_V1)
            .build();
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
    fn test_for_prices_fastlanes_padding_on_short_arrays() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 10 u64 values take 80 bytes uncompressed, whereas a 1024-padded 4-bit FastLanes block
        // takes 512 + 8 bytes. Sequence must win instead of FoR outscoring it and failing size.
        let input = PrimitiveArray::from_iter(1000u64..1010).into_array();

        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&FOR_V1)
            .with_new_scheme(&SequenceScheme)
            .build();
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
