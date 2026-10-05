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
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::for_v1_id;
use vortex_fastlanes::for_v2_id;

use super::BitPackingScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
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

/// Frame of Reference encoding.
///
/// The v1 mode subtracts one reference, the array's minimum. The v2 mode subtracts one reference
/// per 1024-element chunk, the chunk's minimum. Arrays with a single reference, including v2
/// arrays whose references compress to a constant, serialize as `fastlanes.for`, while per-chunk
/// references serialize as `fastlanes.for.v2`.
///
/// The default uses v1. [`refine`](Scheme::refine) picks v2 when the v2 ID is allowed and v1
/// otherwise.
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
        match self.mode {
            FoRSchemeMode::V1 => vec![for_v1_id()],
            FoRSchemeMode::V2 => vec![for_v1_id(), for_v2_id()],
        }
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
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // FoR only subtracts the min. Without further compression (e.g. BitPacking), the output is
        // the same size.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Both modes estimate single-reference FoR. This is conservative for v2: per-chunk
        // references never pack wider than one reference, since each chunk's range lies within the
        // array's, so the ratio can understate v2. It also skips arrays where only per-chunk
        // references would help, such as those whose minimum is zero, or whose single-reference
        // width is no narrower than plain BitPacking.
        let stats = data.integer_stats(exec_ctx);

        // Only apply when the min is not already zero.
        if stats.erased().min_is_zero() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Difference between max and min.
        let for_bitwidth = match stats.erased().max_minus_min().checked_ilog2() {
            Some(l) => l + 1,
            // If max-min == 0, the we should be compressing this as a constant array.
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

        let full_width: u32 = data
            .array_as_primitive()
            .ptype()
            .bit_width()
            .try_into()
            .vortex_expect("bit width must fit in u32");

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            full_width as f64 / for_bitwidth as f64,
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
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
        let leaf_ctx = compress_ctx.clone().as_leaf();
        let biased_data =
            ArrayAndStats::new(biased.into_array(), compress_ctx.merged_stats_options());
        let compressed = BitPackingScheme.compress(compressor, &biased_data, leaf_ctx, exec_ctx)?;

        // TODO(connor): This should really be `new_unchecked`.
        let for_compressed = match for_array.constant_reference() {
            Some(reference) => FoR::try_new(compressed, reference)?,
            None => {
                let references = compressor.compress_child(
                    for_array.references(),
                    &compress_ctx,
                    self.id(),
                    0,
                    exec_ctx,
                )?;
                FoR::try_new_chunked(compressed, references, for_array.offset())?
            }
        };
        for_compressed
            .as_ref()
            .statistics()
            .inherit_from(for_array.as_ref().statistics());

        Ok(for_compressed.into_array())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_fastlanes::for_v1_id;
    use vortex_fastlanes::for_v2_id;

    use super::FOR_V1;
    use super::FOR_V2;
    use crate::Scheme;
    use crate::SchemeExt;

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
        let expected: &dyn Scheme = if expect_v2 { &FOR_V2 } else { &FOR_V1 };
        assert_eq!(refined.produced_encodings(), expected.produced_encodings());
    }
}
