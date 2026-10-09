// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BitPacking integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitWidths;
use vortex_fastlanes::bitpack_compress::bit_width_histogram;
use vortex_fastlanes::bitpack_compress::bitpack_blocked_to_best_bit_widths;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::find_best_bit_width;
use vortex_fastlanes::bitpacked_v1_id;
use vortex_fastlanes::bitpacked_v2_id;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::compress_patches;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum BitPackingSchemeMode {
    V1,
    V2,
}

/// The BitPacking scheme that only produces arrays with one bit width.
pub(crate) static BITPACKING_V1: BitPackingScheme = BitPackingScheme::v1();

/// The BitPacking scheme that may give each 1024-element block its own bit width.
pub(crate) static BITPACKING_V2: BitPackingScheme = BitPackingScheme::v2();

/// BitPacking encoding for non-negative integers.
///
/// The v1 mode packs every value at one bit width, and serializes as `fastlanes.bitpacked`. The v2
/// mode always chooses a width for each 1024-element block, and serializes as
/// `fastlanes.bitpacked.v2`.
///
/// The default uses v1. [`refine`](Scheme::refine) picks v2 when the v2 ID is allowed and v1
/// otherwise.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitPackingScheme {
    mode: BitPackingSchemeMode,
}

impl BitPackingScheme {
    /// Creates a BitPacking scheme configured for v1, which uses one bit width.
    pub const fn v1() -> Self {
        Self {
            mode: BitPackingSchemeMode::V1,
        }
    }

    /// Creates a BitPacking scheme configured for v2, which may use one bit width per block.
    pub const fn v2() -> Self {
        Self {
            mode: BitPackingSchemeMode::V2,
        }
    }
}

impl Default for BitPackingScheme {
    fn default() -> Self {
        Self::v1()
    }
}

impl Scheme for BitPackingScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.bitpacking"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        let mut encodings = match self.mode {
            // Global-width arrays serialize under the frozen v1 ID.
            BitPackingSchemeMode::V1 => vec![bitpacked_v1_id()],
            BitPackingSchemeMode::V2 => vec![bitpacked_v2_id()],
        };
        if use_experimental_patches() {
            encodings.push(Patched.id());
        }
        encodings
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        if allowed(&bitpacked_v2_id()) {
            &BITPACKING_V2
        } else {
            &BITPACKING_V1
        }
    }

    /// Children: block offsets=0 in v2 mode.
    fn num_children(&self) -> usize {
        match self.mode {
            BitPackingSchemeMode::V1 => 0,
            BitPackingSchemeMode::V2 => 1,
        }
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.integer_stats(exec_ctx);

        // BitPacking only works for non-negative values.
        if stats.erased().min_is_negative() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if self.mode == BitPackingSchemeMode::V2 {
            return self.compress_blocked(compressor, data, compress_ctx, exec_ctx);
        }

        let primitive_array = data.array_as_primitive();

        let histogram = bit_width_histogram(primitive_array, exec_ctx)?;
        let bw = find_best_bit_width(primitive_array.ptype(), &histogram)?;

        // If best bw is determined to be the current bit-width, return the original array.
        if bw as usize == primitive_array.ptype().bit_width() {
            return Ok(primitive_array.array().clone());
        }

        // Otherwise we can bitpack the array.
        let primitive_array = primitive_array.into_owned();
        let packed = bitpack_encode(&primitive_array, bw, Some(&histogram), exec_ctx)?;

        let packed_stats = packed.statistics().to_owned();
        let ptype = packed.dtype().as_ptype();
        let mut parts = BitPacked::into_parts(packed);

        let array = if use_experimental_patches() {
            let patches = parts.patches.take();
            // Transpose patches into G-ALP style PatchedArray, wrapping an inner BitPackedArray.
            let array = BitPacked::try_new(
                parts.packed,
                ptype,
                parts.validity,
                None,
                bw,
                parts.len,
                parts.offset,
            )?
            .into_array();

            match patches {
                None => array,
                Some(p) => Patched::from_array_and_patches(array, &p, exec_ctx)?
                    .with_stats_set(packed_stats)
                    .into_array(),
            }
        } else {
            // Compress patches and place back into BitPackedArray.
            let patches = parts
                .patches
                .take()
                .map(|p| compress_patches(p, exec_ctx))
                .transpose()?;
            parts.patches = patches;
            BitPacked::try_new(
                parts.packed,
                ptype,
                parts.validity,
                parts.patches,
                bw,
                parts.len,
                parts.offset,
            )?
            .with_stats_set(packed_stats)
            .into_array()
        };

        Ok(array)
    }
}

impl BitPackingScheme {
    /// Bit-pack each 1024-element block at its own best width.
    fn compress_blocked(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive_array = data.array_as_primitive().into_owned();
        let packed = bitpack_blocked_to_best_bit_widths(&primitive_array, exec_ctx)?;

        let packed_stats = packed.statistics().to_owned();
        let ptype = packed.dtype().as_ptype();
        let mut parts = BitPacked::into_parts(packed);
        let BitWidths::Blocked(block_offsets) = parts.bit_widths else {
            vortex_bail!("Blocked bit-packing must produce block offsets");
        };
        let block_offsets =
            compressor.compress_child(&block_offsets, &compress_ctx, self.id(), 0, exec_ctx)?;

        let array = if use_experimental_patches() {
            let patches = parts.patches.take();
            // Transpose patches into G-ALP style PatchedArray, wrapping an inner BitPackedArray.
            let array = BitPacked::try_new_with_block_offsets(
                parts.packed,
                ptype,
                parts.validity,
                None,
                block_offsets,
                parts.len,
                parts.offset,
            )?
            .into_array();

            match patches {
                None => array,
                Some(p) => Patched::from_array_and_patches(array, &p, exec_ctx)?
                    .with_stats_set(packed_stats)
                    .into_array(),
            }
        } else {
            // Compress patches and place back into BitPackedArray.
            let patches = parts
                .patches
                .take()
                .map(|p| compress_patches(p, exec_ctx))
                .transpose()?;
            BitPacked::try_new_with_block_offsets(
                parts.packed,
                ptype,
                parts.validity,
                patches,
                block_offsets,
                parts.len,
                parts.offset,
            )?
            .with_stats_set(packed_stats)
            .into_array()
        };

        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_fastlanes::bitpacked_v1_id;
    use vortex_fastlanes::bitpacked_v2_id;

    use super::BITPACKING_V1;
    use super::BITPACKING_V2;
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
        #[values(&BITPACKING_V1, &BITPACKING_V2)] scheme: &'static dyn Scheme,
    ) {
        let allowed = |id: &ArrayId| {
            (allow_v1 && *id == bitpacked_v1_id()) || (allow_v2 && *id == bitpacked_v2_id())
        };
        let refined = scheme.refine(&allowed);
        assert_eq!(refined.id(), scheme.id());
        let expected: &dyn Scheme = if expect_v2 {
            &BITPACKING_V2
        } else {
            &BITPACKING_V1
        };
        assert_eq!(refined.produced_encodings(), expected.produced_encodings());
    }
}
