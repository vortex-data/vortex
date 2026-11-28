// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BitPacking integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::PType;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedDataParts;
use vortex_fastlanes::bitpack_compress::bit_width_histogram;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::find_best_bit_width;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;
use crate::schemes::patches::KeptPatches;

/// BitPacking encoding for non-negative integers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct BitPackingScheme;

impl BitPackingScheme {
    /// Bit-packs `primitive_array`, or returns `None` when it is empty or its best bit width does
    /// not save at least [`MIN_BITS_SAVED`] bits.
    pub(super) fn compress_array(
        &self,
        primitive_array: &PrimitiveArray,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<BitPackedArray>> {
        if primitive_array.is_empty() {
            return Ok(None);
        }
        let histogram = bit_width_histogram(primitive_array.as_view(), exec_ctx)?;
        let bw = find_best_bit_width(primitive_array.ptype(), &histogram)?;
        if !should_pack(primitive_array.ptype(), bw) {
            return Ok(None);
        }
        Ok(Some(bitpack_encode(
            primitive_array,
            bw,
            Some(&histogram),
            exec_ctx,
        )?))
    }

    /// Bit-packs `data`, or returns `None` when compressing is not beneficial.
    pub(super) fn try_encode(
        &self,
        data: &ArrayAndStats,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Encoded>> {
        let primitive_array = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let Some(packed) = self.compress_array(&primitive_array, exec_ctx)? else {
            return Ok(None);
        };
        let packed_stats = packed.statistics().to_owned();
        let ptype = packed.dtype().as_ptype();
        let BitPackedDataParts {
            offset,
            bit_width,
            len,
            packed,
            patches,
            validity,
        } = BitPacked::into_parts(packed);

        let (patches, keep) = KeptPatches::new(self, 0, data.array(), patches, exec_ctx)?;

        Ok(Some(
            Encoded::new(Vec::new(), keep, move |children, _| {
                let validity = children.validity()?;
                let array = patches.build(children, |patches| {
                    let array = BitPacked::try_new(
                        packed, ptype, validity, patches, bit_width, len, offset,
                    )?;
                    Ok(array.into_array())
                })?;
                array.statistics().set_all(packed_stats);
                Ok(array)
            })
            .with_validity(validity),
        ))
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
        // `compress_patches` stores constant patch values as a Constant array.
        let mut encodings = vec![BitPacked.id(), Constant.id()];
        if use_experimental_patches() {
            encodings.push(Patched.id());
        }
        encodings
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

    /// Children: patches, kept, from 0 on.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        Ok(self
            .try_encode(data, exec_ctx)?
            .unwrap_or_else(|| Encoded::leaf(data.array().clone())))
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

/// The fewest bits per value that packing must save.
const MIN_BITS_SAVED: usize = 2;

fn should_pack(ptype: PType, bit_width: u8) -> bool {
    ptype.bit_width() - usize::from(bit_width) >= MIN_BITS_SAVED
}
