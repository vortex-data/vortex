// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALP's op for a [`BlockProgram`](vortex_array::block::BlockProgram): each block of encoded
//! integers decodes into floats and is patched while it is in L1.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::block::Block;
use vortex_array::block::BlockOp;
use vortex_array::block::BlockPatches;
use vortex_array::block::Written;
use vortex_array::dtype::NativePType;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPFloat;
use crate::Exponents;
use crate::match_each_alp_float_ptype;

/// ALP's op over its encoded child: decode each block, then apply the patches that fall in it.
pub fn alp_block_op(
    array: ArrayView<'_, ALP>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn BlockOp>> {
    let (exponents, patches) = (array.exponents(), array.patches());
    match_each_alp_float_ptype!(array.dtype().as_ptype(), |F| {
        decode_op::<F>(exponents, patches.as_ref(), ctx)
    })
}

fn decode_op<F>(
    exponents: Exponents,
    patches: Option<&Patches>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn BlockOp>>
where
    F: ALPFloat + NativePType,
    F::ALPInt: NativePType,
{
    let patches = patches
        .map(|patches| BlockPatches::try_new(patches, ctx, |_, value: F| value))
        .transpose()?;
    Ok(Box::new(Decode::<F> { exponents, patches }))
}

struct Decode<F> {
    exponents: Exponents,
    patches: Option<BlockPatches<F>>,
}

impl<F> BlockOp for Decode<F>
where
    F: ALPFloat + NativePType,
    F::ALPInt: NativePType,
{
    fn process<'a>(&mut self, start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        let Block {
            values,
            mut validity,
        } = block;
        let exponents = self.exponents;
        let mut written = values.map(|encoded: F::ALPInt| F::decode_single(encoded, exponents));
        if let Some(patches) = &mut self.patches {
            patches.apply(start, written.as_mut_slice::<F>(), &mut validity);
        }
        Ok(written)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::block::BlockProgram;
    use vortex_array::block::BlockSource;
    use vortex_array::block::PrimitiveSource;
    use vortex_array::dtype::NativePType;
    use vortex_array::match_each_integer_ptype;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_fastlanes::FoR;
    use vortex_fastlanes::bitpack_compress::bitpack_encode;
    use vortex_fastlanes::block::for_block_source;
    use vortex_session::VortexSession;

    use super::alp_block_op;
    use crate::ALP;
    use crate::ALPArrayExt;
    use crate::ALPArraySlotsExt;
    use crate::alp_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_fastlanes::initialize(&session);
        crate::initialize(&session);
        session
    });

    const LEN: usize = 5000;

    /// ALP over FoR over BitPacked, the layout the compressor writes for decimal floats, of `f64`,
    /// or of `f32` when `single`. Every 97th value, when `patched`, is not representable, so ALP
    /// patches it.
    fn alp_for_bp(patched: bool, nullable: bool, single: bool) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..LEN).map(|i| {
            if patched && i.is_multiple_of(97) {
                PI * i as f64
            } else {
                100.0 + (i % 1000) as f64 * 0.25
            }
        });
        let valid = |i: usize| !nullable || !i.is_multiple_of(13);
        let values = if single {
            PrimitiveArray::from_option_iter(
                values
                    .enumerate()
                    .map(|(i, v)| valid(i).then_some(v as f32)),
            )
        } else {
            PrimitiveArray::from_option_iter(values.enumerate().map(|(i, v)| valid(i).then_some(v)))
        };
        let alp = alp_encode(values.as_view(), None, &mut ctx)?;
        assert_eq!(alp.patches().is_some(), patched);
        let encoded = alp.encoded().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let for_bp =
            match_each_integer_ptype!(encoded.ptype(), |T| { for_over_bitpacked::<T>(&encoded)? });
        Ok(ALP::try_new(for_bp, alp.exponents(), alp.patches())?.into_array())
    }

    /// `encoded` shifted by its minimum and bit-packed, under FoR with that minimum.
    fn for_over_bitpacked<T>(encoded: &PrimitiveArray) -> VortexResult<ArrayRef>
    where
        T: NativePType + num_traits::PrimInt,
        Scalar: From<T>,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let values = encoded.as_slice::<T>();
        let min = values.iter().copied().min().unwrap_or_else(T::zero);
        let max = values.iter().copied().max().unwrap_or_else(T::zero);
        let shifted = values.iter().map(|&e| e - min).collect::<Buffer<T>>();
        let bit_width = T::PTYPE.bit_width() as u32 - (max - min).leading_zeros();
        let shifted = PrimitiveArray::new(shifted, encoded.validity()?);
        let bp = bitpack_encode(&shifted, u8::try_from(bit_width)?, None, &mut ctx)?;
        Ok(FoR::try_new(bp.into_array(), Scalar::from(min))?.into_array())
    }

    /// ALP's encoded child as a source: FoR's fused unpack over BitPacked, or its canonical
    /// values otherwise.
    fn leaf_source(encoded: &ArrayRef) -> VortexResult<Box<dyn BlockSource>> {
        let mut ctx = SESSION.create_execution_ctx();
        if let Some(for_) = encoded.as_opt::<FoR>()
            && let Some(source) = for_block_source(for_, &mut ctx)?
        {
            return Ok(source);
        }
        Ok(PrimitiveSource::boxed(
            &encoded.clone().execute::<PrimitiveArray>(&mut ctx)?,
        ))
    }

    /// ALP's decode over FoR over BitPacked, block by block, matches decoding ALP level-wise:
    /// nulls, patches, `f32`, and slices that start part way into a chunk or end past a patch.
    #[rstest]
    #[case::plain(false, false, false, None)]
    #[case::patched(true, false, false, None)]
    #[case::nullable_patched(true, true, false, None)]
    #[case::single_patched(true, true, true, None)]
    #[case::sliced_mid_chunk(true, true, false, Some((517, 4013)))]
    #[case::sliced_to_a_patch(true, false, false, Some((1023, 11 * 97 + 1)))]
    #[case::sliced_single(true, false, true, Some((1000, 1100)))]
    fn decode_matches_level_wise(
        #[case] patched: bool,
        #[case] nullable: bool,
        #[case] single: bool,
        #[case] rows: Option<(usize, usize)>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut array = alp_for_bp(patched, nullable, single)?;
        if let Some((start, end)) = rows {
            array = array.slice(start..end)?.execute::<ArrayRef>(&mut ctx)?;
        }
        let alp = array.as_::<ALP>();
        let encoded = alp.encoded().clone();
        let validity = encoded.validity()?.execute_mask(encoded.len(), &mut ctx)?;
        let mut program = BlockProgram::new(
            leaf_source(&encoded)?,
            encoded.dtype().as_ptype(),
            encoded.len(),
        );
        program.push(alp_block_op(alp, &mut ctx)?, array.dtype().as_ptype());
        let output = program.run(&validity, array.dtype().nullability(), &mut ctx)?;
        let expected = array.clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(output, expected, &mut ctx);
        Ok(())
    }
}
