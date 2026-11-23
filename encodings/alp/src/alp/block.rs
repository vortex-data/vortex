// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-fused execution for ALP: each block of encoded integers decodes into floats and is
//! patched while L1-resident. See [`vortex_array::block`].

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::block::Block;
use vortex_array::block::BlockOp;
use vortex_array::block::BlockPatches;
use vortex_array::block::Written;
use vortex_array::dtype::NativePType;
use vortex_error::VortexResult;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPFloat;
use crate::ALPSlots;
use crate::Exponents;
use crate::match_each_alp_float_ptype;

pub(crate) fn alp_block_op(
    array: ArrayView<'_, ALP>,
    slot_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockOp>>> {
    if slot_idx != ALPSlots::ENCODED {
        return Ok(None);
    }
    let exponents = array.exponents();
    let patches = array.patches();
    match_each_alp_float_ptype!(array.dtype().as_ptype(), |F| {
        // ALP's execute executes its patches before it asks for its encoded spine, so they are
        // canonical here and resolving them only collects them.
        let patches = match patches {
            Some(patches) => match BlockPatches::try_new(&patches, ctx, |_, value: F| value)? {
                Some(patches) => Some(patches),
                None => return Ok(None),
            },
            None => None,
        };
        Ok(Some(
            Box::new(Decode::<F> { exponents, patches }) as Box<dyn BlockOp>
        ))
    })
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
        let exponents = self.exponents;
        let mut written = block.map(|encoded: F::ALPInt| F::decode_single(encoded, exponents));
        if let Some(patches) = &mut self.patches {
            patches.apply(start, written.as_mut_slice::<F>());
        }
        Ok(written)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::f64::consts::PI;
    use std::ops::Range;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::NativePType;
    use vortex_array::match_each_integer_ptype;
    use vortex_array::scalar::Scalar;
    use vortex_array::test_harness::fusion::fusion_trace;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_fastlanes::FoR;
    use vortex_fastlanes::bitpack_compress::bitpack_encode;
    use vortex_session::VortexSession;

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

    /// ALP over FoR over BitPacked, the layout the compressor writes for decimal floats, of
    /// `f64` or of `f32` when `single`. Every 97th value, when `patched`, is not representable,
    /// so ALP patches it.
    fn alp_for_bp(patched: bool, nullable: bool, single: bool) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..LEN).map(|i| {
            if patched && i % 97 == 0 {
                PI * i as f64
            } else {
                100.0 + (i % 1000) as f64 * 0.25
            }
        });
        let values = match (single, nullable) {
            (true, true) => PrimitiveArray::from_option_iter(
                values
                    .enumerate()
                    .map(|(i, v)| (i % 13 != 0).then_some(v as f32)),
            ),
            (true, false) => PrimitiveArray::from_iter(values.map(|v| v as f32)),
            (false, true) => PrimitiveArray::from_option_iter(
                values.enumerate().map(|(i, v)| (i % 13 != 0).then_some(v)),
            ),
            (false, false) => PrimitiveArray::from_iter(values),
        };
        let alp = alp_encode(values.as_view(), None, &mut ctx)?;
        assert_eq!(alp.patches().is_some(), patched);

        let encoded = alp.encoded().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let (shifted, bit_width, min) = match_each_integer_ptype!(encoded.ptype(), |T| {
            let values = encoded.as_slice::<T>();
            let min = values.iter().copied().min().unwrap_or_default();
            let max = values.iter().copied().max().unwrap_or_default();
            let shifted = values.iter().map(|&e| e - min).collect::<Buffer<T>>();
            let bit_width = T::PTYPE.bit_width() as u32 - (max - min).leading_zeros();
            (
                PrimitiveArray::new(shifted, encoded.validity()?),
                bit_width,
                Scalar::from(min),
            )
        });
        let bp = bitpack_encode(&shifted, u8::try_from(bit_width)?, None, &mut ctx)?;
        let for_bp = FoR::try_new(bp.into_array(), min)?.into_array();
        Ok(ALP::try_new(for_bp, alp.exponents(), alp.patches())?.into_array())
    }

    /// Execute `array` with fusion off and on, check they agree, and return the fused run's
    /// fusion steps.
    fn fused(array: ArrayRef) -> VortexResult<String> {
        fusion_trace(array, &SESSION)
    }

    /// ALP asks for FoR over BitPacked as its spine. On the way back up FoR's fused unpack starts
    /// a block program, ALP extends it with its decode, and the root drives it: each block is
    /// unpacked with its reference added, then decoded and patched into the output.
    #[rstest]
    #[case::plain(false, false, false, None)]
    #[case::patched(true, false, false, None)]
    #[case::nullable_patched(true, true, false, None)]
    #[case::single_patched(true, true, true, None)]
    #[case::sliced_mid_chunk(true, true, false, Some(517..4013))]
    #[case::sliced_to_a_patch(true, false, false, Some(1023..11 * 97 + 1))]
    #[case::sliced_single(true, false, true, Some(1000..1100))]
    fn alp_over_for_bp_decodes_per_block(
        #[case] patched: bool,
        #[case] nullable: bool,
        #[case] single: bool,
        #[case] rows: Option<Range<usize>>,
    ) -> VortexResult<()> {
        let mut array = alp_for_bp(patched, nullable, single)?;
        if let Some(rows) = rows {
            array = array.slice(rows)?;
        }
        let len = array.len();
        let alp = array.dtype().to_string();
        let (int, float) = if single {
            ("i32", "f32")
        } else {
            ("i64", "f64")
        };
        let for_bp = format!("{int}{}", if nullable { "?" } else { "" });
        let trace = fused(array)?;
        assert_eq!(
            trace,
            format!(
                "ExecuteSpine slot=0 parent=vortex.alp({alp}, len={len}) \
                 child=fastlanes.for.v2({for_bp}, len={len})
fuse start array=fastlanes.for.v2({for_bp}, len={len})
fuse extend parent=vortex.alp({alp}, len={len}) ops=1
drive ops=1 len={len}
block_op 0 copy {int}->{float} scratch0 -> output"
            )
        );
        Ok(())
    }

    /// ALP over its canonical encoded child runs its own decode: there is no chain to fuse.
    #[test]
    fn alp_over_primitive_runs_its_kernel() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..LEN).map(|i| 1.5 + i as f64 * 0.125));
        let alp = alp_encode(values.as_view(), None, &mut ctx)?.into_array();
        assert_eq!(fused(alp)?, "");
        Ok(())
    }
}
