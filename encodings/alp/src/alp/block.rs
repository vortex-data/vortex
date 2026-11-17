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
        // ALP's execute asks for its patches before its encoded spine, so a fused chain only
        // reaches ALP with them executed; a leaf ALP over a canonical child may still hold
        // encoded patches, which resolving executes.
        let patches = patches
            .map(|patches| BlockPatches::try_new(&patches, ctx, |_, value: F| value))
            .transpose()?;
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
mod tests {
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::block::fused_chain_count;
    use vortex_array::scalar::Scalar;
    use vortex_array::test_harness::trace::trace_op;
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

    /// ALP over FoR over BitPacked, the layout the compressor writes for decimal floats. Every
    /// 97th value, when `patched`, is not representable, so ALP patches it.
    fn alp_for_bp(patched: bool, nullable: bool) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..LEN).map(|i| {
            if patched && i % 97 == 0 {
                PI * i as f64
            } else {
                100.0 + (i % 1000) as f64 * 0.25
            }
        });
        let values = if nullable {
            PrimitiveArray::from_option_iter(
                values.enumerate().map(|(i, v)| (i % 13 != 0).then_some(v)),
            )
        } else {
            PrimitiveArray::from_iter(values)
        };
        let alp = alp_encode(values.as_view(), None, &mut ctx)?;
        assert_eq!(alp.patches().is_some(), patched);

        let encoded = alp.encoded().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let min = encoded
            .as_slice::<i64>()
            .iter()
            .copied()
            .min()
            .unwrap_or_default();
        let shifted = PrimitiveArray::new(
            encoded
                .as_slice::<i64>()
                .iter()
                .map(|&e| e - min)
                .collect::<Buffer<i64>>(),
            encoded.validity()?,
        );
        let bit_width = 64
            - shifted
                .as_slice::<i64>()
                .iter()
                .max()
                .copied()
                .unwrap_or(0)
                .leading_zeros();
        let bp = bitpack_encode(&shifted, u8::try_from(bit_width)?, None, &mut ctx)?;
        let for_bp = FoR::try_new(bp.into_array(), Scalar::from(min))?.into_array();
        Ok(ALP::try_new(for_bp, alp.exponents(), alp.patches())?.into_array())
    }

    /// Execute `array` with fusion on and off, check they agree, and return the fusion steps of
    /// the fused run's trace.
    fn fused(array: ArrayRef) -> VortexResult<String> {
        let expected = array.clone().execute::<PrimitiveArray>(
            &mut SESSION.create_execution_ctx().with_block_fusion(false),
        )?;
        let mut ctx = SESSION.create_execution_ctx();
        let traced = trace_op(|| array.execute::<PrimitiveArray>(&mut ctx))?;
        assert_arrays_eq!(traced.output, expected, &mut ctx);
        Ok(traced
            .trace
            .to_string()
            .lines()
            .map(str::trim)
            .filter(|line| {
                ["ExecuteSpine", "fuse ", "drive ", "block_op"]
                    .iter()
                    .any(|prefix| line.starts_with(prefix))
            })
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// ALP asks for FoR over BitPacked as its spine. On the way back up FoR's fused unpack starts
    /// a block program, ALP extends it with its decode, and the root drives it: each block is
    /// unpacked with its reference added, then decoded and patched into the output.
    #[rstest]
    #[case::plain(false, false)]
    #[case::patched(true, false)]
    #[case::nullable_patched(true, true)]
    fn alp_over_for_bp_decodes_per_block(
        #[case] patched: bool,
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let array = alp_for_bp(patched, nullable)?;
        let alp = array.dtype().to_string();
        let for_bp = array.as_::<ALP>().encoded().dtype().to_string();
        let before = fused_chain_count();
        let trace = fused(array)?;
        assert_eq!(fused_chain_count() - before, 1);
        assert_eq!(
            trace,
            format!(
                "ExecuteSpine slot=0 parent=vortex.alp({alp}, len={LEN}) \
                 child=fastlanes.for.v2({for_bp}, len={LEN})
fuse start input=source array=fastlanes.for.v2({for_bp}, len={LEN})
fuse extend parent=vortex.alp({alp}, len={LEN}) ops=1
drive input=source ops=1 len={LEN}
block_op 0 copy i64->f64 scratch0 -> output"
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
        let before = fused_chain_count();
        assert_eq!(fused(alp)?, "");
        assert_eq!(fused_chain_count(), before);
        Ok(())
    }
}
