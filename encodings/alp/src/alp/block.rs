// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-fused execution for ALP: each block of encoded integers decodes into floats and is
//! patched while L1-resident. See [`vortex_array::block`].

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::block::BlockIn;
use vortex_array::block::BlockOp;
use vortex_array::block::BlockOut;
use vortex_array::block::BlockPatches;
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
    fn process(
        &mut self,
        start: usize,
        input: BlockIn<'_>,
        mut out: BlockOut<'_>,
    ) -> VortexResult<()> {
        let exponents = self.exponents;
        let values = out.map_from(input, |encoded: F::ALPInt| {
            F::decode_single(encoded, exponents)
        });
        if let Some(patches) = &mut self.patches {
            patches.apply(start, values);
        }
        Ok(())
    }
}
