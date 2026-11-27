// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::block::Block;
use crate::block::BlockOp;
use crate::block::BlockProgram;
use crate::block::Values;
use crate::block::Written;
use crate::dtype::NativePType;
use crate::dtype::UnsignedPType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;

/// Decode compressed dictionary codes directly into a primitive lookup. Null indices may contain
/// arbitrary payloads, so leave null handling and signed index casts to the ordinary take kernel.
pub(super) fn take_blocks(
    values: ArrayView<'_, Primitive>,
    codes: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<PrimitiveArray>> {
    let code_type = codes.dtype().as_ptype();
    if !code_type.is_unsigned_int()
        || !values.validity()?.definitely_no_nulls()
        || codes.dtype().is_nullable()
    {
        return Ok(None);
    }
    let Some(source) = codes.block_source(ctx)? else {
        return Ok(None);
    };
    let mut program = BlockProgram::new(source, code_type, codes.len());
    let op = match_each_unsigned_integer_ptype!(code_type, |I| {
        match_each_native_ptype!(values.ptype(), |T| {
            Box::new(Lookup::<I, T> {
                table: values.to_buffer::<T>(),
                index: PhantomData,
            }) as Box<dyn BlockOp>
        })
    });
    program.push(op, values.ptype());
    let nullability = values.dtype().nullability() | codes.dtype().nullability();
    Ok(Some(program.run(nullability, ctx)?))
}

struct Lookup<I, T> {
    table: Buffer<T>,
    index: PhantomData<I>,
}

impl<I: UnsignedPType, T: NativePType> BlockOp for Lookup<I, T> {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        let Values::Copy { input, mut out } = block.values else {
            vortex_panic!("dictionary lookup writes a separate output block")
        };
        let codes = input.as_slice::<I>();
        let table = self.table.as_slice();
        vortex_ensure!(
            codes
                .iter()
                .copied()
                .max()
                .and_then(|code| code.to_usize())
                .is_some_and(|max| max < table.len()),
            "dictionary code out of bounds for {} values",
            table.len()
        );
        // SAFETY: The preceding pass bounds-checked every code against this table. Keeping the
        // check outside the gather loop allows independent table loads to execute together.
        out.push_map(codes, |code| unsafe { *table.get_unchecked(code.as_()) });
        out.finish()
    }

    fn supports_in_place(&self) -> bool {
        false
    }
}
