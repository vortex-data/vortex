// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution for primitive [`Interleave`] values.

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::super::InterleaveSlots;
use super::fixed_width::Source;
use super::fixed_width::gather;
use crate::AnyColumnar;
use crate::array::Array;
use crate::arrays::Constant;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::ToBytes;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_ptype_width;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;

pub(super) fn execute(
    mut array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();
    array =
        require_child!(array, array.array_indices(), InterleaveSlots::ARRAY_INDICES => Primitive);
    array = require_child!(array, array.row_indices(), InterleaveSlots::ROW_INDICES => Primitive);
    for i in 0..num_values {
        array = require_child!(array, array.value(i), InterleaveSlots::VALUES_OFFSET + i => AnyColumnar);
    }

    let validity = array.as_ref().validity()?;
    let ptype = array.dtype().as_ptype();
    let values = match_each_ptype_width!(
        ptype,
        |W| {
            gather_values::<W>(&array, ctx.allocator())?
                .into_byte_buffer()
                .aligned(Alignment::new(W))
        },
        _ => { unreachable!("primitive widths are specialized") },
    );
    let output = PrimitiveArray::from_byte_buffer(values, ptype, validity);

    Ok(ExecutionResult::done(output))
}

fn gather_values<const W: usize>(
    array: &Array<Interleave>,
    allocator: &BufferAllocatorRef,
) -> VortexResult<Buffer<[u8; W]>> {
    let values = (0..array.num_values())
        .map(|i| {
            let value = array.value(i);
            let len = value.len();
            if let Some(constant) = value.as_opt::<Constant>() {
                // Validity carries nullness; a null constant's payload is never observed.
                let payload = constant
                    .scalar()
                    .as_primitive()
                    .pvalue()
                    .map(|value| {
                        let mut bytes = [0u8; W];
                        bytes.copy_from_slice(value.to_le_bytes());
                        bytes
                    })
                    .unwrap_or([0; W]);
                Source {
                    data: Buffer::full_in(payload, 1, allocator.clone()),
                    len,
                    row_mask: 0,
                }
            } else {
                Source {
                    data: Buffer::<[u8; W]>::from_byte_buffer(
                        value.as_::<Primitive>().buffer_handle().to_host_sync(),
                    ),
                    len,
                    row_mask: usize::MAX,
                }
            }
        })
        .collect::<Vec<_>>();
    let branches = array.array_indices().as_::<Primitive>();
    let rows = array.row_indices().as_::<Primitive>();

    match_each_unsigned_integer_ptype!(branches.ptype(), |A| {
        match_each_unsigned_integer_ptype!(rows.ptype(), |R| {
            gather(
                &values,
                branches.as_slice::<A>(),
                rows.as_slice::<R>(),
                allocator,
            )
        })
    })
}
