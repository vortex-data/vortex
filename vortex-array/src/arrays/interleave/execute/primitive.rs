// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution for primitive [`Interleave`] values.

use num_traits::AsPrimitive;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::super::InterleaveSlots;
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

/// Physical storage for one primitive source array.
struct PrimitiveSource<T> {
    data: Buffer<T>,
    len: usize,
    // Maps logical rows to `data`: zero for constants and identity for decoded buffers.
    row_mask: usize,
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
                PrimitiveSource {
                    data: Buffer::full_in(payload, 1, allocator.clone()),
                    len,
                    row_mask: 0,
                }
            } else {
                PrimitiveSource {
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

fn gather<T, A, R>(
    values: &[PrimitiveSource<T>],
    branches: &[A],
    rows: &[R],
    allocator: &BufferAllocatorRef,
) -> VortexResult<Buffer<T>>
where
    T: Copy,
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    // `zip` truncates to the shorter input.
    vortex_ensure_eq!(
        rows.len(),
        branches.len(),
        "interleave row_indices length does not match array_indices length",
    );

    // Write rows through a `&mut` slice. Yielding `Result<T, _>` per row splits each `[u8; W]` copy
    // into narrow moves (the `Ok` payload is misaligned), and pushing through `BufferMut` reloads
    // its length on every row.
    let len = branches.len();
    let mut output = BufferMut::with_capacity_in(len, allocator.clone());
    let slots = &mut output.spare_capacity_mut()[..len];
    for ((slot, branch), row) in slots.iter_mut().zip(branches).zip(rows) {
        let Some(source) = values.get((*branch).as_()) else {
            vortex_bail!("interleave array index out of bounds");
        };
        let row = (*row).as_();
        vortex_ensure!(row < source.len, "interleave row index out of bounds");
        slot.write(source.data[row & source.row_mask]);
    }
    // SAFETY: the loop returns on any error, so reaching here means it wrote all `len` slots:
    // `slots`, `branches`, and `rows` all have length `len`.
    unsafe { output.set_len(len) };
    Ok(output.freeze())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::BufferAllocatorRef;

    #[test]
    fn rejects_out_of_bounds_selectors() {
        let values = [PrimitiveSource {
            data: Buffer::full_in(1u32, 1, BufferAllocatorRef::statically_allocated()),
            len: 1,
            row_mask: 0,
        }];

        let allocator = BufferAllocatorRef::statically_allocated();
        let error = gather(&values, &[1u8], &[0u8], &allocator).unwrap_err();
        assert!(
            error.to_string().contains("array index out of bounds"),
            "{error}"
        );
        let error = gather(&values, &[0u8], &[1u8], &allocator).unwrap_err();
        assert!(
            error.to_string().contains("row index out of bounds"),
            "{error}"
        );
    }
}
