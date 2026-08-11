// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution for primitive [`Interleave`] values.

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use crate::AnyColumnar;
use crate::array::Array;
use crate::arrays::Constant;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;

pub(super) fn execute(
    mut array: Array<Interleave>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();
    array = require_child!(array, array.array_indices(), 0 => Primitive);
    array = require_child!(array, array.row_indices(), 1 => Primitive);
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i + 2 => AnyColumnar);
    }

    let validity = array.as_ref().validity()?;
    let output = match_each_native_ptype!(array.dtype().as_ptype(), |T| {
        let values = gather_values::<T>(&array)?;
        VortexResult::Ok(PrimitiveArray::new(values, validity))
    })?;

    Ok(ExecutionResult::done(output))
}

/// Physical storage for one primitive source array.
struct PrimitiveSource<T> {
    data: Buffer<T>,
    len: usize,
    // Maps logical rows to `data`: zero for constants and identity for decoded buffers.
    row_mask: usize,
}

fn gather_values<T: NativePType>(array: &Array<Interleave>) -> VortexResult<Buffer<T>> {
    let values = (0..array.num_values())
        .map(|i| {
            let value = array.value(i);
            let len = value.len();
            if let Some(constant) = value.as_opt::<Constant>() {
                // Validity carries nullness; a null constant's payload is never observed.
                let payload = constant
                    .scalar()
                    .as_primitive()
                    .typed_value::<T>()
                    .unwrap_or_default();
                PrimitiveSource {
                    data: buffer![payload],
                    len,
                    row_mask: 0,
                }
            } else {
                PrimitiveSource {
                    data: value.as_::<Primitive>().to_buffer::<T>(),
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
            gather(&values, branches.as_slice::<A>(), rows.as_slice::<R>())
        })
    })
}

fn gather<T, A, R>(
    values: &[PrimitiveSource<T>],
    branches: &[A],
    rows: &[R],
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    // `zip` truncates to the shorter input.
    vortex_ensure!(
        rows.len() == branches.len(),
        "interleave selectors differ in length: array_indices {}, row_indices {}",
        branches.len(),
        rows.len()
    );

    let output =
        BufferMut::try_from_trusted_len_iter(branches.iter().zip(rows).map(|(branch, row)| {
            let Some(source) = values.get((*branch).as_()) else {
                vortex_bail!("interleave array index out of bounds");
            };
            let row = (*row).as_();
            vortex_ensure!(row < source.len, "interleave row index out of bounds");
            Ok(source.data[row & source.row_mask])
        }))?;
    Ok(output.freeze())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_out_of_bounds_selectors() {
        let values = [PrimitiveSource {
            data: buffer![1u32],
            len: 1,
            row_mask: 0,
        }];

        assert!(gather(&values, &[1u8], &[0u8]).is_err());
        assert!(gather(&values, &[0u8], &[1u8]).is_err());
    }
}
