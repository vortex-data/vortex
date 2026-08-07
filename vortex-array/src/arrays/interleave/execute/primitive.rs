// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution for primitive [`Interleave`] values.

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::validate_selectors;
use crate::array::Array;
use crate::array::ArrayView;
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
        if array.value(i).as_opt::<Constant>().is_none() {
            array = require_child!(array, array.value(i), i + 2 => Primitive);
        }
    }

    let validity = array.as_ref().validity()?;
    let output = match_each_native_ptype!(array.value(0).dtype().as_ptype(), |T| {
        let values = gather_values::<T>(&array)?;
        VortexResult::Ok(PrimitiveArray::new(values, validity))
    })?;

    Ok(ExecutionResult::done(output))
}

/// Physical primitive values; nullness remains in the source array's validity.
enum PrimitiveValues<T> {
    Buffer(Buffer<T>),
    Constant { value: T, len: usize },
}

impl<T: Copy> PrimitiveValues<T> {
    fn len(&self) -> usize {
        match self {
            Self::Buffer(values) => values.len(),
            Self::Constant { len, .. } => *len,
        }
    }

    fn value(&self, index: usize) -> T {
        match self {
            Self::Buffer(values) => values[index],
            Self::Constant { value, .. } => *value,
        }
    }
}

fn gather_values<T: NativePType>(array: &Array<Interleave>) -> VortexResult<Buffer<T>> {
    let values = (0..array.num_values())
        .map(|i| {
            let value = array.value(i);
            if let Some(constant) = value.as_opt::<Constant>() {
                PrimitiveValues::Constant {
                    value: constant
                        .scalar()
                        .as_primitive()
                        .typed_value::<T>()
                        // Validity carries nullness; a null constant's payload is never observed.
                        .unwrap_or_default(),
                    len: value.len(),
                }
            } else {
                PrimitiveValues::Buffer(value.as_::<Primitive>().to_buffer::<T>())
            }
        })
        .collect::<Vec<_>>();
    let branches = array.array_indices().as_::<Primitive>();
    let rows = array.row_indices().as_::<Primitive>();

    match_each_unsigned_integer_ptype!(branches.ptype(), |A| {
        gather_rows::<T, A>(&values, branches.as_slice::<A>(), rows)
    })
}

fn gather_rows<T, A>(
    values: &[PrimitiveValues<T>],
    branches: &[A],
    rows: ArrayView<'_, Primitive>,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    A: AsPrimitive<usize>,
{
    match_each_unsigned_integer_ptype!(rows.ptype(), |R| {
        gather(values, branches, rows.as_slice::<R>())
    })
}

fn gather<T, A, R>(
    values: &[PrimitiveValues<T>],
    branches: &[A],
    rows: &[R],
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    let len = validate_selectors(values.len(), |branch| values[branch].len(), branches, rows)?;
    let mut output = BufferMut::with_capacity(len);
    for i in 0..len {
        output.push(values[branches[i].as_()].value(rows[i].as_()));
    }
    Ok(output.freeze())
}
