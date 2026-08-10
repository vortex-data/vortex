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
use crate::AnyColumnar;
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
        array = require_child!(array, array.value(i), i + 2 => AnyColumnar);
    }

    let validity = array.as_ref().validity()?;
    let output = match_each_native_ptype!(array.dtype().as_ptype(), |T| {
        let values = gather_values::<T>(&array)?;
        VortexResult::Ok(PrimitiveArray::new(values, validity))
    })?;

    Ok(ExecutionResult::done(output))
}

/// Physical primitive values; nullness remains in the source array's validity.
enum PrimitiveValues<T> {
    Buffer(Buffer<T>),
    Constant(T),
}

impl<T: Copy> PrimitiveValues<T> {
    /// Returns the physical value at `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// For [`Self::Buffer`], `index` must be less than the buffer length. [`Self::Constant`]
    /// accepts any index because it has a single physical value.
    unsafe fn value_unchecked(&self, index: usize) -> T {
        match self {
            // SAFETY: the caller guarantees that `index` is in bounds.
            Self::Buffer(values) => *unsafe { values.get_unchecked(index) },
            Self::Constant(value) => *value,
        }
    }
}

fn gather_values<T: NativePType>(array: &Array<Interleave>) -> VortexResult<Buffer<T>> {
    let values = (0..array.num_values())
        .map(|i| {
            let value = array.value(i);
            if let Some(constant) = value.as_opt::<Constant>() {
                PrimitiveValues::Constant(
                    constant
                        .scalar()
                        .as_primitive()
                        .typed_value::<T>()
                        // Validity carries nullness; a null constant's payload is never observed.
                        .unwrap_or_default(),
                )
            } else {
                PrimitiveValues::Buffer(value.as_::<Primitive>().to_buffer::<T>())
            }
        })
        .collect::<Vec<_>>();
    let branches = array.array_indices().as_::<Primitive>();
    let rows = array.row_indices().as_::<Primitive>();

    match_each_unsigned_integer_ptype!(branches.ptype(), |A| {
        gather_rows::<T, A>(array, &values, branches.as_slice::<A>(), rows)
    })
}

fn gather_rows<T, A>(
    array: &Array<Interleave>,
    values: &[PrimitiveValues<T>],
    branches: &[A],
    rows: ArrayView<'_, Primitive>,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    A: AsPrimitive<usize>,
{
    match_each_unsigned_integer_ptype!(rows.ptype(), |R| {
        gather(array, values, branches, rows.as_slice::<R>())
    })
}

fn gather<T, A, R>(
    array: &Array<Interleave>,
    values: &[PrimitiveValues<T>],
    branches: &[A],
    rows: &[R],
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    let len = validate_selectors(
        values.len(),
        |branch| array.value(branch).len(),
        branches,
        rows,
    )?;

    // SAFETY: `validate_selectors` proved both selector lengths and every logical source bound.
    // Each `Buffer` has the same length as its source array, while `Constant` ignores the row.
    Ok(unsafe { gather_unchecked(len, values, branches, rows) })
}

/// Gathers one primitive value per output from `values[branches[i]]` at position `rows[i]`.
///
/// # Safety
///
/// `branches` and `rows` must both contain at least `len` elements. For every `i < len`,
/// `branches[i] < values.len()` and, when the selected value is a [`PrimitiveValues::Buffer`],
/// `rows[i]` must be less than that buffer's length.
unsafe fn gather_unchecked<T, A, R>(
    len: usize,
    values: &[PrimitiveValues<T>],
    branches: &[A],
    rows: &[R],
) -> Buffer<T>
where
    T: NativePType,
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    let mut output = BufferMut::with_capacity(len);
    for ((branch, row), slot) in branches.iter().zip(rows).zip(output.spare_capacity_mut()) {
        let branch = (*branch).as_();
        let row = (*row).as_();
        // SAFETY: the caller guarantees that the selected branch and row are in bounds for
        // `values` and the selected physical value buffer.
        slot.write(unsafe { values.get_unchecked(branch).value_unchecked(row) });
    }

    // SAFETY: the caller guarantees both selector slices have at least `len` elements, so the loop
    // initialized exactly `len` output slots.
    unsafe { output.set_len(len) };
    output.freeze()
}
