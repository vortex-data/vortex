// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helpers shared by the ALP comparison and between kernels.
//!
//! Both kernels answer a predicate against the encoded integers and then correct the rows that
//! ALP could not encode, whose exact float values live in the patches.

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use crate::ALP;

/// A predicate that has the same answer for every row while keeping the array's validity.
pub(super) fn constant_predicate(
    array: ArrayView<'_, ALP>,
    value: bool,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    Ok(match array.validity()?.union_nullability(nullability) {
        Validity::NonNullable | Validity::AllValid => {
            ConstantArray::new(Scalar::bool(value, nullability), len).into_array()
        }
        Validity::AllInvalid => {
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), len).into_array()
        }
        validity @ Validity::Array(_) => {
            BoolArray::new(BitBuffer::full(value, len), validity).into_array()
        }
    })
}

/// Re-evaluates `predicate` on each patched row of `result`, a boolean answer computed from the
/// encoded values alone.
pub(super) fn apply_patch_predicate<F: NativePType>(
    result: ArrayRef,
    patches: &Patches,
    predicate: impl Fn(F) -> bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let result = result.execute::<BoolArray>(ctx)?;
    let validity = result.validity()?;
    let mut bits = result
        .into_bit_buffer()
        .try_into_mut()
        .unwrap_or_else(|bits| BitBufferMut::copy_from(&bits));

    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let values = values.as_slice::<F>();
    let offset = patches.offset();

    match_each_integer_ptype!(indices.ptype(), |I| {
        for (&index, &value) in indices.as_slice::<I>().iter().zip(values) {
            let index: usize = index.as_();
            // Patch indices are absolute; a slice keeps only those within its range.
            if let Some(local) = index.checked_sub(offset)
                && local < bits.len()
            {
                bits.set_to(local, predicate(value));
            }
        }
    });

    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}
