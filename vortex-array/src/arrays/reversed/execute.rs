// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::bool::BoolArrayExt as _;
use crate::arrays::primitive::PrimitiveArrayExt as _;
use crate::arrays::struct_::StructArrayExt as _;
use crate::canonical::Canonical;
use crate::executor::ExecutionCtx;
use crate::match_each_native_ptype;
use crate::validity::Validity;
use crate::{ArrayRef, IntoArray as _};

/// Reverses a canonical array, dispatching to type-specific fast paths where possible.
///
/// Fast paths:
/// - `Bool`: reverses the bit buffer directly via `value_unchecked` — O(n), no extra allocation.
/// - `Primitive`: reverses the element buffer directly — O(n), no extra allocation.
/// - `Struct`: reverses each field lazily via [`ArrayRef::reverse`] — allows per-field
///   optimisations (e.g. the `Dict` reduce rule fires on dict-encoded fields).
///
/// All other canonical variants fall back to a reversed-index `take`, which is equivalent
/// to the generic path but is deferred to decode time.
pub(super) fn reverse_canonical(
    child: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let n = child.len();
    if n <= 1 {
        return Ok(child.clone());
    }

    let canonical = child.clone().execute::<Canonical>(ctx)?;
    Ok(match canonical {
        Canonical::Bool(a) => reverse_bool(&a)?.into_array(),
        Canonical::Primitive(a) => reverse_primitive(&a)?.into_array(),
        Canonical::Struct(a) => reverse_struct(&a)?.into_array(),
        // All other canonical types: reverse via take with reversed indices.
        _ => {
            let indices = PrimitiveArray::from_iter((0u64..n as u64).rev()).into_array();
            child.take(indices)?
        }
    })
}

/// Reverses a `BoolArray` by reading each bit in reverse order.
///
/// Uses `value_unchecked` for O(n) direct bit access with no intermediate `Vec` allocation,
/// and correctly handles the buffer's bit offset.
fn reverse_bool(array: &BoolArray) -> VortexResult<BoolArray> {
    let validity = reverse_validity(array.validity()?)?;
    let bits = array.to_bit_buffer();
    let n = bits.len();
    let reversed = BitBuffer::collect_bool(n, |i| {
        // SAFETY: `n - 1 - i` is in `[0, n)` since `i` is in `[0, n)`.
        unsafe { bits.value_unchecked(n - 1 - i) }
    });
    Ok(BoolArray::new(reversed, validity))
}

/// Reverses a `PrimitiveArray` by iterating the typed buffer backwards.
///
/// This is O(n × element_width) and sequential in both reads and writes, so it is
/// highly cache-friendly and eligible for auto-vectorisation.
fn reverse_primitive(array: &PrimitiveArray) -> VortexResult<PrimitiveArray> {
    let validity = reverse_validity(array.validity()?)?;
    match_each_native_ptype!(array.ptype(), |T| {
        let reversed: Vec<T> = array.as_slice::<T>().iter().rev().copied().collect();
        Ok(PrimitiveArray::new(Buffer::from(reversed), validity))
    })
}

/// Reverses a `StructArray` by lazily reversing each child field.
///
/// Each field is reversed via [`ArrayRef::reverse`], which in turn runs the optimizer.
/// For dict-encoded fields this fires the `ReverseReduce for Dict` rule, so only the
/// (small) codes array is reversed; the values dictionary remains untouched.
fn reverse_struct(array: &StructArray) -> VortexResult<StructArray> {
    let validity = reverse_validity(array.struct_validity())?;
    let names = array.names().clone();
    let n = array.len();
    let reversed_fields = array
        .iter_unmasked_fields()
        .map(|field| field.reverse())
        .collect::<VortexResult<Vec<ArrayRef>>>()?;
    StructArray::try_new(names, reversed_fields, n, validity)
}

/// Reverses a [`Validity`] value.
///
/// `NonNullable`, `AllValid`, and `AllInvalid` are identity under reversal.
/// `Array` variants are reversed lazily: `arr.reverse()` creates a
/// `ReversedArray` wrapper that is further optimised at decode time.
fn reverse_validity(validity: Validity) -> VortexResult<Validity> {
    match validity {
        Validity::NonNullable => Ok(Validity::NonNullable),
        Validity::AllValid => Ok(Validity::AllValid),
        Validity::AllInvalid => Ok(Validity::AllInvalid),
        Validity::Array(arr) => Ok(Validity::Array(arr.reverse()?)),
    }
}
