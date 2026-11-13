// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::bytes_dict_builder;
use decimal::decimal_dict_builder;
use primitive::primitive_dict_builder;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Decimal;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;

mod bytes;
mod decimal;
mod primitive;
mod values;

#[derive(Clone)]
pub struct DictConstraints {
    /// Limits the encoded storage for dictionary values.
    pub max_bytes: usize,
    /// Limits the number of dictionary values.
    pub max_len: usize,
}

pub const UNCONSTRAINED: DictConstraints = DictConstraints {
    max_bytes: usize::MAX,
    max_len: usize::MAX,
};

/// Returns the narrowest unsigned code type that can index `max_len` dictionary values.
///
/// Codes count from zero, so a code type indexes one more value than its maximum: `u8` codes
/// index up to 256 values.
pub const fn codes_ptype_for_max_len(max_len: u64) -> PType {
    match max_len {
        len if len <= 1 << u8::BITS => PType::U8,
        len if len <= 1 << u16::BITS => PType::U16,
        len if len <= 1 << u32::BITS => PType::U32,
        _ => PType::U64,
    }
}

pub trait DictEncoder: Send {
    /// Assign dictionary codes to the given input array.
    fn encode(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray>;

    /// Clear the encoder state to make it ready for a new round of decoding.
    fn reset(&mut self) -> ArrayRef;

    /// Returns the PType of the codes this encoder produces.
    fn codes_ptype(&self) -> PType;
}

/// Create a dictionary encoder using the default allocator.
#[deprecated(note = "use `dict_encoder_in` with an explicit allocator")]
pub fn dict_encoder(array: &ArrayRef, constraints: &DictConstraints) -> Box<dyn DictEncoder> {
    dict_encoder_in(
        array,
        constraints,
        BufferAllocatorRef::statically_allocated(),
    )
}

/// Create a dictionary encoder using `allocator`.
pub fn dict_encoder_in(
    array: &ArrayRef,
    constraints: &DictConstraints,
    allocator: BufferAllocatorRef,
) -> Box<dyn DictEncoder> {
    // Choose by dtype: the encoders execute an input that is not already canonical, such as a
    // lazy slice.
    match array.dtype() {
        DType::Primitive(ptype, nullability) => match_each_native_ptype!(*ptype, |P| {
            primitive_dict_builder::<P>(*nullability, constraints, allocator)
        }),
        DType::Utf8(_) | DType::Binary(_) => {
            bytes_dict_builder(array.dtype().clone(), constraints, allocator)
        }
        DType::Decimal(decimal_dtype, nullability) => {
            // A canonical array may store values that do not fit the precision's values type.
            let canonical = decimal_dtype.canonical_values_type();
            let values_type = array
                .as_opt::<Decimal>()
                .map_or(canonical, |decimal| decimal.values_type().max(canonical));
            match_each_decimal_value_type!(values_type, |D| {
                decimal_dict_builder::<D>(*decimal_dtype, *nullability, constraints, allocator)
            })
        }
        dtype => {
            vortex_panic!(
                "Can only dictionary encode primitive, decimal or binary arrays, got {dtype}"
            )
        }
    }
}

/// Encode an array as a `DictArray` subject to the given constraints.
///
/// The codes use the narrowest unsigned integer type that can address the dictionary, so callers
/// have nothing left to narrow.
///
/// Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for external compatibility.
pub fn dict_encode_with_constraints(
    array: &ArrayRef,
    constraints: &DictConstraints,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictArray> {
    // Every row contributes at most one dictionary entry, so the array bounds the dictionary
    // length whatever the caller asked for. Encoders pick their code width up front from this
    // bound, so tightening it keeps a one-shot encode from writing `u64` codes for a dictionary
    // that cannot hold more than a handful of entries.
    let constraints = DictConstraints {
        max_bytes: constraints.max_bytes,
        max_len: constraints.max_len.min(array.len().max(1)),
    };
    let mut encoder = dict_encoder_in(array, &constraints, ctx.allocator().clone());
    let codes = encoder.encode(array, ctx)?;
    let values = encoder.reset();
    let codes = narrow_codes(codes, values.len(), ctx)?;
    // SAFETY: The encoding process will produce a value set of codes and values
    // All values in the dictionary are guaranteed to be referenced by at least one code
    // since we build the dictionary from the codes we observe during encoding
    unsafe {
        Ok(DictArray::new_unchecked(codes.into_array(), values).set_all_values_referenced(true))
    }
}

/// Narrow codes addressing a `dict_len`-entry dictionary to the smallest type that can hold them.
///
/// Encoders hand out consecutive codes and only ever add an entry they immediately use, so the
/// largest code is `dict_len - 1`. That makes the narrowest code type known without the scan
/// [`PrimitiveArrayExt::narrow`] would otherwise run over the codes.
fn narrow_codes(
    codes: PrimitiveArray,
    dict_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let ptype = codes_ptype_for_max_len(dict_len as u64);
    if codes.ptype() == ptype {
        return Ok(codes);
    }
    codes
        .as_ref()
        .cast(DType::Primitive(ptype, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(ctx)
}

pub fn dict_encode(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DictArray> {
    let dict_array = dict_encode_with_constraints(array, &UNCONSTRAINED, ctx)?;
    if dict_array.len() != array.len() {
        vortex_bail!(
            "must have encoded all {} elements, but only encoded {}",
            array.len(),
            dict_array.len(),
        );
    }
    Ok(dict_array)
}
