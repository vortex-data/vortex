// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::bytes_dict_builder;
use primitive::primitive_dict_builder;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinView;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;

mod bytes;
mod primitive;

#[derive(Clone)]
pub struct DictConstraints {
    pub max_bytes: usize,
    pub max_len: usize,
}

pub const UNCONSTRAINED: DictConstraints = DictConstraints {
    max_bytes: usize::MAX,
    max_len: usize::MAX,
};

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
    let dict_builder: Box<dyn DictEncoder> = if let Some(pa) = array.as_opt::<Primitive>() {
        match_each_native_ptype!(pa.ptype(), |P| {
            primitive_dict_builder::<P>(pa.dtype().nullability(), constraints, allocator)
        })
    } else if let Some(vbv) = array.as_opt::<VarBinView>() {
        bytes_dict_builder(vbv.dtype().clone(), constraints, allocator)
    } else if let Some(vb) = array.as_opt::<VarBin>() {
        bytes_dict_builder(vb.dtype().clone(), constraints, allocator)
    } else {
        vortex_panic!("Can only encode primitive or varbin/view arrays")
    };
    dict_builder
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
    let ptype = if dict_len <= u8::MAX as usize + 1 {
        PType::U8
    } else if dict_len <= u16::MAX as usize + 1 {
        PType::U16
    } else if dict_len <= u32::MAX as usize + 1 {
        PType::U32
    } else {
        PType::U64
    };
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
