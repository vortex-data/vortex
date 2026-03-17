// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized execution logic for DictArray - takes from values using codes (indices).
//!
//! These functions bypass the generic `TakeExecute` pipeline to avoid unnecessary overhead
//! when decoding dictionaries. Since codes are already canonicalized as unsigned
//! `PrimitiveArray` indices, we skip dtype checks, unsigned conversion, and
//! execute-to-PrimitiveArray conversions that the generic path performs.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::bool::compute::take::take_valid_indices;
use crate::arrays::decimal::compute::take::take_to_buffer;
use crate::arrays::dict::TakeExecute;
use crate::arrays::dict::TakeReduce;
use crate::arrays::primitive::compute::take::take_primitive_direct;
use crate::arrays::varbinview::compute::take::take_views;
use crate::buffer::BufferHandle;
use crate::match_each_integer_ptype;
use crate::match_each_decimal_value_type;
use crate::validity::Validity;
use crate::builtins::ArrayBuiltins;
use crate::vtable::ValidityHelper;

/// Take from a canonical array using indices (codes), returning a new canonical array.
///
/// This is the core operation for dictionary decoding - it expands the dictionary
/// by looking up each code in the values array.
pub fn take_canonical(
    values: Canonical,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    Ok(match values {
        Canonical::Null(a) => Canonical::Null(take_null(&a, codes)),
        Canonical::Bool(a) => Canonical::Bool(take_bool(&a, codes)?),
        Canonical::Primitive(a) => Canonical::Primitive(take_primitive(&a, codes)?),
        Canonical::Decimal(a) => Canonical::Decimal(take_decimal(&a, codes)?),
        Canonical::VarBinView(a) => Canonical::VarBinView(take_varbinview(&a, codes)?),
        Canonical::List(a) => Canonical::List(take_listview(&a, codes)),
        Canonical::FixedSizeList(a) => {
            Canonical::FixedSizeList(take_fixed_size_list(&a, codes, ctx))
        }
        Canonical::Struct(a) => Canonical::Struct(take_struct(&a, codes)),
        Canonical::Extension(a) => Canonical::Extension(take_extension(&a, codes, ctx)),
    })
}

/// Compute result validity for dict decoding by combining values' validity with codes' validity.
///
/// This avoids going through VTable dispatch for the common cases where values are
/// non-nullable (just use codes' validity directly).
fn dict_take_validity(values_validity: &Validity, codes: &PrimitiveArray) -> VortexResult<Validity> {
    match values_validity {
        Validity::NonNullable => {
            // Values are non-nullable: result validity is determined solely by codes.
            let codes_validity = codes.validity();
            match codes_validity {
                Validity::NonNullable => Ok(Validity::NonNullable),
                // Codes dtype is nullable even if all valid
                other => Ok(other.clone()),
            }
        }
        Validity::AllValid => {
            // Values are all valid: result validity is determined solely by codes.
            Ok(match codes.validity() {
                Validity::NonNullable | Validity::AllValid => Validity::AllValid,
                other => other.clone(),
            })
        }
        Validity::AllInvalid => Ok(Validity::AllInvalid),
        Validity::Array(_) => {
            // Values have per-element validity: need to gather validity at code positions.
            // Fall back to standard Validity::take which handles null codes correctly.
            values_validity.take(&codes.clone().into_array())
        }
    }
}

/// Take for NullArray is trivial - just create a new NullArray with the new length.
fn take_null(_array: &NullArray, codes: &PrimitiveArray) -> NullArray {
    NullArray::new(codes.len())
}

/// Optimized bool take: directly gathers bits using codes without TakeExecute overhead.
fn take_bool(
    array: &BoolArray,
    codes: &PrimitiveArray,
) -> VortexResult<BoolArray> {
    let validity = dict_take_validity(array.validity(), codes)?;

    // For null codes, we need to fill them with valid indices before gathering bits.
    let codes_mask = codes.validity_mask()?;
    let buffer = match &codes_mask {
        Mask::AllTrue(_) => {
            match_each_integer_ptype!(codes.ptype(), |I| {
                take_valid_indices(&array.to_bit_buffer(), codes.as_slice::<I>())
            })
        }
        Mask::AllFalse(_) => {
            // All codes are null → all results are null, bits don't matter.
            vortex_buffer::BitBuffer::new_unset(codes.len())
        }
        Mask::Values(_) => {
            // Some codes are null: fill null positions with index 0 to avoid OOB.
            let codes_filled = codes
                .clone()
                .into_array()
                .fill_null(crate::scalar::Scalar::from(0).cast(codes.dtype())?)?;
            let codes_filled = codes_filled.as_::<crate::arrays::Primitive>().clone();
            match_each_integer_ptype!(codes_filled.ptype(), |I| {
                take_valid_indices(&array.to_bit_buffer(), codes_filled.as_slice::<I>())
            })
        }
    };

    Ok(BoolArray::new(buffer, validity))
}

/// Optimized primitive take: directly calls the SIMD/AVX2 kernel without TakeExecute overhead.
fn take_primitive(
    array: &PrimitiveArray,
    codes: &PrimitiveArray,
) -> VortexResult<PrimitiveArray> {
    let validity = dict_take_validity(array.validity(), codes)?;
    let result = take_primitive_direct(array, codes, validity)?;
    Ok(result.as_::<crate::arrays::Primitive>().clone())
}

/// Optimized decimal take: directly gathers decimal values without TakeExecute overhead.
fn take_decimal(
    array: &DecimalArray,
    codes: &PrimitiveArray,
) -> VortexResult<DecimalArray> {
    let validity = dict_take_validity(array.validity(), codes)?;

    let decimal = match_each_decimal_value_type!(array.values_type(), |D| {
        match_each_integer_ptype!(codes.ptype(), |I| {
            let buffer =
                take_to_buffer::<I, D>(codes.as_slice::<I>(), array.buffer::<D>().as_slice());
            // SAFETY: Take operation preserves decimal dtype and creates valid buffer.
            unsafe { DecimalArray::new_unchecked(buffer, array.decimal_dtype(), validity) }
        })
    });

    Ok(decimal)
}

/// Optimized VarBinView take: directly gathers views without TakeExecute overhead.
fn take_varbinview(
    array: &VarBinViewArray,
    codes: &PrimitiveArray,
) -> VortexResult<VarBinViewArray> {
    let validity = dict_take_validity(array.validity(), codes)?;
    let codes_mask = codes.validity_mask()?;
    let views_buffer = match_each_integer_ptype!(codes.ptype(), |I| {
        take_views(array.views(), codes.as_slice::<I>(), &codes_mask)
    });

    // SAFETY: taking views at valid indices maintains invariants; buffers are shared.
    unsafe {
        Ok(VarBinViewArray::new_handle_unchecked(
            BufferHandle::new_host(views_buffer.into_byte_buffer()),
            array.buffers().clone(),
            array
                .dtype()
                .union_nullability(codes.dtype().nullability()),
            validity,
        ))
    }
}

fn take_listview(array: &ListViewArray, codes: &PrimitiveArray) -> ListViewArray {
    <ListView as TakeReduce>::take(array, &codes.clone().into_array())
        .vortex_expect("take listview array")
        .vortex_expect("take listview should not return None")
        .as_::<ListView>()
        .clone()
}

fn take_fixed_size_list(
    array: &FixedSizeListArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> FixedSizeListArray {
    <FixedSizeList as TakeExecute>::take(array, &codes.clone().into_array(), ctx)
        .vortex_expect("take fixed size list array")
        .vortex_expect("take fixed size list should not return None")
        .as_::<FixedSizeList>()
        .clone()
}

fn take_struct(array: &StructArray, codes: &PrimitiveArray) -> StructArray {
    <crate::arrays::Struct as TakeReduce>::take(array, &codes.clone().into_array())
        .vortex_expect("take struct array")
        .vortex_expect("take struct should not return None")
        .as_::<crate::arrays::Struct>()
        .clone()
}

fn take_extension(
    array: &ExtensionArray,
    codes: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> ExtensionArray {
    <crate::arrays::Extension as TakeExecute>::take(array, &codes.clone().into_array(), ctx)
        .vortex_expect("take extension storage")
        .vortex_expect("take extension should not return None")
        .as_::<crate::arrays::Extension>()
        .clone()
}
