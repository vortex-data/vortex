// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Apply a per-count conversion to a temporal array or scalar.
//!
//! Each temporal conversion that changes the stored counts, whether an exact `cast` or a named
//! function such as `vortex_functions::expr::change_time_unit`, maps each count on its
//! own and writes the result in the storage width of the target type. This module owns that
//! plumbing, so that each conversion supplies only its per-count arithmetic.

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::extension::ExtDTypeRef;
use crate::scalar::Scalar;

/// Apply `convert` to each valid count of a temporal array, and return an array of `target`.
///
/// `target` **must** be a temporal dtype. A null row stays null, and `convert` never sees its
/// payload. The conversion fails on the first count that `convert` rejects or that does not fit in
/// the storage width of `target`.
pub fn map_counts(
    input: ArrayRef,
    target: &DType,
    ctx: &mut ExecutionCtx,
    convert: impl Fn(i64) -> VortexResult<i64>,
) -> VortexResult<ArrayRef> {
    if input.is_empty() {
        return Ok(Canonical::empty(target).into_array());
    }
    // A constant converts its one value, so the result stays constant.
    if let Some(scalar) = input.as_constant() {
        let converted = map_scalar_count(scalar, target, convert)?;
        return Ok(ConstantArray::new(converted, input.len()).into_array());
    }

    let ext = input.execute::<ExtensionArray>(ctx)?;
    let storage = ext.storage_array().clone().execute::<PrimitiveArray>(ctx)?;
    let target = target
        .as_extension_opt()
        .vortex_expect("`map_counts` requires a temporal target dtype");

    match storage.ptype() {
        PType::I32 => map_storage_counts::<i32>(&storage, target, ctx, convert),
        PType::I64 => map_storage_counts::<i64>(&storage, target, ctx, convert),
        ptype => unsupported_storage(ptype),
    }
}

/// Apply `convert` to the count of a temporal scalar, and return a scalar of `target`.
///
/// `target` **must** be a temporal dtype. A null scalar becomes a null of `target`, which therefore
/// must be nullable.
pub(crate) fn map_scalar_count(
    input: &Scalar,
    target: &DType,
    convert: impl Fn(i64) -> VortexResult<i64>,
) -> VortexResult<Scalar> {
    if input.is_null() {
        vortex_ensure!(
            target.is_nullable(),
            "Expected a nullable target for a null value, got {target}"
        );
        return Ok(Scalar::null(target.clone()));
    }

    let target_ext = target
        .as_extension_opt()
        .vortex_expect("`map_scalar_count` requires a temporal target dtype");
    let storage = input.as_extension().to_storage_scalar();
    let value = match storage.as_primitive().ptype() {
        PType::I32 => storage.as_primitive().typed_value::<i32>().map(i64::from),
        PType::I64 => storage.as_primitive().typed_value::<i64>(),
        ptype => unsupported_storage(ptype),
    }
    .vortex_expect("the null check above returns early for a null input");
    let value = convert(value)?;

    let storage = match target_ext.storage_dtype().as_ptype() {
        PType::I32 => Scalar::primitive(i32::try_from(value)?, target.nullability()),
        PType::I64 => Scalar::primitive(value, target.nullability()),
        ptype => unsupported_storage(ptype),
    };
    Scalar::try_extension_ref(target_ext.clone(), storage)
}

/// Select the output storage width and apply the conversion to valid rows.
fn map_storage_counts<T: NativePType + Into<i64>>(
    storage: &PrimitiveArray,
    target: &ExtDTypeRef,
    ctx: &mut ExecutionCtx,
    convert: impl Fn(i64) -> VortexResult<i64>,
) -> VortexResult<ArrayRef> {
    let values = storage.as_slice::<T>();
    let validity = storage.validity()?.cast_nullability(
        target.storage_dtype().nullability(),
        values.len(),
        ctx,
    )?;
    let mask = validity.execute_mask(values.len(), ctx)?;

    let output = match target.storage_dtype().as_ptype() {
        PType::I32 => PrimitiveArray::new(
            map_valid_counts::<T, i32>(values, &mask, convert)?,
            validity,
        ),
        PType::I64 => PrimitiveArray::new(
            map_valid_counts::<T, i64>(values, &mask, convert)?,
            validity,
        ),
        ptype => unsupported_storage(ptype),
    };

    Ok(ExtensionArray::try_new(target.clone(), output.into_array())?.into_array())
}

/// Convert each valid count and fit the result into the output width.
fn map_valid_counts<T, O>(
    values: &[T],
    mask: &Mask,
    convert: impl Fn(i64) -> VortexResult<i64>,
) -> VortexResult<Buffer<O>>
where
    T: NativePType + Into<i64>,
    O: NativePType + TryFrom<i64>,
    VortexError: From<<O as TryFrom<i64>>::Error>,
{
    let mut output = BufferMut::<O>::with_capacity(values.len());
    for (&value, valid) in values.iter().zip(mask.iter()) {
        // A null row can hold any payload, such as `i64::MAX`, so converting it could fail. Its
        // output is only a placeholder under the null.
        let count = if valid { convert(value.into())? } else { 0 };
        output.push(O::try_from(count)?);
    }
    Ok(output.freeze())
}

/// Panic on a storage type that no built-in temporal dtype has.
///
/// The `storage_dtype` of each temporal extension type returns `i32` or `i64`, and an extension
/// array or scalar must match the storage dtype of its type.
fn unsupported_storage(ptype: PType) -> ! {
    vortex_panic!("Expected i32 or i64 temporal storage, got {ptype}")
}
