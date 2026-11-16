// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shreds unshredded Parquet Variant storage.
//!
//! A reader of an unshredded Variant decodes every value to extract one field of it. Shredding
//! the fields that the values have in common moves each into a typed column of its own, which a
//! reader extracts without touching the rest.
//!
//! Shredding never changes what a Variant holds: a row whose value does not fit the schema keeps
//! it in the residual binary value. A value fits a typed column only if it has the type of the
//! column, so the schema only has to fit most rows to be worth it, and [`InferredShredding`]
//! infers it from a sample.

mod exact;
mod infer;
#[cfg(test)]
mod tests;

pub use infer::InferredShredding;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantArraySlotsExt;

/// Shreds the unshredded storage `array` by `schema`.
///
/// The result holds the same values as `array`, in Parquet Variant storage with a shredded
/// child, and is `nullable` or not.
pub(crate) fn shred(
    array: &ArrayView<'_, ParquetVariant>,
    schema: &DType,
    nullable: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let value = array
        .value()
        .ok_or_else(|| vortex_err!("unshredded Parquet Variant storage has no value"))?;

    let len = array.len();
    let valid = &array.parquet_variant_validity().execute_mask(len, ctx)?
        & &value.validity()?.execute_mask(len, ctx)?;
    let metadata = array.metadata().clone().execute::<VarBinViewArray>(ctx)?;
    let value = value.clone().execute::<VarBinViewArray>(ctx)?;

    let (value, typed_value) = exact::shred(&metadata, &value, &valid, schema, ctx.allocator())?;

    let validity = array.validity()?;
    let validity = if nullable {
        validity.into_nullable()
    } else {
        validity
    };
    ParquetVariant::try_new(
        validity,
        array.metadata().clone(),
        Some(value),
        Some(typed_value),
    )
    .map(IntoArray::into_array)
}
