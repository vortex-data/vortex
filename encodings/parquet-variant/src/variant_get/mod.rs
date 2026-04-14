// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execute-parent kernel for `variant_get` on `ParquetVariantArray`.
//!
//! Delegates to `parquet_variant_compute::variant_get` after converting to Arrow.

use std::sync::Arc;

use arrow_schema::Field;
use arrow_schema::FieldRef;
use parquet_variant::VariantPath;
use parquet_variant::VariantPathElement as ArrowVariantPathElement;
use parquet_variant_compute::GetOptions;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::variant_get::VariantGetOptions;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement as VortexVariantPathElement;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub(crate) struct VariantGetExecuteParent;

impl ExecuteParentKernel<ParquetVariant> for VariantGetExecuteParent {
    type Parent = ExactScalarFn<VariantGet>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, ParquetVariant>,
        parent: ScalarFnArrayView<'_, VariantGet>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        variant_get_impl(array, parent.options, ctx).map(Some)
    }
}

fn variant_get_impl(
    array: ArrayView<'_, ParquetVariant>,
    options: &VariantGetOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Convert to Arrow VariantArray
    let arrow_variant = array.to_arrow(ctx)?;

    let path = options
        .path()
        .iter()
        .cloned()
        .map(|element| match element {
            VortexVariantPathElement::Field(name) => ArrowVariantPathElement::Field {
                name: name.to_string().into(),
            },
            VortexVariantPathElement::Index(index) => ArrowVariantPathElement::Index { index },
        })
        .collect::<Vec<_>>();
    let mut arrow_options = GetOptions::new_with_path(VariantPath::new(path));
    if let Some(as_dtype) = options.effective_as_dtype() {
        arrow_options = arrow_options.with_as_type(Some(FieldRef::new(Field::new(
            "result",
            as_dtype.to_arrow_dtype()?,
            as_dtype.is_nullable(),
        ))));
    }

    // Delegate to the parquet-variant-compute kernel.
    // With as_type = None, the result is itself a VariantArray.
    let inner: Arc<dyn arrow_array::Array> = Arc::new(arrow_variant.into_inner());
    let arrow_result = parquet_variant_compute::variant_get(&inner, arrow_options)
        .map_err(|e| vortex_err!("variant_get failed: {e}"))?;

    if options.effective_as_dtype().is_some() {
        return ArrayRef::from_arrow(arrow_result.as_ref(), true);
    }

    // Convert back to Vortex.
    let result_variant = ArrowVariantArray::try_new(
        arrow_result
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| vortex_err!("variant_get did not return a StructArray"))?,
    )
    .map_err(|e| vortex_err!("failed to create VariantArray from result: {e}"))?;
    let value_nullable = result_variant
        .inner()
        .fields()
        .iter()
        .find(|field| field.name() == "value")
        .map(|field| field.is_nullable())
        .unwrap_or(false);
    let typed_value_nullable = result_variant
        .inner()
        .fields()
        .iter()
        .find(|field| field.name() == "typed_value")
        .map(|field| field.is_nullable())
        .unwrap_or(false);

    // Ensure the result is always nullable (matching variant_get's return_dtype).
    // Arrow may return a non-nullable result when no nulls are present.
    let validity = result_variant
        .nulls()
        .map(|nulls| {
            if nulls.null_count() == nulls.len() {
                Validity::AllInvalid
            } else {
                Validity::from(BitBuffer::from(nulls.inner().clone()))
            }
        })
        .unwrap_or(Validity::AllValid);

    let metadata = ArrayRef::from_arrow(
        result_variant.metadata_field() as &dyn arrow_array::Array,
        false,
    )?;
    let value = result_variant
        .value_field()
        .map(|v| ArrayRef::from_arrow(v as &dyn arrow_array::Array, value_nullable))
        .transpose()?;
    let typed_value = result_variant
        .typed_value_field()
        .map(|tv| ArrayRef::from_arrow(tv.as_ref(), typed_value_nullable))
        .transpose()?;

    let pv = ParquetVariant::try_new(validity, metadata, value, typed_value)?;
    debug_assert_eq!(
        pv.dtype(),
        &DType::Variant(Nullability::Nullable),
        "variant_get result must be nullable"
    );
    Ok(VariantArray::new(pv.into_array()).into_array())
}
