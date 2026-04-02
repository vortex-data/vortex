// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execute-parent kernel for `variant_get` on `ParquetVariantArray`.
//!
//! Delegates to `parquet_variant_compute::variant_get` after converting to Arrow.

use std::sync::Arc;

use parquet_variant::VariantPathElement;
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
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
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
        let field_name: &FieldName = parent.options;
        variant_get_impl(array, field_name, ctx).map(Some)
    }
}

fn variant_get_impl(
    array: ArrayView<'_, ParquetVariant>,
    field_name: &FieldName,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Convert to Arrow VariantArray
    let arrow_variant = array.to_arrow(ctx)?;

    // Build path for a single field access
    let path_element = VariantPathElement::Field {
        name: field_name.as_ref().into(),
    };
    let options = GetOptions::new_with_path(vec![path_element].into());

    // Delegate to the parquet-variant-compute kernel.
    // With as_type = None, the result is itself a VariantArray.
    let inner: Arc<dyn arrow_array::Array> = Arc::new(arrow_variant.into_inner());
    let arrow_result = parquet_variant_compute::variant_get(&inner, options)
        .map_err(|e| vortex_err!("variant_get failed: {e}"))?;

    // Convert back to Vortex.
    let result_variant = ArrowVariantArray::try_new(
        arrow_result
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| vortex_err!("variant_get did not return a StructArray"))?,
    )
    .map_err(|e| vortex_err!("failed to create VariantArray from result: {e}"))?;

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
        .map(|v| ArrayRef::from_arrow(v as &dyn arrow_array::Array, true))
        .transpose()?;
    let typed_value = result_variant
        .typed_value_field()
        .map(|tv| ArrayRef::from_arrow(tv.as_ref(), true))
        .transpose()?;

    let pv = ParquetVariant::try_new(validity, metadata, value, typed_value)?;
    debug_assert_eq!(
        pv.dtype(),
        &DType::Variant(Nullability::Nullable),
        "variant_get result must be nullable"
    );
    Ok(VariantArray::new(pv.into_array()).into_array())
}
