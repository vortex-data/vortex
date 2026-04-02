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
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::dtype::FieldName;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantData;

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

    // Convert back to Vortex
    let result_variant = ArrowVariantArray::try_new(
        arrow_result
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| vortex_err!("variant_get did not return a StructArray"))?,
    )
    .map_err(|e| vortex_err!("failed to create VariantArray from result: {e}"))?;

    ParquetVariantData::from_arrow_variant(&result_variant)
}
