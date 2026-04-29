// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_array::StructArray;
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
use vortex_array::arrays::Variant;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::variant::VariantArrayExt;
use vortex_array::arrow::FromArrowArray;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement as VortexVariantPathElement;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantData;

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
        // For now, we delegate to the Arrow Parquet Variant implementation rather than
        // re-implementing path traversal and shredded/unshredded merge semantics here.
        let arrow_variant = array.to_arrow(ctx)?;

        let path = parent
            .options
            .path()
            .iter()
            .map(|element| match element {
                VortexVariantPathElement::Field(name) => ArrowVariantPathElement::Field {
                    name: name.to_string().into(),
                },
                VortexVariantPathElement::Index(index) => {
                    ArrowVariantPathElement::Index { index: *index }
                }
            })
            .collect::<Vec<_>>();
        let inner: Arc<dyn ArrowArray> = Arc::new(arrow_variant.into_inner());
        let mut arrow_options = GetOptions::new_with_path(VariantPath::new(path));

        if let Some(as_dtype) = parent.options.effective_as_dtype() {
            arrow_options = arrow_options.with_as_type(Some(FieldRef::new(Field::new(
                "result",
                as_dtype.to_arrow_dtype()?,
                as_dtype.is_nullable(),
            ))));
        }

        let arrow_result = parquet_variant_compute::variant_get(&inner, arrow_options)
            .map_err(|e| vortex_err!("variant_get failed: {e}"))?;

        if parent.options.effective_as_dtype().is_some() {
            return ArrayRef::from_arrow(arrow_result.as_ref(), true).map(Some);
        }

        let result_variant = ArrowVariantArray::try_new(
            arrow_result
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| vortex_err!("variant_get did not return a StructArray"))?,
        )
        .map_err(|e| vortex_err!("failed to create VariantArray from result: {e}"))?;

        // Untyped `variant_get` always returns `Variant(Nullable)`. Arrow may omit the
        // top-level null buffer for all-valid batches, which `from_arrow_variant` would
        // otherwise interpret as `Variant(NonNullable)`.
        force_nullable_result(ParquetVariantData::from_arrow_variant(&result_variant)?).map(Some)
    }
}

/// Preserves `Variant(Nullable)` for untyped `variant_get` results.
///
/// Arrow may return an all-valid batch without a top-level null buffer; without this fix-up,
/// `from_arrow_variant` would infer `Variant(NonNullable)`.
fn force_nullable_result(result: ArrayRef) -> VortexResult<ArrayRef> {
    if result.dtype().is_nullable() {
        return Ok(result);
    }

    let variant = result.as_::<Variant>();
    let core_storage = variant.core_storage();
    let core_storage = core_storage.as_opt::<ParquetVariant>().ok_or_else(|| {
        vortex_err!(
            "variant_get expected ParquetVariant core_storage, got {}",
            core_storage.encoding_id()
        )
    })?;
    let nullable_core_storage = ParquetVariant::try_new(
        Validity::AllValid,
        core_storage.metadata_array().clone(),
        core_storage.value_array().cloned(),
        core_storage.typed_value_array().cloned(),
    )?;
    let nullable_core_storage = nullable_core_storage.into_array();
    let rebuilt = if let Some((source_encoding_id, slot_name)) = variant.derived_shredded_source() {
        VariantArray::try_new_derived(nullable_core_storage, source_encoding_id, slot_name)?
    } else {
        VariantArray::try_new(nullable_core_storage, variant.shredded())?
    };
    Ok(rebuilt.into_array())
}
