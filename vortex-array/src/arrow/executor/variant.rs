// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StructArray as ArrowStructArray;
use arrow_schema::Field;
use arrow_schema::Fields;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayVisitor;
use crate::arrays::VariantVTable;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;

pub(super) fn to_arrow_variant(
    array: ArrayRef,
    target_fields: Option<&Fields>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();
    let nulls = to_arrow_null_buffer(array.validity()?, len, ctx)?;
    let inner = match array.try_into::<VariantVTable>() {
        Ok(variant) => variant.child().clone(),
        Err(array) => array,
    };

    let named_children = inner.named_children();
    if named_children.is_empty() {
        vortex_bail!("Variant array has no children");
    }

    let mut metadata: Option<ArrayRef> = None;
    let mut value: Option<ArrayRef> = None;
    let mut typed_value: Option<ArrayRef> = None;

    for (name, child) in named_children {
        match name.as_str() {
            "validity" => {}
            "metadata" => metadata = Some(child),
            "value" => value = Some(child),
            "typed_value" => typed_value = Some(child),
            _ => {
                vortex_bail!("Unsupported variant child {name}");
            }
        }
    }

    let metadata = match metadata {
        Some(metadata) => metadata,
        None => vortex_bail!("Variant array missing metadata child"),
    };

    let mut ordered: Vec<(String, ArrayRef)> = vec![("metadata".to_string(), metadata.clone())];
    if let Some(value) = value.clone() {
        ordered.push(("value".to_string(), value));
    }
    if let Some(typed_value) = typed_value.clone() {
        ordered.push(("typed_value".to_string(), typed_value));
    }

    let (fields, arrays) = if let Some(fields) = target_fields {
        let mut arrays = Vec::with_capacity(fields.len());
        for field in fields.iter() {
            let child = match field.name().as_str() {
                "metadata" => Some(&metadata),
                "value" => value.as_ref(),
                "typed_value" => typed_value.as_ref(),
                other => {
                    vortex_bail!("Unsupported variant field {other}");
                }
            };

            let Some(child) = child else {
                vortex_bail!("Variant array missing child for field {}", field.name());
            };

            arrays.push(child.clone().execute_arrow(Some(field.data_type()), ctx)?);
        }

        // Ensure we didn't silently drop any children
        vortex_ensure!(
            fields.len() == ordered.len(),
            "Variant array has {} children but target Arrow type has {} fields",
            ordered.len(),
            fields.len()
        );

        (fields.clone(), arrays)
    } else {
        let mut fields = Vec::with_capacity(ordered.len());
        let mut arrays = Vec::with_capacity(ordered.len());

        for (name, child) in ordered {
            let arrow = child.clone().execute_arrow(None, ctx)?;
            fields.push(Field::new(
                name,
                arrow.data_type().clone(),
                child.dtype().is_nullable(),
            ));
            arrays.push(arrow);
        }

        (Fields::from(fields), arrays)
    };

    Ok(Arc::new(ArrowStructArray::try_new(fields, arrays, nulls)?))
}
