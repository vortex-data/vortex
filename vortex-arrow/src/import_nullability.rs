// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Apply declared nullability during Arrow array import.
//!
//! Leaf and container conversions share the [session's import rules]. Extension storage imports
//! use a field derived from the storage dtype so they can apply those rules without rebuilding the
//! Arrow array.
//!
//! [session's import rules]: crate::ArrowSession::from_arrow_array

use std::sync::Arc;

use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::ArrayRef;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::convert::nulls;

/// Build the leaf with the validity needed to defer any fill until execution.
pub(crate) fn import_leaf(
    arrow_nulls: Option<&NullBuffer>,
    nullable: bool,
    import: impl FnOnce(Validity) -> ArrayRef,
) -> VortexResult<ArrayRef> {
    let needs_fill = !nullable && arrow_nulls.is_some_and(|nulls| nulls.null_count() > 0);
    let validity = nulls(arrow_nulls, nullable || needs_fill)?;
    let array = import(validity);
    if !needs_fill {
        return Ok(array);
    }

    let zero = Scalar::zero_value(&array.dtype().as_nonnullable());
    array.fill_null(zero)
}

/// Convert an Arrow null buffer for a container that keeps its storage.
///
/// See [`crate::ArrowSession::from_arrow_array`] for the nullability rules.
pub fn level_validity(null_buffer: Option<&NullBuffer>, nullable: bool) -> VortexResult<Validity> {
    if nullable {
        nulls(null_buffer, true)
    } else {
        Ok(Validity::NonNullable)
    }
}

/// Apply a compatible Vortex storage dtype's nullability to the Arrow import field.
///
/// Each extension importer owns its storage layout, so recursion stops at nested extension
/// fields.
pub(crate) fn storage_import_field(field: &Field, dtype: &DType) -> Field {
    let field = field.clone().with_nullable(dtype.is_nullable());

    let data_type = match (field.data_type(), dtype) {
        (DataType::Struct(fields), DType::Struct(declared, _)) => DataType::Struct(
            fields
                .iter()
                .zip(declared.fields())
                .map(|(child, dtype)| storage_import_field(child, &dtype))
                .collect(),
        ),
        (DataType::List(child), DType::List(element, _)) => {
            DataType::List(Arc::new(storage_import_field(child, element)))
        }
        (DataType::LargeList(child), DType::List(element, _)) => {
            DataType::LargeList(Arc::new(storage_import_field(child, element)))
        }
        (DataType::ListView(child), DType::List(element, _)) => {
            DataType::ListView(Arc::new(storage_import_field(child, element)))
        }
        (DataType::LargeListView(child), DType::List(element, _)) => {
            DataType::LargeListView(Arc::new(storage_import_field(child, element)))
        }
        (DataType::FixedSizeList(child, size), DType::FixedSizeList(element, ..)) => {
            DataType::FixedSizeList(Arc::new(storage_import_field(child, element)), *size)
        }
        (DataType::Map(entries, keys_sorted), DType::Map(map, _)) => {
            let DataType::Struct(fields) = entries.data_type() else {
                vortex_panic!(
                    "The Arrow storage dtype check requires struct map entries, got {}",
                    entries.data_type()
                );
            };
            let fields = [
                Arc::new(storage_import_field(&fields[0], &map.key_dtype())),
                Arc::new(storage_import_field(&fields[1], &map.value_dtype())),
            ];
            let entries = entries
                .as_ref()
                .clone()
                .with_data_type(DataType::Struct(fields.into()));
            DataType::Map(Arc::new(entries), *keys_sorted)
        }
        (DataType::Dictionary(key, value), _) => {
            let value_field = field.clone().with_data_type(value.as_ref().clone());
            let value = storage_import_field(&value_field, dtype);
            DataType::Dictionary(key.clone(), Box::new(value.data_type().clone()))
        }
        (DataType::RunEndEncoded(ends, value), _) => DataType::RunEndEncoded(
            Arc::clone(ends),
            Arc::new(storage_import_field(value, dtype)),
        ),
        _ => return field,
    };

    field.with_data_type(data_type)
}
