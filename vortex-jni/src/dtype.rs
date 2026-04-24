// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bridge functions between a Vortex [`DType`] and Arrow's C Data Interface
//! [`FFI_ArrowSchema`]. Java receives DType information exclusively as Arrow schema.

use std::ptr;

use arrow_array::ffi::FFI_ArrowSchema;
use arrow_schema::DataType;
use arrow_schema::FieldRef;
use arrow_schema::Fields;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexResult;

/// Export a Vortex [`DType`] to the Arrow C Data Interface struct at `schema_addr`. Views
/// (Utf8View/BinaryView) are downgraded to regular Utf8/Binary so Spark and other consumers
/// without view support can read them.
pub(crate) fn export_dtype_to_arrow(dtype: &DType, schema_addr: i64) -> VortexResult<()> {
    let arrow_schema = dtype.to_arrow_schema()?;
    let viewless = strip_views(DataType::Struct(arrow_schema.fields().clone()));
    let fields = match viewless {
        DataType::Struct(fields) => fields,
        _ => unreachable!("Vortex DType always exports as a struct"),
    };
    let schema = arrow_schema::Schema::new(fields);
    let ffi_schema = FFI_ArrowSchema::try_from(&schema)?;
    unsafe {
        ptr::write(schema_addr as *mut FFI_ArrowSchema, ffi_schema);
    }
    Ok(())
}

/// Replace view-based Arrow types with their non-view counterparts throughout the tree.
pub(crate) fn strip_views(data_type: DataType) -> DataType {
    match data_type {
        DataType::BinaryView => DataType::Binary,
        DataType::Utf8View => DataType::Utf8,
        DataType::List(inner) | DataType::ListView(inner) => {
            let new_inner = (*inner)
                .clone()
                .with_data_type(strip_views(inner.data_type().clone()));
            DataType::List(FieldRef::new(new_inner))
        }
        DataType::LargeList(inner) | DataType::LargeListView(inner) => {
            let new_inner = (*inner)
                .clone()
                .with_data_type(strip_views(inner.data_type().clone()));
            DataType::LargeList(FieldRef::new(new_inner))
        }
        DataType::Struct(fields) => {
            let viewless_fields: Vec<FieldRef> = fields
                .iter()
                .map(|field_ref| {
                    let field = (**field_ref).clone();
                    let data_type = field.data_type().clone();
                    FieldRef::new(field.with_data_type(strip_views(data_type)))
                })
                .collect();
            DataType::Struct(Fields::from(viewless_fields))
        }
        DataType::FixedSizeList(inner, size) => {
            let new_inner = (*inner)
                .clone()
                .with_data_type(strip_views(inner.data_type().clone()));
            DataType::FixedSizeList(FieldRef::new(new_inner), size)
        }
        dt => dt,
    }
}

/// Decode an [`FFI_ArrowSchema`] pointed to by `schema_addr` into a Vortex [`DType`].
pub(crate) fn import_dtype_from_arrow(schema_addr: i64) -> VortexResult<DType> {
    let ffi_schema = unsafe { &*(schema_addr as *const FFI_ArrowSchema) };
    let arrow_schema = arrow_schema::Schema::try_from(ffi_schema)?;
    Ok(DType::from_arrow(&arrow_schema))
}
