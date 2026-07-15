// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bridge functions between a Vortex [`DType`] and Arrow's C Data Interface
//! [`FFI_ArrowSchema`]. Java receives DType information exclusively as Arrow schema.

use std::ptr;

use arrow_array::ffi::FFI_ArrowSchema;
use arrow_schema::Schema;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex_arrow::ToArrowType;

/// Export a Vortex [`DType`] to the Arrow C Data Interface struct at `schema_addr`. String and
/// binary columns are exported as their native view types (Utf8View/BinaryView); consumers are
/// expected to handle them.
pub(crate) fn export_dtype_to_arrow(dtype: &DType, schema_addr: i64) -> VortexResult<()> {
    let arrow_schema = dtype.to_arrow_schema()?;
    let ffi_schema = FFI_ArrowSchema::try_from(&arrow_schema)?;
    unsafe {
        ptr::write(schema_addr as *mut FFI_ArrowSchema, ffi_schema);
    }
    Ok(())
}

/// Decode an [`FFI_ArrowSchema`] pointed to by `schema_addr` into an Arrow [`Schema`].
pub(crate) fn import_arrow_schema(schema_addr: i64) -> VortexResult<Schema> {
    let ffi_schema = unsafe { &*(schema_addr as *const FFI_ArrowSchema) };
    Ok(Schema::try_from(ffi_schema)?)
}
