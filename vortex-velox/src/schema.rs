// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use arrow_array::ffi::FFI_ArrowSchema;
use vortex_arrow::ArrowSessionExt;
use vortex_error::vortex_err;
use vortex_ffi::try_or;
use vortex_ffi::vx_error;

use crate::source::vx_velox_source;
use crate::temporal::validate_velox_arrow_type;

/// Export an opened source schema through the Arrow C Data Interface.
///
/// The caller owns the output and must invoke its release callback.
///
/// # Safety
///
/// `source` must point to a live source. `schema_out` must identify uninitialized writable
/// storage. `error_out` must be null or valid for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_source_export_schema(
    source: *const vx_velox_source,
    schema_out: *mut FFI_ArrowSchema,
    error_out: *mut *mut vx_error,
) -> i32 {
    try_or(error_out, 1, || {
        let source = unsafe {
            source
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox source must not be null"))?
        };
        if schema_out.is_null() {
            return Err(vortex_err!("Arrow schema output must not be null"));
        }
        let arrow_schema = source
            .file()
            .session()
            .arrow()
            .to_arrow_schema(source.file().dtype())?;
        for field in arrow_schema.fields() {
            validate_velox_arrow_type(field.data_type())?;
        }
        let schema = FFI_ArrowSchema::try_from(&arrow_schema)?;
        unsafe { ptr::write(schema_out, schema) };
        Ok(0)
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::ffi::FFI_ArrowSchema;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn rejects_null_source() -> VortexResult<()> {
        let mut schema = FFI_ArrowSchema::empty();
        let mut error = ptr::null_mut();
        let status =
            unsafe { vx_velox_source_export_schema(ptr::null(), &raw mut schema, &raw mut error) };
        assert_eq!(status, 1);
        assert!(!error.is_null());
        unsafe { vortex_ffi::vx_error_free(error) };
        Ok(())
    }
}
