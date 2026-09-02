// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;
use std::slice;

use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::layout::layouts::row_idx::row_idx;
use vortex_ffi::try_or;
use vortex_ffi::vx_error;
use vortex_ffi::vx_expression;
use vortex_ffi::vx_expression_new_with;
use vortex_ffi::vx_view;

unsafe fn projection(
    names: *const vx_view,
    len: usize,
    row_index_name: vx_view,
) -> VortexResult<*mut vx_expression> {
    vortex_ensure!(!row_index_name.ptr.is_null() || row_index_name.len == 0);
    // SAFETY: The caller keeps this view valid for the duration of this call.
    let row_index_name = unsafe { row_index_name.as_str() }?;
    vortex_ensure!(
        !row_index_name.is_empty(),
        "row index field name must not be empty"
    );

    let names = if names.is_null() {
        vortex_ensure!(len == 0, "null field names pointer with non-zero length");
        &[]
    } else {
        // SAFETY: The caller provides `len` initialized views when the pointer is non-null.
        unsafe { slice::from_raw_parts(names, len) }
    };

    let mut fields = Vec::with_capacity(len + 1);
    fields.push((FieldName::from(row_index_name), row_idx()));
    for name in names {
        // SAFETY: Each caller-provided view remains valid for this call.
        let name = unsafe { name.as_str() }?;
        vortex_ensure!(
            name != row_index_name,
            "row index field name conflicts with projected field: {name}"
        );
        fields.push((FieldName::from(name), get_item(name, root())));
    }
    Ok(vx_expression_new_with(pack(
        fields,
        Nullability::NonNullable,
    )))
}

/// Create a struct projection with an absolute file-row index as its first field.
///
/// The remaining fields select the supplied names from the scan root. The
/// returned expression stays owned by the caller.
///
/// # Safety
///
/// `names` must be null when `len` is zero or point to `len` valid views.
/// Every view and `row_index_name` must remain valid for this call.
/// `error_out` must be null or point to writable storage. No input operation can unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_expression_select_with_row_index(
    names: *const vx_view,
    len: usize,
    row_index_name: vx_view,
    error_out: *mut *mut vx_error,
) -> *mut vx_expression {
    try_or(error_out, ptr::null_mut(), || unsafe {
        projection(names, len, row_index_name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::vx_velox_expression_free;

    fn view(value: &str) -> vx_view {
        vx_view {
            ptr: value.as_ptr().cast(),
            len: value.len(),
        }
    }

    #[test]
    fn creates_row_index_projection() {
        let names = [view("a"), view("b")];
        let mut error = ptr::null_mut();
        let expression = unsafe {
            vx_velox_expression_select_with_row_index(
                names.as_ptr(),
                names.len(),
                view("$row_index"),
                &raw mut error,
            )
        };
        assert!(error.is_null());
        assert!(!expression.is_null());
        unsafe { vx_velox_expression_free(expression) };
    }

    #[test]
    fn rejects_name_collision() {
        let names = [view("$row_index")];
        let mut error = ptr::null_mut();
        let expression = unsafe {
            vx_velox_expression_select_with_row_index(
                names.as_ptr(),
                names.len(),
                view("$row_index"),
                &raw mut error,
            )
        };
        assert!(expression.is_null());
        assert!(!error.is_null());
        unsafe { vortex_ffi::vx_error_free(error) };
    }
}
