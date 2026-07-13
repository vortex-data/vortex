// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use std::ptr;
use std::sync::Arc;

use vortex::dtype::DType;
use vortex::dtype::StructFields;
use vortex::error::VortexExpect;

use crate::box_wrapper;
use crate::dtype::vx_dtype;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::string::vx_view;

box_wrapper!(
    /// Represents a Vortex struct data type, without top-level nullability.
    StructFields,
    vx_struct_fields
);

/// Return the number of fields in the struct dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_nfields(dtype: *const vx_struct_fields) -> u64 {
    // TODO(joe): propagate this error up instead of expecting
    unsafe { dtype.as_ref() }
        .vortex_expect("null ptr")
        .0
        .nfields() as u64
}

/// Return field name at a given index.
/// If index is out of bounds, returns {NULL, 0}.
///
/// Returned view is valid as long as "dtype" is valid.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_field_name(
    dtype: *const vx_struct_fields,
    idx: usize,
) -> vx_view {
    // TODO(joe): propagate this error up instead of expecting
    let ptr = unsafe { dtype.as_ref() }.vortex_expect("null ptr");
    let struct_dtype = &ptr.0;
    if idx >= struct_dtype.nfields() {
        return vx_view::null();
    }
    vx_view::from_str(struct_dtype.names()[idx].inner())
}

/// Return an owned dtype of the field at a given index.
/// Returns NULL if index is out of bounds or if dtype cannot be parsed.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_field_dtype(
    dtype: *const vx_struct_fields,
    idx: usize,
) -> *const vx_dtype {
    // TODO(joe): propagate this error up instead of expecting
    let ptr = unsafe { dtype.as_ref() }.vortex_expect("null ptr");
    let struct_dtype = &ptr.0;

    if idx >= struct_dtype.nfields() {
        return ptr::null();
    }

    match struct_dtype.field_by_index(idx) {
        Some(field_dtype) => vx_dtype::new(Arc::new(field_dtype)),
        None => ptr::null(),
    }
}

pub(crate) struct StructDTypeBuilder {
    names: Vec<Arc<str>>,
    fields: Vec<DType>,
}

box_wrapper!(
    /// Builder for creating a [`vx_struct_fields`].
    StructDTypeBuilder,
    vx_struct_fields_builder
);

/// Create a new struct dtype builder.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_builder_new() -> *mut vx_struct_fields_builder {
    vx_struct_fields_builder::new(StructDTypeBuilder {
        names: Vec::new(),
        fields: Vec::new(),
    })
}

/// Add a field to the struct dtype builder.
///
/// "name" is copied. Takes ownership of "dtype".
/// Caller must free or finalize the builder.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_builder_add_field(
    builder: *mut vx_struct_fields_builder,
    name: vx_view,
    dtype: *const vx_dtype,
    error_out: *mut *mut vx_error,
) {
    try_or_default(error_out, || {
        let builder = vx_struct_fields_builder::as_mut(builder);
        let field = vx_dtype::into_arc(dtype).deref().clone();
        let name = Arc::from(unsafe { name.as_str() }?);
        builder.fields.push(field);
        builder.names.push(name);
        Ok(())
    })
}

/// Finalize the struct dtype builder, returning a new `vx_struct_fields`.
///
/// Takes ownership of the `builder`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_builder_finalize(
    builder: *mut vx_struct_fields_builder,
) -> *mut vx_struct_fields {
    let StructDTypeBuilder { names, fields } = *vx_struct_fields_builder::into_box(builder);
    let struct_dtype = StructFields::new(names.into(), fields);
    vx_struct_fields::new(struct_dtype)
}
