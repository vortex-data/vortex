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
use crate::string::vx_string;

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

/// Return a borrowed reference to the name of the field at the given index.
///
/// The returned pointer is valid as long as the struct fields is valid.
/// Do NOT free the returned string pointer - it shares the lifetime of the struct fields.
/// Returns null if the index is out of bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_field_name(
    dtype: *const vx_struct_fields,
    idx: usize,
) -> *const vx_string {
    // TODO(joe): propagate this error up instead of expecting
    let ptr = unsafe { dtype.as_ref() }.vortex_expect("null ptr");
    let struct_dtype = &ptr.0;
    if idx >= struct_dtype.nfields() {
        return ptr::null();
    }
    let name = struct_dtype.names()[idx].inner();
    vx_string::new_ref(name)
}

/// Returns an *owned* reference to the dtype of the field at the given index.
///
/// The return type is owned since struct dtypes can be lazily parsed from a binary format, in
/// which case it's not possible to return a borrowed reference to the field dtype.
///
/// Returns null if the index is out of bounds or if the field dtype cannot be parsed.
// TODO(ngates): should StructDType cache owned fields internally?
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
/// Takes ownership of both the `name` and `dtype` pointers.
/// Must either free or finalize the builder.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_builder_add_field(
    builder: *mut vx_struct_fields_builder,
    name: *const vx_string,
    dtype: *const vx_dtype,
) {
    let builder = vx_struct_fields_builder::as_mut(builder);
    builder.names.push(vx_string::into_arc(name));
    builder
        .fields
        .push(vx_dtype::into_arc(dtype).deref().clone());
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
