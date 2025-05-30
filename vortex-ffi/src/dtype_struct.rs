use std::ops::Deref;
use std::sync::Arc;

use vortex::dtype::{DType, StructDType};
use vortex::error::{VortexExpect, vortex_panic};

use crate::arc_wrapper;
use crate::dtype::vx_dtype;
use crate::string::vx_string;

arc_wrapper!(
    /// Represents a Vortex struct data type, without top-level nullability.
    StructDType,
    vx_struct_dtype
);

/// Return the number of fields in the struct dtype.
pub unsafe extern "C-unwind" fn vx_struct_dtype_nfields(dtype: *const vx_struct_dtype) -> usize {
    vx_struct_dtype::as_ref(dtype).nfields()
}

/// Return a borrowed reference to the name of the field at the given index.
pub unsafe extern "C-unwind" fn vx_struct_dtype_field_name(
    dtype: *const vx_struct_dtype,
    idx: usize,
) -> *const vx_string {
    let struct_dtype = vx_struct_dtype::as_ref(dtype);
    if idx >= struct_dtype.nfields() {
        vortex_panic!("Field index out of bounds");
    }
    vx_string::new_ref(&struct_dtype.names()[idx])
}

/// Returns an *owned* reference to the dtype of the field at the given index.
///
/// The return type is owned since struct dtypes can be lazily parsed from a binary format, in
/// which case it's not possible to return a borrowed reference to the field dtype.
// TODO(ngates): should StructDType cache owned fields internally?
// TODO(ngates): should this output a vx_error?
pub unsafe extern "C-unwind" fn vx_struct_dtype_field_dtype(
    dtype: *const vx_struct_dtype,
    idx: usize,
) -> *const vx_dtype {
    let struct_dtype = vx_struct_dtype::as_ref(dtype);
    vx_dtype::new(Arc::new(
        struct_dtype
            .field_by_index(idx)
            .vortex_expect("Failed to parse lazy field dtype"),
    ))
}

/// Builder for creating a [`vx_struct_dtype`].
#[allow(non_camel_case_types)]
pub struct vx_struct_dtype_builder {
    names: Vec<Arc<str>>,
    fields: Vec<DType>,
}

/// Create a new struct dtype builder.
pub unsafe extern "C-unwind" fn vx_struct_dtype_builder_new() -> *mut vx_struct_dtype_builder {
    Box::into_raw(Box::new(vx_struct_dtype_builder {
        names: Vec::new(),
        fields: Vec::new(),
    }))
}

/// Add a field to the struct dtype builder.
///
/// Takes ownership of both the `name` and `dtype` pointers.
/// Must either free or finalize the builder.
pub unsafe extern "C-unwind" fn vx_struct_dtype_builder_add_field(
    builder: *mut vx_struct_dtype_builder,
    name: *const vx_string,
    dtype: *const vx_dtype,
) {
    let builder = unsafe { builder.as_mut() }.vortex_expect("null pointer");
    builder.names.push(vx_string::into_arc(name));
    builder
        .fields
        .push(vx_dtype::into_arc(dtype).deref().clone());
}

/// Finalize the struct dtype builder, returning a new `vx_struct_dtype`.
///
/// Takes ownership of the `builder`.
pub unsafe extern "C-unwind" fn vx_struct_dtype_builder_finalize(
    builder: *mut vx_struct_dtype_builder,
) -> *const vx_struct_dtype {
    if builder.is_null() {
        vortex_panic!("null pointer");
    }
    let builder = unsafe { Box::from_raw(builder) };
    let struct_dtype = StructDType::new(builder.names.into(), builder.fields.into());
    vx_struct_dtype::new(Arc::new(struct_dtype))
}
