use std::ops::Deref;
use std::sync::Arc;

use vortex::dtype::{DType, StructFields};
use vortex::error::{VortexExpect, vortex_panic};

use crate::dtype::vx_dtype;
use crate::string::vx_string;
use crate::{arc_wrapper, box_wrapper};

arc_wrapper!(
    /// Represents a Vortex struct data type, without top-level nullability.
    StructFields,
    vx_struct_fields
);

/// Return the number of fields in the struct dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_nfields(dtype: *const vx_struct_fields) -> usize {
    vx_struct_fields::as_ref(dtype).nfields()
}

/// Return a borrowed reference to the name of the field at the given index.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_field_name(
    dtype: *const vx_struct_fields,
    idx: usize,
) -> *const vx_string {
    let struct_dtype = vx_struct_fields::as_ref(dtype);
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
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_struct_fields_field_dtype(
    dtype: *const vx_struct_fields,
    idx: usize,
) -> *const vx_dtype {
    let struct_dtype = vx_struct_fields::as_ref(dtype);
    vx_dtype::new(Arc::new(
        struct_dtype
            .field_by_index(idx)
            .vortex_expect("Failed to parse lazy field dtype"),
    ))
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
    vx_struct_fields_builder::new(Box::new(StructDTypeBuilder {
        names: Vec::new(),
        fields: Vec::new(),
    }))
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
) -> *const vx_struct_fields {
    let builder = vx_struct_fields_builder::into_box(builder);
    let struct_dtype = StructFields::new(builder.names.into(), builder.fields);
    vx_struct_fields::new(Arc::new(struct_dtype))
}
