// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use std::ptr;
use std::sync::Arc;

use vortex::dtype::{DType, StructFields};
use vortex::error::VortexExpect;

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
pub unsafe extern "C-unwind" fn vx_struct_fields_nfields(dtype: *const vx_struct_fields) -> u64 {
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
    let ptr = unsafe { dtype.as_ref() }.vortex_expect("null ptr");
    let struct_dtype = &ptr.0;
    if idx >= struct_dtype.nfields() {
        return ptr::null();
    }
    vx_string::new(struct_dtype.names()[idx].clone())
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
    idx: u64,
) -> *const vx_dtype {
    let ptr = unsafe { dtype.as_ref() }.vortex_expect("null ptr");
    let struct_dtype = &ptr.0;

    let idx_usize = match usize::try_from(idx) {
        Ok(i) => i,
        Err(_) => return ptr::null(),
    };

    if idx_usize >= struct_dtype.nfields() {
        return ptr::null();
    }

    match struct_dtype.field_by_index(idx_usize) {
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

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::dtype::*;
    use crate::ptype::vx_ptype;
    use crate::string::*;

    #[test]
    fn test_struct_fields_error_handling() {
        unsafe {
            // Create a simple struct fields using the FFI functions
            let builder = vx_struct_fields_builder_new();
            let name_field = vx_string_new_from_cstr(c"name".as_ptr());
            let age_field = vx_string_new_from_cstr(c"age".as_ptr());
            let utf8_dtype = vx_dtype_new_utf8(false);
            let i32_dtype = vx_dtype_new_primitive(vx_ptype::PTYPE_I32, false);

            vx_struct_fields_builder_add_field(builder, name_field, utf8_dtype);
            vx_struct_fields_builder_add_field(builder, age_field, i32_dtype);

            let fields = vx_struct_fields_builder_finalize(builder);

            // Test valid field access
            assert_eq!(vx_struct_fields_nfields(fields), 2);

            let field_name_0 = vx_struct_fields_field_name(fields, 0);
            assert!(!field_name_0.is_null());
            let field_dtype_0 = vx_struct_fields_field_dtype(fields, 0);
            assert!(!field_dtype_0.is_null());

            // Test out-of-bounds field access
            let invalid_name = vx_struct_fields_field_name(fields, 999);
            assert!(invalid_name.is_null());

            let invalid_dtype = vx_struct_fields_field_dtype(fields, 999);
            assert!(invalid_dtype.is_null());

            // Clean up
            vx_string_free(field_name_0);
            vx_dtype_free(field_dtype_0);
            vx_struct_fields_free(fields);
        }
    }

    #[test]
    fn test_struct_fields_memory_management() {
        unsafe {
            // Test memory management patterns for field access
            let builder = vx_struct_fields_builder_new();

            // Create multiple fields to test memory handling
            for i in 0..5 {
                let field_name = format!("field_{}\0", i);
                let name_str = vx_string_new_from_cstr(field_name.as_ptr() as *const i8);
                let dtype = vx_dtype_new_primitive(vx_ptype::PTYPE_I32, false);
                vx_struct_fields_builder_add_field(builder, name_str, dtype);
            }

            let fields = vx_struct_fields_builder_finalize(builder);
            let n_fields = vx_struct_fields_nfields(fields);
            assert_eq!(n_fields, 5);

            // Access all fields and ensure proper cleanup
            for i in 0..n_fields {
                let field_name = vx_struct_fields_field_name(fields, i as usize);
                let field_dtype = vx_struct_fields_field_dtype(fields, i);

                assert!(!field_name.is_null());
                assert!(!field_dtype.is_null());

                // Verify we can access string properties
                let name_len = vx_string_len(field_name);
                assert!(name_len > 0);

                // Verify dtype properties
                let variant = vx_dtype_get_variant(field_dtype);
                assert_eq!(variant, vx_dtype_variant::DTYPE_PRIMITIVE);

                // Clean up owned references
                vx_string_free(field_name);
                vx_dtype_free(field_dtype);
            }

            vx_struct_fields_free(fields);
        }
    }

    #[test]
    fn test_struct_builder_error_conditions() {
        unsafe {
            // Test builder behavior with edge cases
            let builder = vx_struct_fields_builder_new();

            // Finalize empty builder (should work)
            let empty_fields = vx_struct_fields_builder_finalize(builder);
            assert_eq!(vx_struct_fields_nfields(empty_fields), 0);
            vx_struct_fields_free(empty_fields);
        }
    }

    #[test]
    fn test_field_name_string_safety() {
        unsafe {
            // Test string handling in field names
            let builder = vx_struct_fields_builder_new();

            // Test with various string types including empty and special chars
            let test_names = [
                c"normal_field",
                c"", // Empty string (just null terminator)
                c"field_with_underscore_123",
                c"field-with-dashes",
            ];

            for (i, name) in test_names.iter().enumerate() {
                let name_str = vx_string_new_from_cstr(name.as_ptr());
                let dtype = vx_dtype_new_primitive(
                    if i % 2 == 0 {
                        vx_ptype::PTYPE_I32
                    } else {
                        vx_ptype::PTYPE_U64
                    },
                    false,
                );
                vx_struct_fields_builder_add_field(builder, name_str, dtype);
            }

            let fields = vx_struct_fields_builder_finalize(builder);

            // Verify all fields were added correctly
            assert_eq!(vx_struct_fields_nfields(fields) as usize, test_names.len());

            // Test string access and verify content
            for i in 0..test_names.len() {
                let field_name = vx_struct_fields_field_name(fields, i);
                assert!(!field_name.is_null());

                let name_len = vx_string_len(field_name);
                let expected_len = test_names[i].count_bytes();
                assert_eq!(name_len, expected_len);

                vx_string_free(field_name);
            }

            vx_struct_fields_free(fields);
        }
    }
}
