// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use std::ffi::c_char;
use std::ptr;
use std::sync::Arc;

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::array::validity::Validity;
use vortex::dtype::FieldName;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;

use crate::array::vx_array;
use crate::array::vx_validity;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::to_field_name;

pub(crate) struct StructBuilder {
    names: Vec<FieldName>,
    fields: Vec<ArrayRef>,
    validity: Validity,
}

crate::box_wrapper!(StructBuilder, vx_struct_column_builder);

/// Create a new column-wise struct array builder with given validity and a
/// capacity hint. validity can't be NULL.
/// Capacity hint is for the number of columns.
/// If you don't know capacity, pass 0.
/// if validity is NULL, returns NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_struct_column_builder_new(
    validity: *const vx_validity,
    capacity: usize,
) -> *mut vx_struct_column_builder {
    if validity.is_null() {
        return ptr::null_mut();
    }

    let names = Vec::with_capacity(capacity);
    let fields = Vec::with_capacity(capacity);
    let validity = unsafe { &*validity }.into();
    let builder = StructBuilder {
        names,
        fields,
        validity,
    };
    vx_struct_column_builder::new(builder)
}

/// Add a named field to a struct array builder.
/// All arguments must be non-NULL.
/// If field's length doesn't match lengths of previous fields, sets error.
/// If an error is returned, the builder is still valid, and caller must
/// deallocate it using vx_struct_column_builder_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_struct_column_builder_add_field(
    builder: *mut vx_struct_column_builder,
    name: *const c_char,
    field: *const vx_array,
    error: *mut *mut vx_error,
) {
    try_or_default(error, || {
        vortex_ensure!(!builder.is_null());
        vortex_ensure!(!name.is_null());
        vortex_ensure!(!field.is_null());
        let builder = vx_struct_column_builder::as_mut(builder);

        let name = unsafe { to_field_name(name)? };
        let field = (*vx_array::as_ref(field)).clone();

        if !builder.fields.is_empty() && field.len() != builder.fields[0].len() {
            vortex_bail!(
                "Field length mismatch: expected {}, got {}",
                builder.fields[0].len(),
                field.len()
            );
        }
        builder.names.push(name);
        builder.fields.push(field);
        Ok(())
    })
}

/// Finalize a struct array builder, returning a struct array.
/// Consumes the builder. Caller doesn't need to free the builder after calling
/// this function.
///
/// Example:
///
/// vx_error* error = NULL;
///
/// vx_validity validity = {};
/// validity.type = VX_VALIDITY_NON_NULLABLE;
///
/// const vx_array* field_array = vx_array_new_null(5);
/// const vx_struct_column_builder* builder =
///     vx_struct_column_builder_new(&validity, 1);
///
/// vx_struct_column_builder_add_field(builder, "age", field_array, &error);
///
/// vx_array* struct_array = vx_struct_column_builder_finalize(builder, &error);
///
/// vx_array_free(struct_array);
/// vx_array_free(field_array);
///
#[unsafe(no_mangle)]
pub extern "C-unwind" fn vx_struct_column_builder_finalize(
    builder: *mut vx_struct_column_builder,
    error: *mut *mut vx_error,
) -> *const vx_array {
    try_or_default(error, || {
        vortex_ensure!(!builder.is_null());
        let builder = *vx_struct_column_builder::into_box(builder);
        let rows = if builder.fields.is_empty() {
            0
        } else {
            builder.fields[0].len()
        };
        let array =
            StructArray::try_new(builder.names.into(), builder.fields, rows, builder.validity)?;
        Ok(vx_array::new(Arc::new(array.into_array())))
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::Arc;

    use vortex::array::IntoArray;
    use vortex::array::LEGACY_SESSION;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;

    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::array::vx_array_new_null;
    use crate::array::vx_array_new_primitive;
    use crate::array::vx_validity;
    use crate::array::vx_validity_type;
    use crate::error::vx_error_free;
    use crate::ptype::vx_ptype;
    use crate::struct_array::vx_struct_column_builder_add_field;
    use crate::struct_array::vx_struct_column_builder_finalize;
    use crate::struct_array::vx_struct_column_builder_free;
    use crate::struct_array::vx_struct_column_builder_new;

    #[test]
    fn test_empty() {
        let validity = vx_validity {
            r#type: vx_validity_type::VX_VALIDITY_NON_NULLABLE,
            array: ptr::null(),
        };
        unsafe {
            let builder = vx_struct_column_builder_new(&raw const validity, 0);
            assert!(!builder.is_null());
            vx_struct_column_builder_free(builder);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_many() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let names = ["age", "name"];
        let age_field = PrimitiveArray::new(buffer![30u8, 25u8, 35u8], Validity::NonNullable);
        let name_field = VarBinViewArray::from_iter_str(["Alice", "Bob", "Charlie"]);
        let struct_array = StructArray::try_new(
            names.into(),
            vec![age_field.into_array(), name_field.clone().into_array()],
            3,
            Validity::NonNullable,
        )
        .unwrap();

        let validity = vx_validity {
            r#type: vx_validity_type::VX_VALIDITY_NON_NULLABLE,
            array: ptr::null(),
        };

        unsafe {
            let mut error = ptr::null_mut();
            let builder = vx_struct_column_builder_new(&raw const validity, 2);
            assert!(!builder.is_null());

            let ffi_age = [30u8, 25u8, 35u8];
            let ffi_age_field = vx_array_new_primitive(
                vx_ptype::PTYPE_U8,
                &raw const ffi_age as *const c_void,
                3,
                &raw const validity,
                &raw mut error,
            );
            assert!(!ffi_age_field.is_null());
            assert!(error.is_null());

            vx_struct_column_builder_add_field(
                builder,
                c"age".as_ptr(),
                ffi_age_field,
                &raw mut error,
            );
            assert!(error.is_null());

            // Check field mismatch
            let ffi_null_field = vx_array_new_null(5);
            vx_struct_column_builder_add_field(
                builder,
                c"null".as_ptr(),
                ffi_null_field,
                &raw mut error,
            );
            assert!(!error.is_null());
            vx_error_free(error);
            vx_array_free(ffi_null_field);

            // Can't create a string array from C API yet.
            let ffi_name_field = vx_array::new(Arc::new(name_field.into_array()));
            vx_struct_column_builder_add_field(
                builder,
                c"name".as_ptr(),
                ffi_name_field,
                &raw mut error,
            );
            assert!(error.is_null());

            let array = vx_struct_column_builder_finalize(builder, &raw mut error);
            assert!(error.is_null());
            assert!(!array.is_null());

            {
                let array = vx_array::as_ref(array)
                    .clone()
                    .execute::<StructArray>(&mut ctx)
                    .unwrap();
                assert_arrays_eq!(array, struct_array);
            }

            vx_array_free(array);
            vx_array_free(ffi_name_field);
            vx_array_free(ffi_age_field);
        }
    }
}
