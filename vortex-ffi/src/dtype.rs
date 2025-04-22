use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;

use vortex::dtype::datetime::{DATE_ID, TIME_ID, TIMESTAMP_ID, TemporalMetadata};
use vortex::dtype::{DType, FieldNames, PType, StructDType};
use vortex::error::{VortexExpect, VortexUnwrap, vortex_bail};

use crate::error::{try_or, vx_error};

/// Pointer to a `DType` value that has been heap-allocated.
/// Create a new simple dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new(variant: u8, nullable: bool) -> *mut DType {
    assert!(
        variant < DTYPE_STRUCT,
        "DType_new: invalid variant: {variant}"
    );

    let dtype = match variant {
        DTYPE_NULL => DType::Null,
        DTYPE_BOOL => DType::Bool(nullable.into()),
        DTYPE_PRIMITIVE_U8 => DType::Primitive(PType::U8, nullable.into()),
        DTYPE_PRIMITIVE_U16 => DType::Primitive(PType::U16, nullable.into()),
        DTYPE_PRIMITIVE_U32 => DType::Primitive(PType::U32, nullable.into()),
        DTYPE_PRIMITIVE_U64 => DType::Primitive(PType::U64, nullable.into()),
        DTYPE_PRIMITIVE_I8 => DType::Primitive(PType::I8, nullable.into()),
        DTYPE_PRIMITIVE_I16 => DType::Primitive(PType::I16, nullable.into()),
        DTYPE_PRIMITIVE_I32 => DType::Primitive(PType::I32, nullable.into()),
        DTYPE_PRIMITIVE_I64 => DType::Primitive(PType::I64, nullable.into()),
        DTYPE_PRIMITIVE_F16 => DType::Primitive(PType::F16, nullable.into()),
        DTYPE_PRIMITIVE_F32 => DType::Primitive(PType::F32, nullable.into()),
        DTYPE_PRIMITIVE_F64 => DType::Primitive(PType::F64, nullable.into()),
        DTYPE_UTF8 => DType::Utf8(nullable.into()),
        DTYPE_BINARY => DType::Binary(nullable.into()),
        DTYPE_STRUCT => unimplemented!("DTYPE_STRUCT is not supported in DType_new"),
        DTYPE_LIST => unimplemented!("DTYPE_LIST is not supported in DType_new"),
        DTYPE_EXTENSION => unimplemented!("DTYPE_EXTENSION is not supported in DType_new"),
        _ => panic!("DType_new: invalid DType variant: {variant}"),
    };

    Box::into_raw(Box::new(dtype))
}

/// Create a new List type with the provided element type.
///
/// Upon successful return, this function moves the value out of the provided element pointer,
/// so it is not safe to reference afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_list(
    element: *mut DType,
    nullable: bool,
) -> *mut DType {
    assert!(!element.is_null(), "DType_new_list: null ptr");

    let element_type = *Box::from_raw(element);
    let element_dtype = Arc::new(element_type);
    let list_dtype = DType::List(element_dtype, nullable.into());

    Box::into_raw(Box::new(list_dtype))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_new_struct(
    names: *const *const c_char,
    dtypes: *const *mut DType,
    len: u32,
    nullable: bool,
) -> *mut DType {
    // Check that all of the field names/dtypes are non-null
    let mut rust_names = Vec::with_capacity(len as usize);
    let mut field_dtypes = Vec::with_capacity(len as usize);

    for i in 0..len {
        let name_ptr = *names.add(i as usize);
        let name: Arc<str> = CStr::from_ptr(name_ptr).to_string_lossy().into();
        let dtype = Box::from_raw(*dtypes.add(i as usize));
        let dtype = *dtype;

        rust_names.push(name);
        field_dtypes.push(dtype);
    }

    let field_names = FieldNames::from(rust_names);
    let struct_dtype = Arc::new(StructDType::new(field_names, field_dtypes));

    Box::into_raw(Box::new(DType::Struct(struct_dtype, nullable.into())))
}

/// Free an [`DType`] and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_free(dtype: *mut DType) {
    drop(Box::from_raw(dtype));
}

/// Get the dtype variant tag for an [`DType`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_get(dtype: *const DType) -> u8 {
    match dtype.as_ref().vortex_expect("null dtype") {
        DType::Null => DTYPE_NULL,
        DType::Bool(_) => DTYPE_BOOL,
        DType::Primitive(ptype, _) => match ptype {
            PType::U8 => DTYPE_PRIMITIVE_U8,
            PType::U16 => DTYPE_PRIMITIVE_U16,
            PType::U32 => DTYPE_PRIMITIVE_U32,
            PType::U64 => DTYPE_PRIMITIVE_U64,
            PType::I8 => DTYPE_PRIMITIVE_I8,
            PType::I16 => DTYPE_PRIMITIVE_I16,
            PType::I32 => DTYPE_PRIMITIVE_I32,
            PType::I64 => DTYPE_PRIMITIVE_I64,
            PType::F16 => DTYPE_PRIMITIVE_F16,
            PType::F32 => DTYPE_PRIMITIVE_F32,
            PType::F64 => DTYPE_PRIMITIVE_F64,
        },
        DType::Utf8(_) => DTYPE_UTF8,
        DType::Binary(_) => DTYPE_BINARY,
        DType::Struct(..) => DTYPE_STRUCT,
        DType::List(..) => DTYPE_LIST,
        DType::Extension(_) => DTYPE_EXTENSION,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_nullable(dtype: *const DType) -> bool {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");
    dtype.is_nullable()
}

/// For `DTYPE_STRUCT` variant DTypes, get the number of fields.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_field_count(dtype: *const DType) -> u32 {
    let DType::Struct(struct_dtype, _) = unsafe { dtype.as_ref() }.vortex_expect("dtype null")
    else {
        panic!("vx_dtype_field_count: not a struct dtype")
    };

    struct_dtype
        .nfields()
        .try_into()
        .vortex_expect("field count must fit in u32")
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_field_name(
    dtype: *const DType,
    index: u32,
    dst: *mut c_void,
    len: *mut c_int,
) {
    assert!(!dst.is_null(), "DType_field_name: null ptr dst");
    assert!(!len.is_null(), "DType_field_name: null ptr len");

    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");
    let DType::Struct(struct_dtype, _) = dtype else {
        panic!("vx_dtype_field_name: not a struct dtype")
    };

    let field_name = struct_dtype.names()[index as usize].as_ref();
    let bytes = field_name.as_bytes();

    let dst_slice = std::slice::from_raw_parts_mut(dst as *mut u8, bytes.len());
    dst_slice.copy_from_slice(bytes);

    *len = bytes.len().try_into().vortex_unwrap();
}

/// Get the dtype of a field in a `DTYPE_STRUCT` variant DType.
///
/// This returns a new owned, allocated copy of the DType that must be freed subsequently
/// by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_field_dtype(
    dtype: *const DType,
    index: u32,
) -> *mut DType {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");
    let DType::Struct(struct_dtype, _) = dtype else {
        panic!("DType_field_dtype: not a struct dtype")
    };

    let field = struct_dtype
        .field_by_index(index as usize)
        .vortex_expect("field index out of bounds");

    // TODO(aduffy): can we represent this via a SharedReference instead? It seems like we
    //  want only one owned copy of the field array. We can then make a new copy from the
    //  existing copy if we want to hold onto it.
    Box::into_raw(Box::new(field))
}

/// For a list DType, get the inner element type.
///
/// The pointee's lifetime is tied to the lifetime of the list DType. It should not be
/// accessed after the list DType has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_element_type(
    dtype: *const DType,
    error: *mut *mut vx_error,
) -> *const DType {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    try_or(error, ptr::null(), || {
        let DType::List(element_dtype, _) = dtype else {
            vortex_bail!("vx_dtype_element_type: not a list dtype")
        };
        Ok(element_dtype.as_ref())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_time(dtype: *const DType) -> bool {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    match dtype {
        DType::Extension(ext_dtype) => ext_dtype.id() == &*TIME_ID,
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dype_is_date(dtype: *const DType) -> bool {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    match dtype {
        DType::Extension(ext_dtype) => ext_dtype.id() == &*DATE_ID,
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_is_timestamp(dtype: *const DType) -> bool {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    match dtype {
        DType::Extension(ext_dtype) => ext_dtype.id() == &*TIMESTAMP_ID,
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_time_unit(dtype: *const DType) -> u8 {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    let DType::Extension(ext_dtype) = dtype else {
        panic!("DType_time_unit: not a time dtype")
    };

    let metadata = ext_dtype.metadata().vortex_expect("time unit metadata");
    let time_unit = metadata.as_ref()[0];

    time_unit
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_dtype_time_zone(
    dtype: *const DType,
    dst: *mut c_void,
    len: *mut c_int,
) {
    let dtype = unsafe { dtype.as_ref() }.vortex_expect("dtype null");

    let DType::Extension(ext_dtype) = dtype else {
        panic!("vx_dtype_time_unit: not a time dtype")
    };

    match TemporalMetadata::try_from(ext_dtype).vortex_expect("timestamp") {
        TemporalMetadata::Timestamp(_, zone) => {
            if let Some(zone) = zone {
                let bytes = zone.as_bytes();
                let dst = std::slice::from_raw_parts_mut(dst as *mut u8, bytes.len());
                dst.copy_from_slice(bytes);
                *len = bytes.len().try_into().vortex_unwrap();
            } else {
                // No time zone, using local timestamps.
                *len = 0;
            }
        }
        _ => panic!("DType_time_zone: not a timestamp metadata: {:?}", ext_dtype),
    }
}

pub const DTYPE_NULL: u8 = 0;
pub const DTYPE_BOOL: u8 = 1;
pub const DTYPE_PRIMITIVE_U8: u8 = 2;
pub const DTYPE_PRIMITIVE_U16: u8 = 3;
pub const DTYPE_PRIMITIVE_U32: u8 = 4;
pub const DTYPE_PRIMITIVE_U64: u8 = 5;
pub const DTYPE_PRIMITIVE_I8: u8 = 6;
pub const DTYPE_PRIMITIVE_I16: u8 = 7;
pub const DTYPE_PRIMITIVE_I32: u8 = 8;
pub const DTYPE_PRIMITIVE_I64: u8 = 9;
pub const DTYPE_PRIMITIVE_F16: u8 = 10;
pub const DTYPE_PRIMITIVE_F32: u8 = 11;
pub const DTYPE_PRIMITIVE_F64: u8 = 12;
pub const DTYPE_UTF8: u8 = 13;
pub const DTYPE_BINARY: u8 = 14;
pub const DTYPE_STRUCT: u8 = 15;
pub const DTYPE_LIST: u8 = 16;
pub const DTYPE_EXTENSION: u8 = 17;

#[cfg(test)]
mod tests {
    use std::ffi::{c_int, c_void};

    use vortex::dtype::DType;

    use crate::dtype::{
        DTYPE_BOOL, DTYPE_PRIMITIVE_U8, DTYPE_STRUCT, DTYPE_UTF8, vx_dtype_field_count,
        vx_dtype_field_dtype, vx_dtype_field_name, vx_dtype_free, vx_dtype_get, vx_dtype_new,
        vx_dtype_new_struct,
    };

    #[test]
    fn test_simple() {
        unsafe {
            let ffi_dtype = vx_dtype_new(DTYPE_BOOL, true);

            // functions check
            assert_eq!(vx_dtype_get(ffi_dtype), DTYPE_BOOL);

            // Field access checks.
            let dtype = &*ffi_dtype;
            assert_eq!(dtype, &DType::Bool(true.into()));

            // Free the memory.
            vx_dtype_free(ffi_dtype);
        }
    }

    #[test]
    fn test_struct() {
        unsafe {
            let name = vx_dtype_new(DTYPE_UTF8, false);
            let age = vx_dtype_new(DTYPE_PRIMITIVE_U8, true);

            let names = [c"name".as_ptr(), c"age".as_ptr()];

            let dtypes = [name, age];

            let person = vx_dtype_new_struct(names.as_ptr(), dtypes.as_ptr(), 2, false);

            assert_eq!(vx_dtype_get(person), DTYPE_STRUCT);
            assert_eq!(vx_dtype_field_count(person), 2);

            let mut name_bytes = vec![0u8; 64];
            let mut name_len: c_int = 0;
            vx_dtype_field_name(
                person,
                0,
                name_bytes.as_mut_ptr() as *mut c_void,
                &mut name_len,
            );
            // Check name_bytes
            let field_name = std::str::from_utf8_unchecked(&name_bytes[..name_len as usize]);
            assert_eq!(field_name, "name");

            vx_dtype_field_name(
                person,
                1,
                name_bytes.as_mut_ptr() as *mut c_void,
                &mut name_len,
            );
            // Check name_bytes
            let field_name = std::str::from_utf8_unchecked(&name_bytes[..name_len as usize]);
            assert_eq!(field_name, "age");

            let dtype0 = vx_dtype_field_dtype(person, 0);
            let dtype1 = vx_dtype_field_dtype(person, 1);
            assert_eq!(vx_dtype_get(dtype0), DTYPE_UTF8);
            assert_eq!(vx_dtype_get(dtype1), DTYPE_PRIMITIVE_U8);

            vx_dtype_free(dtype0);
            vx_dtype_free(dtype1);

            vx_dtype_free(person);
        }
    }
}
