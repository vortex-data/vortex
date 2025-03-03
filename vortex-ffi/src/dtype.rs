use std::ffi::{CStr, CString, c_char, c_int};
use std::sync::Arc;

use vortex::dtype::{DType, FieldNames, PType, StructDType};
use vortex::error::{VortexExpect, VortexUnwrap};

/// Pointer to a `DType` value that has been heap-allocated.
pub type DTypePtr = *mut DType;

/// Create a new simple dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_new(variant: u8, nullable: bool) -> DTypePtr {
    assert!(
        variant < DTYPE_STRUCT,
        "FFIDType_new: invalid variant: {variant}"
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
        DTYPE_STRUCT => unimplemented!("DTYPE_STRUCT is not supported in FFIDType_new"),
        DTYPE_LIST => unimplemented!("DTYPE_LIST is not supported in FFIDType_new"),
        DTYPE_EXTENSION => unimplemented!("DTYPE_EXTENSION is not supported in FFIDType_new"),
        _ => panic!("DType_new: invalid DType variant: {variant}"),
    };

    Box::into_raw(Box::new(dtype))
}

/// Create a new List type with the provided element type.
///
/// Upon successful return, this function moves the value out of the provided element pointer,
/// so it is not safe to reference afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_new_list(element: DTypePtr, nullable: bool) -> DTypePtr {
    assert!(!element.is_null(), "DType_new_list: null ptr");

    let element_type = *Box::from_raw(element);
    let element_dtype = Arc::new(element_type);
    let list_dtype = DType::List(element_dtype, nullable.into());

    Box::into_raw(Box::new(list_dtype))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_new_struct(
    names: *const *const c_char,
    dtypes: *const DTypePtr,
    len: u32,
    nullable: bool,
) -> DTypePtr {
    assert!(!names.is_null(), "DType_new_struct: names is null");
    assert!(!dtypes.is_null(), "DType_new_struct: dtypes is null");

    // Check that all of the field names/dtypes are non-null
    for i in 0..len {
        let name_ptr = *names.add(i as usize);
        assert!(!name_ptr.is_null(), "DType_new_struct: null name ptr");

        let dtype_ptr = *dtypes.add(i as usize);
        assert!(!dtype_ptr.is_null(), "DType_new_struct: null dtype ptr");
    }

    let mut rust_names = Vec::with_capacity(len as usize);
    let mut field_dtypes = Vec::with_capacity(len as usize);

    for i in 0..len {
        let name_ptr = *names.add(i as usize);
        let name: Arc<str> = CStr::from_ptr(name_ptr).to_str().unwrap().into();
        let dtype = (**dtypes.add(i as usize)).clone();

        rust_names.push(name);
        field_dtypes.push(dtype);
    }

    let field_names = FieldNames::from(rust_names);
    let struct_dtype = Arc::new(StructDType::new(field_names, field_dtypes));

    Box::into_raw(Box::new(DType::Struct(struct_dtype, nullable.into())))
}

// TODO(aduffy): create constructors for StructDType and ExtDType.

/// Free an [`FFIDType`] and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_free(dtype: DTypePtr) {
    assert!(!dtype.is_null(), "DType_free: null ptr");

    drop(Box::from_raw(dtype));
}

/// Get the dtype variant tag for an [`FFIDType`].
///
/// # Example
///
/// ```rust
/// use vortex_jni::dtype::{FFIDType, DType, DTYPE_BOOL};
///
/// let dtype = DType::Bool(true);
/// let ffi_dtype = FFIDType::from(&dtype);
///
/// assert_eq!(FFIDType_get(&ffi_dtype), DTYPE_BOOL);
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_get(dtype: *const DType) -> u8 {
    assert!(!dtype.is_null(), "DType_get: null ptr");

    let dtype = &*dtype;

    match dtype {
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
pub unsafe extern "C" fn DType_nullable(dtype: *const DType) -> bool {
    assert!(!dtype.is_null(), "DType_nullable: null ptr");
    let dtype = &*dtype;

    dtype.is_nullable()
}

/// For `DTYPE_STRUCT` variant DTypes, get the number of fields.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_field_count(dtype: *const DType) -> u32 {
    assert!(!dtype.is_null(), "DType_field_count: null ptr");

    let DType::Struct(struct_dtype, _) = &*dtype else {
        panic!("FFIDType_field_count: not a struct dtype")
    };

    struct_dtype
        .nfields()
        .try_into()
        .vortex_expect("field count must fit in u32")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_field_name(
    dtype: *const DType,
    index: u32,
    dst: *mut c_char,
    len: *mut c_int,
) {
    assert!(!dtype.is_null(), "DType_field_name: null dtype ptr");

    let DType::Struct(struct_dtype, _) = &*dtype else {
        panic!("FFIDType_field_name: not a struct dtype")
    };

    let field_name = struct_dtype.names()[index as usize].as_ref();

    let cstr = CString::new(field_name).expect("CString");

    // write the cstr + NUL byte to dst
    std::ptr::copy(cstr.as_ptr(), dst, cstr.as_bytes().len());

    *len = cstr.as_bytes().len().try_into().vortex_unwrap();
}

/// Get the dtype of a field in a `DTYPE_STRUCT` variant DType.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_field_dtype(dtype: *const DType, index: u32) -> DTypePtr {
    let DType::Struct(struct_dtype, _) = &*dtype else {
        panic!("FFIDType_field_dtype: not a struct dtype")
    };

    let field = struct_dtype
        .field_by_index(index as usize)
        .vortex_expect("field index out of bounds");

    Box::into_raw(Box::new(field))
}

/// For a list DType, get the inner element type.
///
/// The pointee's lifetime is tied to the lifetime of the list DType. It should not be
/// accessed after the list DType has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DType_element_type(dtype: *const DType) -> *const DType {
    assert!(!dtype.is_null(), "DType_element_type: null ptr");

    let DType::List(element_dtype, _) = &*dtype else {
        panic!("FFIDType_element_type: not a list dtype")
    };

    element_dtype.as_ref()
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
    use std::ffi::{c_char, c_int};

    use vortex::dtype::DType;

    use crate::dtype::{
        DTYPE_BOOL, DTYPE_PRIMITIVE_U8, DTYPE_STRUCT, DTYPE_UTF8, DType_field_count,
        DType_field_dtype, DType_field_name, DType_free, DType_get, DType_new, DType_new_struct,
    };

    #[test]
    fn test_simple() {
        unsafe {
            let ffi_dtype = DType_new(DTYPE_BOOL, true);

            // functions check
            assert_eq!(DType_get(ffi_dtype), DTYPE_BOOL);

            // Field access checks.
            let dtype = &*ffi_dtype;
            assert_eq!(dtype, &DType::Bool(true.into()));

            // Free the memory.
            DType_free(ffi_dtype);
        }
    }

    #[test]
    fn test_struct() {
        unsafe {
            let name = DType_new(DTYPE_UTF8, false);
            let age = DType_new(DTYPE_PRIMITIVE_U8, true);

            let names = [c"name".as_ptr(), c"age".as_ptr()];

            let dtypes = [name, age];

            let person = DType_new_struct(names.as_ptr(), dtypes.as_ptr(), 2, false);

            assert_eq!(DType_get(person), DTYPE_STRUCT);
            assert_eq!(DType_field_count(person), 2);

            let mut name_bytes = vec![0u8; 64];
            let mut name_len: c_int = 0;
            DType_field_name(
                person,
                0,
                name_bytes.as_mut_ptr() as *mut c_char,
                &mut name_len,
            );
            // Check name_bytes
            let field_name = std::str::from_utf8_unchecked(&name_bytes[..name_len as usize]);
            assert_eq!(field_name, "name");

            DType_field_name(
                person,
                1,
                name_bytes.as_mut_ptr() as *mut c_char,
                &mut name_len,
            );
            // Check name_bytes
            let field_name = std::str::from_utf8_unchecked(&name_bytes[..name_len as usize]);
            assert_eq!(field_name, "age");

            let dtype0 = DType_field_dtype(person, 0);
            let dtype1 = DType_field_dtype(person, 1);
            assert_eq!(DType_get(dtype0), DTYPE_UTF8);
            assert_eq!(DType_get(dtype1), DTYPE_PRIMITIVE_U8);

            DType_free(person);
        }
    }
}
