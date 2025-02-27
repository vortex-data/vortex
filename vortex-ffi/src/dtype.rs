use std::ffi::{CStr, CString, c_char};
use std::mem::ManuallyDrop;
use std::str::FromStr;

use vortex::dtype::{DType, ExtDType, PType, StructDType};
use vortex::error::VortexExpect;

/// A version of [`DType`] that can be sent/received over the FFI interface.
#[repr(C)]
pub struct FFIDType {
    pub dtype: u8,
    pub nullable: bool,
    pub type_info: Option<FFIDTypeInfo>,
}

#[repr(C)]
pub union FFIDTypeInfo {
    pub struct_dtype: ManuallyDrop<Box<FFIStructDType>>,
    pub list_dtype: ManuallyDrop<Box<FFIDType>>,
    pub extension_dtype: ManuallyDrop<Box<FFIExtensionDType>>,
}

/// Native FFI interface for Rust [`StructDType`].
#[repr(C)]
pub struct FFIStructDType {
    pub names: Vec<CString>,
    pub types: Vec<Box<FFIDType>>,
}

/// FFI interface for the element type of a `List` DType.
#[repr(C)]
pub struct FFIListDType {
    pub element: Box<FFIDType>,
}

/// FFI interface for [`ExtDType`].
#[repr(C)]
pub struct FFIExtensionDType {
    pub id: Vec<u8>,
    pub storage_dtype: Box<FFIDType>,
    pub metadata: Option<Vec<u8>>,
}

/// Create a new simple dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIDType_new(variant: u8, nullable: bool) -> *mut FFIDType {
    assert!(
        variant < DTYPE_STRUCT,
        "FFIDType_new: invalid variant: {variant}"
    );

    let ffi_dtype = Box::new(FFIDType {
        dtype: variant,
        nullable,
        type_info: None,
    });

    Box::into_raw(ffi_dtype)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIDType_new_list(
    element: *mut FFIDType,
    nullable: bool,
) -> *mut FFIDType {
    // Assume that the internal type is boxed and we need to unbox it.

    let element = Box::from_raw(element);

    Box::into_raw(Box::new(FFIDType {
        dtype: DTYPE_LIST,
        nullable,
        type_info: Some(FFIDTypeInfo {
            list_dtype: ManuallyDrop::new(element),
        }),
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIDType_new_struct(
    names: *const *const c_char,
    types: *const *mut FFIDType,
    len: usize,
    nullable: bool,
) -> *mut FFIDType {
    let names = (0..len)
        .map(|i| CStr::from_ptr(*names.add(i)).into())
        .collect();

    let types = (0..len).map(|i| Box::from_raw(*types.add(i))).collect();

    Box::into_raw(Box::new(FFIDType {
        dtype: DTYPE_STRUCT,
        nullable,
        type_info: Some(FFIDTypeInfo {
            struct_dtype: ManuallyDrop::new(Box::new(FFIStructDType { names, types })),
        }),
    }))
}

// TODO(aduffy): create constructors for StructDType and ExtDType.

/// Free an [`FFIDType`] and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIDType_free(ffi_dtype: *mut FFIDType) {
    let mut ffi_dtype = Box::from_raw(ffi_dtype);

    // Handle freeing resources for all of the nested DType variants.
    match ffi_dtype.dtype {
        DTYPE_LIST => {
            let type_info = ffi_dtype
                .type_info
                .take()
                .vortex_expect("list type_info")
                .list_dtype;

            FFIDType_free(Box::into_raw(ManuallyDrop::into_inner(type_info)));
        }
        DTYPE_STRUCT => {
            let type_info = ffi_dtype
                .type_info
                .take()
                .vortex_expect("list type_info")
                .struct_dtype;

            let struct_type_info = *ManuallyDrop::into_inner(type_info);
            // Iterate over all of the DTypes internally.
            for dtype in struct_type_info.types {
                FFIDType_free(Box::into_raw(dtype));
            }
        }
        DTYPE_EXTENSION => {
            let type_info = ffi_dtype
                .type_info
                .take()
                .vortex_expect("list type_info")
                .extension_dtype;

            let ext_type_info = *ManuallyDrop::into_inner(type_info);
            FFIDType_free(Box::into_raw(ext_type_info.storage_dtype));
        }
        _ => {
            // non-nested DType, no special cleanup
        }
    }

    drop(ffi_dtype);
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
pub unsafe extern "C" fn FFIDType_get(ffi_dtype: *const FFIDType) -> u8 {
    let ffi_dtype = &*ffi_dtype;
    ffi_dtype.dtype
}

/// For `DTYPE_STRUCT` variant DTypes, get the number of fields.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIDType_field_count(ffi_dtype: *const FFIDType) -> u32 {
    let ffi_dtype = &*ffi_dtype;

    assert_eq!(
        ffi_dtype.dtype, DTYPE_STRUCT,
        "FFIDType_field_count: not a struct dtype"
    );

    let struct_dtype = ffi_dtype
        .type_info
        .as_ref()
        .vortex_expect("struct type_info")
        .struct_dtype
        .as_ref();

    struct_dtype
        .names
        .len()
        .try_into()
        .vortex_expect("names length must fit in u32")
}

/// Get the name of a field in a `DTYPE_STRUCT` variant DType.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIDType_field_name(
    ffi_dtype: *const FFIDType,
    index: u32,
) -> *const c_char {
    let ffi_dtype = &*ffi_dtype;

    assert_eq!(
        ffi_dtype.dtype, DTYPE_STRUCT,
        "FFIDType_field_name: not a struct dtype"
    );

    let struct_dtype = ffi_dtype
        .type_info
        .as_ref()
        .vortex_expect("struct type_info")
        .struct_dtype
        .as_ref();

    struct_dtype.names[index as usize].as_ptr()
}

/// Get the dtype of a field in a `DTYPE_STRUCT` variant DType.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIDType_field_dtype(
    ffi_dtype: *const FFIDType,
    index: u32,
) -> *const FFIDType {
    let ffi_dtype = &*ffi_dtype;

    assert_eq!(
        ffi_dtype.dtype, DTYPE_STRUCT,
        "FFIDType_field_dtype: not a struct dtype"
    );

    let struct_dtype = ffi_dtype
        .type_info
        .as_ref()
        .vortex_expect("struct type_info")
        .struct_dtype
        .as_ref();

    struct_dtype.types[index as usize].as_ref()
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

impl From<&DType> for FFIDType {
    fn from(value: &DType) -> Self {
        match value {
            DType::Null => FFIDType {
                dtype: DTYPE_NULL,
                nullable: true,
                type_info: None,
            },
            DType::Bool(nullability) => FFIDType {
                dtype: DTYPE_BOOL,
                nullable: (*nullability).into(),
                type_info: None,
            },
            DType::Primitive(ptype, nullability) => FFIDType {
                dtype: match &ptype {
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
                nullable: (*nullability).into(),
                type_info: None,
            },
            DType::Utf8(nullability) => FFIDType {
                dtype: DTYPE_UTF8,
                nullable: (*nullability).into(),
                type_info: None,
            },
            DType::Binary(nullability) => FFIDType {
                dtype: DTYPE_BINARY,
                nullable: (*nullability).into(),
                type_info: None,
            },
            DType::Struct(struct_dtype, nullability) => FFIDType {
                dtype: DTYPE_STRUCT,
                nullable: (*nullability).into(),
                type_info: Some(FFIDTypeInfo {
                    struct_dtype: ManuallyDrop::new(Box::new(FFIStructDType::from(
                        struct_dtype.as_ref(),
                    ))),
                }),
            },
            DType::List(element_type, nullability) => FFIDType {
                dtype: DTYPE_LIST,
                nullable: (*nullability).into(),
                type_info: Some(FFIDTypeInfo {
                    list_dtype: ManuallyDrop::new(Box::new(FFIDType::from(element_type.as_ref()))),
                }),
            },
            DType::Extension(ext_dtype) => FFIDType {
                dtype: DTYPE_EXTENSION,
                nullable: ext_dtype.storage_dtype().nullability().into(),
                type_info: Some(FFIDTypeInfo {
                    extension_dtype: ManuallyDrop::new(Box::new(FFIExtensionDType::from(
                        ext_dtype.as_ref(),
                    ))),
                }),
            },
        }
    }
}

impl From<&StructDType> for FFIStructDType {
    #[allow(clippy::expect_used)]
    fn from(struct_dtype: &StructDType) -> Self {
        let names: Vec<CString> = struct_dtype
            .names()
            .iter()
            .map(|x| CString::from_str(x).expect("CString"))
            .collect();
        let types: Vec<Box<FFIDType>> = struct_dtype
            .fields()
            .map(|x| Box::new(FFIDType::from(&x)))
            .collect();

        FFIStructDType { names, types }
    }
}

impl From<&ExtDType> for FFIExtensionDType {
    fn from(ext_dtype: &ExtDType) -> Self {
        let id = ext_dtype.id().as_ref().as_bytes().to_vec();
        let metadata = ext_dtype.metadata().map(|x| x.as_ref().to_vec());

        FFIExtensionDType {
            id,
            metadata,
            storage_dtype: Box::new(FFIDType::from(ext_dtype.storage_dtype())),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use crate::dtype::{
        DTYPE_BOOL, DTYPE_PRIMITIVE_U8, DTYPE_STRUCT, DTYPE_UTF8, FFIDType_field_count,
        FFIDType_field_dtype, FFIDType_field_name, FFIDType_free, FFIDType_get, FFIDType_new,
        FFIDType_new_struct,
    };

    #[test]
    fn test_simple() {
        unsafe {
            let ffi_dtype = FFIDType_new(DTYPE_BOOL, true);

            // functions check
            assert_eq!(FFIDType_get(ffi_dtype), DTYPE_BOOL);

            // Field access checks.
            let dtype = &*ffi_dtype;

            assert_eq!(dtype.dtype, DTYPE_BOOL);
            assert!(dtype.nullable);
            assert!(dtype.type_info.is_none());

            // Free the memory.
            FFIDType_free(ffi_dtype);
        }
    }

    #[test]
    fn test_struct() {
        unsafe {
            let name = FFIDType_new(DTYPE_UTF8, false);
            let age = FFIDType_new(DTYPE_PRIMITIVE_U8, true);

            let names = [c"name".as_ptr(), c"age".as_ptr()];

            let dtypes = [name, age];

            let person = FFIDType_new_struct(names.as_ptr(), dtypes.as_ptr(), 2, false);

            assert_eq!(FFIDType_get(&*person), DTYPE_STRUCT);
            assert_eq!(FFIDType_field_count(&*person), 2);

            let name0 = FFIDType_field_name(&*person, 0);
            let name1 = FFIDType_field_name(&*person, 1);

            // Check that name0 and name1 are correct
            assert_eq!(CStr::from_ptr(name0), c"name",);
            assert_eq!(CStr::from_ptr(name1), c"age");

            let dtype0 = FFIDType_field_dtype(&*person, 0);
            let dtype1 = FFIDType_field_dtype(&*person, 1);
            assert_eq!(FFIDType_get(dtype0), DTYPE_UTF8);
            assert_eq!(FFIDType_get(dtype1), DTYPE_PRIMITIVE_U8);

            FFIDType_free(person);
        }
    }
}
