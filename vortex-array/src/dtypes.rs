//! Pre-created DTypes.
//!
//! Many DTypes are reused throughout the codebase. We statically enumerate the common DTypes, and
//! use them in place of direct construction.
//!
//! While a `DType` is at the time of writing this, 40 bytes, an Arc<DType> is only 8 bytes,
//! and can be shared/copied without any extra allocations.

use std::sync::{Arc, LazyLock};

use vortex_dtype::{DType, Nullability, PType};

pub static DTYPE_BOOL_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Bool(Nullability::NonNullable)));
pub static DTYPE_BOOL_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Bool(Nullability::Nullable)));

pub static DTYPE_U8_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)));
pub static DTYPE_U8_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::U8, Nullability::Nullable)));

pub static DTYPE_U16_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::U16, Nullability::NonNullable)));
pub static DTYPE_U16_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::U16, Nullability::Nullable)));

pub static DTYPE_U32_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable)));
pub static DTYPE_U32_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::U32, Nullability::Nullable)));

pub static DTYPE_U64_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)));
pub static DTYPE_U64_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::U64, Nullability::Nullable)));

pub static DTYPE_I8_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::I8, Nullability::NonNullable)));
pub static DTYPE_I8_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::I8, Nullability::Nullable)));

pub static DTYPE_I16_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::I16, Nullability::NonNullable)));
pub static DTYPE_I16_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::I16, Nullability::Nullable)));

pub static DTYPE_I32_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)));
pub static DTYPE_I32_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)));

pub static DTYPE_I64_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)));
pub static DTYPE_I64_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)));

pub static DTYPE_F16_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::F16, Nullability::NonNullable)));
pub static DTYPE_F16_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::F16, Nullability::Nullable)));

pub static DTYPE_F32_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)));
pub static DTYPE_F32_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::F32, Nullability::Nullable)));

pub static DTYPE_F64_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)));
pub static DTYPE_F64_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Primitive(PType::F64, Nullability::Nullable)));

pub static DTYPE_STRING_NONNULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Utf8(Nullability::NonNullable)));
pub static DTYPE_STRING_NULL: LazyLock<Arc<DType>> =
    LazyLock::new(|| Arc::new(DType::Utf8(Nullability::Nullable)));

#[macro_export]
macro_rules! primitive_dtype {
    ($ptype:expr, $nullability:expr) => {
        match ($ptype, $nullability) {
            (vortex_dtype::PType::U8, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_U8_NONNULL.clone()
            }
            (vortex_dtype::PType::U8, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_U8_NULL.clone()
            }
            (vortex_dtype::PType::U16, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_U16_NONNULL.clone()
            }
            (vortex_dtype::PType::U16, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_U16_NULL.clone()
            }
            (vortex_dtype::PType::U32, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_U32_NONNULL.clone()
            }
            (vortex_dtype::PType::U32, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_U32_NULL.clone()
            }
            (vortex_dtype::PType::U64, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_U64_NONNULL.clone()
            }
            (vortex_dtype::PType::U64, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_U64_NULL.clone()
            }
            (vortex_dtype::PType::I8, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_I8_NONNULL.clone()
            }
            (vortex_dtype::PType::I8, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_I8_NULL.clone()
            }
            (vortex_dtype::PType::I16, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_I16_NONNULL.clone()
            }
            (vortex_dtype::PType::I16, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_I16_NULL.clone()
            }
            (vortex_dtype::PType::I32, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_I32_NONNULL.clone()
            }
            (vortex_dtype::PType::I32, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_I32_NULL.clone()
            }
            (vortex_dtype::PType::I64, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_I64_NONNULL.clone()
            }
            (vortex_dtype::PType::I64, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_I64_NULL.clone()
            }
            (vortex_dtype::PType::F16, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_F16_NONNULL.clone()
            }
            (vortex_dtype::PType::F16, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_F16_NULL.clone()
            }
            (vortex_dtype::PType::F32, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_F32_NONNULL.clone()
            }
            (vortex_dtype::PType::F32, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_F32_NULL.clone()
            }
            (vortex_dtype::PType::F64, vortex_dtype::Nullability::NonNullable) => {
                $crate::dtypes::DTYPE_F64_NONNULL.clone()
            }
            (vortex_dtype::PType::F64, vortex_dtype::Nullability::Nullable) => {
                $crate::dtypes::DTYPE_F64_NULL.clone()
            }
        }
    };
}

#[macro_export]
macro_rules! primitive_dtype_ref {
    ($ptype:expr, $nullability:expr) => {
        match ($ptype, $nullability) {
            (vortex_dtype::PType::U8, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_U8_NONNULL
            }
            (vortex_dtype::PType::U8, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_U8_NULL
            }
            (vortex_dtype::PType::U16, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_U16_NONNULL
            }
            (vortex_dtype::PType::U16, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_U16_NULL
            }
            (vortex_dtype::PType::U32, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_U32_NONNULL
            }
            (vortex_dtype::PType::U32, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_U32_NULL
            }
            (vortex_dtype::PType::U64, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_U64_NONNULL
            }
            (vortex_dtype::PType::U64, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_U64_NULL
            }
            (vortex_dtype::PType::I8, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_I8_NONNULL
            }
            (vortex_dtype::PType::I8, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_I8_NULL
            }
            (vortex_dtype::PType::I16, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_I16_NONNULL
            }
            (vortex_dtype::PType::I16, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_I16_NULL
            }
            (vortex_dtype::PType::I32, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_I32_NONNULL
            }
            (vortex_dtype::PType::I32, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_I32_NULL
            }
            (vortex_dtype::PType::I64, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_I64_NONNULL
            }
            (vortex_dtype::PType::I64, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_I64_NULL
            }
            (vortex_dtype::PType::F16, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_F16_NONNULL
            }
            (vortex_dtype::PType::F16, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_F16_NULL
            }
            (vortex_dtype::PType::F32, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_F32_NONNULL
            }
            (vortex_dtype::PType::F32, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_F32_NULL
            }
            (vortex_dtype::PType::F64, vortex_dtype::Nullability::NonNullable) => {
                &$crate::dtypes::DTYPE_F64_NONNULL
            }
            (vortex_dtype::PType::F64, vortex_dtype::Nullability::Nullable) => {
                &$crate::dtypes::DTYPE_F64_NULL
            }
        }
    };
}

#[macro_export]
macro_rules! bool_dtype {
    ($nullability:expr) => {
        match ($nullability) {
            vortex_dtype::Nullability::NonNullable => $crate::dtypes::DTYPE_BOOL_NONNULL.clone(),
            vortex_dtype::Nullability::Nullable => $crate::dtypes::DTYPE_BOOL_NULL.clone(),
        }
    };
}

pub use {bool_dtype, primitive_dtype, primitive_dtype_ref};
