// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};

use anyhow::Result;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_schema::Field;
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::{DType as RustDType, DecimalDType, Nullability, PType as RustPType};

use crate::ffi;
pub(crate) struct DType {
    pub(crate) inner: RustDType,
}

pub(crate) fn dtype_null() -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Null,
    })
}

pub(crate) fn dtype_bool(nullable: bool) -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Bool(nullability_from_bool(nullable)),
    })
}

pub(crate) fn dtype_primitive(ptype: ffi::PType, nullable: bool) -> Box<DType> {
    let vortex_ptype = match ptype {
        ffi::PType::U8 => RustPType::U8,
        ffi::PType::U16 => RustPType::U16,
        ffi::PType::U32 => RustPType::U32,
        ffi::PType::U64 => RustPType::U64,
        ffi::PType::I8 => RustPType::I8,
        ffi::PType::I16 => RustPType::I16,
        ffi::PType::I32 => RustPType::I32,
        ffi::PType::I64 => RustPType::I64,
        ffi::PType::F16 => RustPType::F16,
        ffi::PType::F32 => RustPType::F32,
        ffi::PType::F64 => RustPType::F64,
        _ => unreachable!(),
    };
    Box::new(DType {
        inner: RustDType::Primitive(vortex_ptype, nullability_from_bool(nullable)),
    })
}

pub(crate) fn dtype_decimal(precision: u8, scale: i8, nullable: bool) -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Decimal(
            DecimalDType::new(precision, scale),
            nullability_from_bool(nullable),
        ),
    })
}

pub(crate) fn dtype_utf8(nullable: bool) -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Utf8(nullability_from_bool(nullable)),
    })
}

pub(crate) fn dtype_binary(nullable: bool) -> Box<DType> {
    Box::new(DType {
        inner: RustDType::Binary(nullability_from_bool(nullable)),
    })
}

impl Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{0}", self.inner)
    }
}

pub(crate) fn nullability_from_bool(nullable: bool) -> Nullability {
    if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    }
}

pub(crate) unsafe fn from_arrow(ffi_schema: *mut u8, non_nullable: bool) -> Result<Box<DType>> {
    let arrow_schema = unsafe { FFI_ArrowSchema::from_raw(ffi_schema as *mut FFI_ArrowSchema) };
    let arrow_dtype = arrow_schema::DataType::try_from(&arrow_schema)?;
    Ok(Box::new(DType {
        inner: RustDType::from_arrow(&Field::new("_", arrow_dtype, !non_nullable)),
    }))
}
