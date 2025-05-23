use std::fmt::{Display, Formatter};

use pyo3::prelude::*;
use vortex::dtype::PType;

/// Enum for primitive types.
#[pyclass(name = "PType", module = "vortex", frozen, eq, eq_int, str)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PyPType {
    /// An 8-bit unsigned integer
    U8 = 0,
    /// A 16-bit unsigned integer
    U16 = 1,
    /// A 32-bit unsigned integer
    U32 = 2,
    /// A 64-bit unsigned integer
    U64 = 3,
    /// An 8-bit signed integer
    I8 = 4,
    /// A 16-bit signed integer
    I16 = 5,
    /// A 32-bit signed integer
    I32 = 6,
    /// A 64-bit signed integer
    I64 = 7,
    /// A 16-bit floating point number
    F16 = 8,
    /// A 32-bit floating point number
    F32 = 9,
    /// A 64-bit floating point number
    F64 = 10,
}

impl Display for PyPType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U8 => write!(f, "u8"),
            Self::U16 => write!(f, "u16"),
            Self::U32 => write!(f, "u32"),
            Self::U64 => write!(f, "u64"),
            Self::I8 => write!(f, "i8"),
            Self::I16 => write!(f, "i16"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::F16 => write!(f, "f16"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
        }
    }
}

impl From<PType> for PyPType {
    fn from(value: PType) -> Self {
        match value {
            PType::U8 => Self::U8,
            PType::U16 => Self::U16,
            PType::U32 => Self::U32,
            PType::U64 => Self::U64,
            PType::I8 => Self::I8,
            PType::I16 => Self::I16,
            PType::I32 => Self::I32,
            PType::I64 => Self::I64,
            PType::F16 => Self::F16,
            PType::F32 => Self::F32,
            PType::F64 => Self::F64,
        }
    }
}

impl From<PyPType> for PType {
    fn from(value: PyPType) -> Self {
        match value {
            PyPType::U8 => PType::U8,
            PyPType::U16 => PType::U16,
            PyPType::U32 => PType::U32,
            PyPType::U64 => PType::U64,
            PyPType::I8 => PType::I8,
            PyPType::I16 => PType::I16,
            PyPType::I32 => PType::I32,
            PyPType::I64 => PType::I64,
            PyPType::F16 => PType::F16,
            PyPType::F32 => PType::F32,
            PyPType::F64 => PType::F64,
        }
    }
}
