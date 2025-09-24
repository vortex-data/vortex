// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display, Formatter};

use vortex_dtype::half::f16;
use vortex_dtype::{DType, NativePType, PType};
use vortex_error::vortex_panic;

use crate::arrays::BinaryView;

/// Defines the "vector type", a physical type describing the data that's held in the vector.
///
/// See the specific vector view types like primitive views for more details.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VType {
    Bool,
    Primitive(PType),
    Binary,
}

impl Display for VType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            VType::Bool => write!(f, "bool"),
            VType::Primitive(ptype) => write!(f, "{}", ptype),
            VType::Binary => write!(f, "binary"),
        }
    }
}

impl VType {
    pub fn of<T: Element>() -> Self {
        T::vtype()
    }

    pub fn byte_width(&self) -> usize {
        match self {
            VType::Bool => 1,
            VType::Primitive(ptype) => ptype.byte_width(),
            VType::Binary => size_of::<BinaryView>(),
        }
    }
}

/// A trait to identify canonical vector types.
pub trait Element: 'static + Copy + Debug + Send {
    fn vtype() -> VType;
}

/// NOTE: for now, we have chosen to store boolean values as byte-sized booleans instead
///  of packed into a bit mask, this is typically more efficient for SIMD compute operations.
///  For masks, we still use bit-packed booleans.
impl Element for bool {
    fn vtype() -> VType {
        VType::Bool
    }
}

macro_rules! canonical_ptype {
    ($T:ty) => {
        impl Element for $T {
            fn vtype() -> VType {
                VType::Primitive(<$T as NativePType>::PTYPE)
            }
        }
    };
}

canonical_ptype!(u8);
canonical_ptype!(u16);
canonical_ptype!(u32);
canonical_ptype!(u64);
canonical_ptype!(i8);
canonical_ptype!(i16);
canonical_ptype!(i32);
canonical_ptype!(i64);
canonical_ptype!(f16);
canonical_ptype!(f32);
canonical_ptype!(f64);

impl Element for BinaryView {
    fn vtype() -> VType {
        VType::Binary
    }
}

impl From<&DType> for VType {
    fn from(value: &DType) -> Self {
        match value {
            DType::Bool(_) => VType::Bool,
            DType::Primitive(ptype, _) => VType::Primitive(*ptype),
            DType::Utf8(_) => VType::Binary,
            DType::Binary(_) => VType::Binary,
            _ => vortex_panic!("Unsupported dtype for VType: {}", value),
        }
    }
}
