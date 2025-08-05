// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::BinaryView;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::half::f16;

/// Defines the "vector type", a physical type describing the data that's held in the vector.
///
/// See the specific vector view types, e.g. [`PrimitiveVector`], for more details.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VType {
    Bool,
    Primitive(PType),
    Utf8,
    Binary,
}

impl VType {
    pub fn byte_width(&self) -> usize {
        match self {
            VType::Bool => 1,
            VType::Primitive(ptype) => ptype.byte_width(),
            VType::Utf8 => size_of::<BinaryView>(),
            VType::Binary => size_of::<BinaryView>(),
        }
    }
}

/// A trait to identify canonical vector types.
pub trait Canonical {
    type Element: 'static + Copy;

    fn vtype() -> VType;
}

struct Bool;
impl Canonical for Bool {
    /// NOTE: for now, we have chosen to store boolean values as byte-sized booleans instead
    ///  of packed into a bit mask, this is typically more efficient for SIMD compute operations.
    ///  For masks, we still use bit-packed booleans.
    type Element = bool;

    fn vtype() -> VType {
        VType::Bool
    }
}

macro_rules! canonical_ptype {
    ($T:ty) => {
        impl Canonical for $T {
            type Element = $T;

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

pub trait BinaryType {
    type Slice: ?Sized;
}

struct Utf8;
impl BinaryType for Utf8 {
    type Slice = str;
}
impl Canonical for Utf8 {
    type Element = BinaryView;

    fn vtype() -> VType {
        VType::Utf8
    }
}

struct Binary;
impl BinaryType for Binary {
    type Slice = [u8];
}
impl Canonical for Binary {
    type Element = BinaryView;

    fn vtype() -> VType {
        VType::Binary
    }
}
