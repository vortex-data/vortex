// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativeDecimalType;
use vortex_dtype::NativePType;
use vortex_dtype::PType;

use crate::Scalar;
use crate::Vector;
use crate::binaryview::BinaryType;
use crate::binaryview::BinaryViewScalar;
use crate::binaryview::BinaryViewType;
use crate::binaryview::BinaryViewVector;
use crate::binaryview::StringType;
use crate::bool::BoolScalar;
use crate::bool::BoolVector;
use crate::decimal::DScalar;
use crate::decimal::DVector;
use crate::decimal::DecimalScalar;
use crate::decimal::DecimalVector;
use crate::fixed_size_list::FixedSizeListScalar;
use crate::fixed_size_list::FixedSizeListVector;
use crate::listview::ListViewScalar;
use crate::listview::ListViewVector;
use crate::null::NullScalar;
use crate::null::NullVector;
use crate::primitive::PScalar;
use crate::primitive::PVector;
use crate::primitive::PrimitiveScalar;
use crate::primitive::PrimitiveVector;
use crate::struct_::StructScalar;
use crate::struct_::StructVector;

/// Represents either a scalar or vector value.
#[derive(Clone, Debug)]
pub enum Datum {
    /// A scalar value.
    Scalar(Scalar),
    /// A vector value.
    Vector(Vector),
}

impl From<Scalar> for Datum {
    fn from(value: Scalar) -> Self {
        Datum::Scalar(value)
    }
}

impl From<Vector> for Datum {
    fn from(value: Vector) -> Self {
        Datum::Vector(value)
    }
}

impl Datum {
    /// Returns the scalar value if this `Datum` is a `Scalar`, otherwise returns `None`.
    pub fn into_scalar(self) -> Option<Scalar> {
        match self {
            Datum::Scalar(scalar) => Some(scalar),
            Datum::Vector(_) => None,
        }
    }

    /// Returns the vector value if this `Datum` is a `Vector`, otherwise returns `None`.
    pub fn into_vector(self) -> Option<Vector> {
        match self {
            Datum::Scalar(_) => None,
            Datum::Vector(vector) => Some(vector),
        }
    }

    /// Returns a reference to the scalar value if this `Datum` is a `Scalar`, otherwise returns `None`.
    pub fn as_scalar(&self) -> Option<&Scalar> {
        match self {
            Datum::Scalar(scalar) => Some(scalar),
            Datum::Vector(_) => None,
        }
    }

    /// Returns a reference to the vector value if this `Datum` is a `Vector`, otherwise returns `None`.
    pub fn as_vector(&self) -> Option<&Vector> {
        match self {
            Datum::Scalar(_) => None,
            Datum::Vector(vector) => Some(vector),
        }
    }
}

impl Datum {
    /// Converts the `Datum` into a `NullDatum`.
    pub fn into_null(self) -> NullDatum {
        match self {
            Datum::Scalar(scalar) => NullDatum::Scalar(scalar.into_null()),
            Datum::Vector(vector) => NullDatum::Vector(vector.into_null()),
        }
    }

    /// Converts the `Datum` into a `BoolDatum`.
    pub fn into_bool(self) -> BoolDatum {
        match self {
            Datum::Scalar(scalar) => BoolDatum::Scalar(scalar.into_bool()),
            Datum::Vector(vector) => BoolDatum::Vector(vector.into_bool()),
        }
    }

    /// Converts the `Datum` into a `PrimitiveDatum`.
    pub fn into_primitive(self) -> PrimitiveDatum {
        match self {
            Datum::Scalar(scalar) => PrimitiveDatum::Scalar(scalar.into_primitive()),
            Datum::Vector(vector) => PrimitiveDatum::Vector(vector.into_primitive()),
        }
    }

    /// Converts the `Datum` into a `DecimalDatum`.
    pub fn into_decimal(self) -> DecimalDatum {
        match self {
            Datum::Scalar(scalar) => DecimalDatum::Scalar(scalar.into_decimal()),
            Datum::Vector(vector) => DecimalDatum::Vector(vector.into_decimal()),
        }
    }

    /// Converts the `Datum` into a `ListViewDatum`.
    pub fn into_list(self) -> ListViewDatum {
        match self {
            Datum::Scalar(scalar) => ListViewDatum::Scalar(scalar.into_list()),
            Datum::Vector(vector) => ListViewDatum::Vector(vector.into_list()),
        }
    }

    /// Converts the `Datum` into a `FixedSizeListDatum`.
    pub fn into_fixed_size_list(self) -> FixedSizeListDatum {
        match self {
            Datum::Scalar(scalar) => FixedSizeListDatum::Scalar(scalar.into_fixed_size_list()),
            Datum::Vector(vector) => FixedSizeListDatum::Vector(vector.into_fixed_size_list()),
        }
    }

    /// Converts the `Datum` into a `StructDatum`.
    pub fn into_struct(self) -> StructDatum {
        match self {
            Datum::Scalar(scalar) => StructDatum::Scalar(scalar.into_struct()),
            Datum::Vector(vector) => StructDatum::Vector(vector.into_struct()),
        }
    }

    /// Converts the `Datum` into a `TypedDatum`.
    pub fn into_typed(self) -> TypedDatum {
        match self {
            Datum::Scalar(sc) => match sc {
                Scalar::Null(s) => TypedDatum::Null(NullDatum::Scalar(s)),
                Scalar::Bool(s) => TypedDatum::Bool(BoolDatum::Scalar(s)),
                Scalar::Decimal(s) => TypedDatum::Decimal(DecimalDatum::Scalar(s)),
                Scalar::Primitive(s) => TypedDatum::Primitive(PrimitiveDatum::Scalar(s)),
                Scalar::String(s) => TypedDatum::String(BinaryViewDatum::Scalar(s)),
                Scalar::Binary(s) => TypedDatum::Binary(BinaryViewDatum::Scalar(s)),
                Scalar::List(s) => TypedDatum::List(ListViewDatum::Scalar(s)),
                Scalar::FixedSizeList(s) => {
                    TypedDatum::FixedSizeList(FixedSizeListDatum::Scalar(s))
                }
                Scalar::Struct(s) => TypedDatum::Struct(StructDatum::Scalar(s)),
            },
            Datum::Vector(vec) => match vec {
                Vector::Null(v) => TypedDatum::Null(NullDatum::Vector(v)),
                Vector::Bool(v) => TypedDatum::Bool(BoolDatum::Vector(v)),
                Vector::Decimal(v) => TypedDatum::Decimal(DecimalDatum::Vector(v)),
                Vector::Primitive(v) => TypedDatum::Primitive(PrimitiveDatum::Vector(v)),
                Vector::String(v) => TypedDatum::String(BinaryViewDatum::Vector(v)),
                Vector::Binary(v) => TypedDatum::Binary(BinaryViewDatum::Vector(v)),
                Vector::List(v) => TypedDatum::List(ListViewDatum::Vector(v)),
                Vector::FixedSizeList(v) => {
                    TypedDatum::FixedSizeList(FixedSizeListDatum::Vector(v))
                }
                Vector::Struct(v) => TypedDatum::Struct(StructDatum::Vector(v)),
            },
        }
    }
}

macro_rules! datum {
    // Non-generic version
    ($Name:ident) => {
        paste::paste! {
            #[doc = concat!("Datum enum for `", stringify!($Name), "`.")]
            pub enum [<$Name Datum>] {
                /// Scalar variant
                Scalar([<$Name Scalar>]),
                /// Vector variant
                Vector([<$Name Vector>]),
            }

            impl From<[<$Name Datum>]> for Datum {
                fn from(val: [<$Name Datum>]) -> Self {
                    match val {
                        [<$Name Datum>]::Scalar(scalar) => Datum::Scalar(Scalar::from(scalar)),
                        [<$Name Datum>]::Vector(vector) => Datum::Vector(Vector::from(vector)),
                    }
                }
            }

            impl From<[<$Name Scalar>]> for Datum {
                fn from(val: [<$Name Scalar>]) -> Self {
                    Datum::Scalar(Scalar::from(val))
                }
            }

            impl From<[<$Name Scalar>]> for [<$Name Datum>] {
                fn from(val: [<$Name Scalar>]) -> Self {
                    [<$Name Datum>]::Scalar(val)
                }
            }

            impl From<[<$Name Vector>]> for [<$Name Datum>] {
                fn from(val: [<$Name Vector>]) -> Self {
                    [<$Name Datum>]::Vector(val)
                }
            }
        }
    };

    // Generic version with trait bound
    ($Name:ident < $T:ident : $Bound:path >) => {
        paste::paste! {
            #[doc = concat!("Datum enum for `", stringify!([<$Name Datum>]<$T: $Bound>), "`.")]
            pub enum [<$Name Datum>]<$T: $Bound> {
                /// Scalar variant
                Scalar([<$Name Scalar>]<$T>),
                /// Vector variant
                Vector([<$Name Vector>]<$T>),
            }

            impl<$T: $Bound> From<[<$Name Datum>]<$T>> for Datum {
                fn from(val: [<$Name Datum>]<$T>) -> Self {
                    match val {
                        [<$Name Datum>]::Scalar(scalar) => Datum::Scalar(Scalar::from(scalar)),
                        [<$Name Datum>]::Vector(vector) => Datum::Vector(Vector::from(vector)),
                    }
                }
            }

            impl<$T: $Bound> From<[<$Name Scalar>]<$T>> for Datum {
                fn from(val: [<$Name Scalar>]<$T>) -> Self {
                    Datum::Scalar(Scalar::from(val))
                }
            }

            impl<$T: $Bound> From<[<$Name Scalar>]<$T>> for [<$Name Datum>]<$T> {
                fn from(val: [<$Name Scalar>]<$T>) -> Self {
                    [<$Name Datum>]::Scalar(val)
                }
            }

            impl<$T: $Bound> From<[<$Name Vector>]<$T>> for [<$Name Datum>]<$T> {
                fn from(val: [<$Name Vector>]<$T>) -> Self {
                    [<$Name Datum>]::Vector(val)
                }
            }
        }
    };
}

datum!(Null);
datum!(Bool);
datum!(Primitive);
datum!(P<T: NativePType>);
datum!(Decimal);
datum!(D<D: NativeDecimalType>);
datum!(BinaryView<T: BinaryViewType>);
datum!(ListView);
datum!(FixedSizeList);
datum!(Struct);

impl PrimitiveDatum {
    /// ptype from datum
    pub fn ptype(&self) -> PType {
        match self {
            PrimitiveDatum::Scalar(sc) => sc.ptype(),
            PrimitiveDatum::Vector(sc) => sc.ptype(),
        }
    }
}

/// A variant of [`Datum`] that is typed.
pub enum TypedDatum {
    /// Null datum.
    Null(NullDatum),
    /// Boolean datum.
    Bool(BoolDatum),
    /// Decimal datum.
    Decimal(DecimalDatum),
    /// Primitive datum.
    Primitive(PrimitiveDatum),
    /// String datum
    String(BinaryViewDatum<StringType>),
    /// Binary datum
    Binary(BinaryViewDatum<BinaryType>),
    /// List datum.
    List(ListViewDatum),
    /// fsl datum.
    FixedSizeList(FixedSizeListDatum),
    /// struct datum.
    Struct(StructDatum),
}
