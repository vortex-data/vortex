// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativeDecimalType;
use vortex_dtype::NativePType;

use crate::Scalar;
use crate::Vector;
use crate::binaryview::BinaryViewScalar;
use crate::binaryview::BinaryViewType;
use crate::binaryview::BinaryViewVector;
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
pub enum Datum {
    /// A scalar value.
    Scalar(Scalar),
    /// A vector value.
    Vector(Vector),
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
}

macro_rules! datum {
    // Non-generic version
    ($Name:ident) => {
        paste::paste! {
            pub enum [<$Name Datum>] {
                Scalar([<$Name Scalar>]),
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
            pub enum [<$Name Datum>]<$T: $Bound> {
                Scalar([<$Name Scalar>]<$T>),
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
