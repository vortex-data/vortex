// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(connor): Explain what vectors are, why we need them for the new operator model of arrays,
// differences from Arrow (builders and arrays and scalars), etc.
//! Immutable and mutable decompressed (canonical) vectors for Vortex.

#![deny(clippy::missing_errors_doc)]
#![deny(clippy::missing_panics_doc)]
#![deny(clippy::missing_safety_doc)]
#![deny(missing_docs)]

pub mod binaryview;
pub mod bool;
pub mod decimal;
pub mod fixed_size_list;
pub mod listview;
pub mod null;
pub mod primitive;
pub mod struct_;

mod datum;
mod scalar;
mod scalar_ops;
mod vector;
mod vector_mut;
mod vector_ops;

pub use datum::*;
pub use scalar::Scalar;
pub use scalar_ops::ScalarOps;
pub use vector::Vector;
pub use vector_mut::VectorMut;
pub use vector_ops::VectorMutOps;
pub use vector_ops::VectorOps;
use vortex_dtype::DType;

mod macros;
mod private;
mod scalar_macros;

/// Returns true if the datum is compatible with the provided data type.
pub fn datum_matches_dtype(datum: &Datum, dtype: &DType) -> bool {
    match datum {
        Datum::Scalar(scalar) => scalar_matches_dtype(scalar, dtype),
        Datum::Vector(vector) => vector_matches_dtype(vector, dtype),
    }
}

/// Returns true if the vector is compatible with the provided data type.
///
/// This means that the vector's physical representation is compatible with the data type,
/// typically meaning the enum variants match. In the case of nested types, this function
/// recursively checks the child types.
///
/// This function also checks that if the data type is non-nullable, the vector contains no nulls,
pub fn vector_matches_dtype(vector: &Vector, dtype: &DType) -> bool {
    if !dtype.is_nullable() && vector.validity().false_count() > 0 {
        // Non-nullable dtype cannot have nulls in the vector.
        return false;
    }

    // Note that we don't match a tuple here to make sure we have an exhaustive match that will
    // fail to compile if we ever add new DTypes.
    match dtype {
        DType::Null => {
            matches!(vector, Vector::Null(_))
        }
        DType::Bool(_) => {
            matches!(vector, Vector::Bool(_))
        }
        DType::Primitive(ptype, _) => match vector {
            Vector::Primitive(v) => ptype == &v.ptype(),
            _ => false,
        },
        DType::Decimal(dec_type, _) => match vector {
            Vector::Decimal(v) => {
                dec_type.precision() == v.precision() && dec_type.scale() == v.scale()
            }
            _ => false,
        },
        DType::Utf8(_) => {
            matches!(vector, Vector::String(_))
        }
        DType::Binary(_) => {
            matches!(vector, Vector::Binary(_))
        }
        DType::List(elements, _) => match vector {
            Vector::List(v) => vector_matches_dtype(v.elements(), elements.as_ref()),
            _ => false,
        },
        DType::FixedSizeList(elements, size, _) => match vector {
            Vector::FixedSizeList(v) => {
                v.element_size() == *size && vector_matches_dtype(v.elements(), elements.as_ref())
            }
            _ => false,
        },
        DType::Struct(fields, _) => match vector {
            Vector::Struct(v) => {
                if fields.nfields() != v.fields().len() {
                    return false;
                }
                for (field_dtype, field_vector) in fields.fields().zip(v.fields().iter()) {
                    if !vector_matches_dtype(field_vector, &field_dtype) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        },
        DType::Extension(ext_dtype) => {
            // For extension types, we check the storage type.
            vector_matches_dtype(vector, ext_dtype.storage_dtype())
        }
    }
}

/// Returns true if the scalar's is compatible with the provided data type.
pub fn scalar_matches_dtype(scalar: &Scalar, dtype: &DType) -> bool {
    if !dtype.is_nullable() && scalar.is_valid() {
        // Non-nullable dtype cannot have nulls in the scalar.
        return false;
    }

    // Note that we don't match a tuple here to make sure we have an exhaustive match that will
    // fail to compile if we ever add new DTypes.
    match dtype {
        DType::Null => {
            matches!(scalar, Scalar::Null(_))
        }
        DType::Bool(_) => {
            matches!(scalar, Scalar::Bool(_))
        }
        DType::Primitive(ptype, _) => match scalar {
            Scalar::Primitive(s) => ptype == &s.ptype(),
            _ => false,
        },
        DType::Decimal(dec_type, _) => match scalar {
            Scalar::Decimal(s) => {
                dec_type.precision() == s.precision() && dec_type.scale() == s.scale()
            }
            _ => false,
        },
        DType::Utf8(_) => {
            matches!(scalar, Scalar::String(_))
        }
        DType::Binary(_) => {
            matches!(scalar, Scalar::Binary(_))
        }
        DType::List(_, _) => match scalar {
            Scalar::List(s) => vector_matches_dtype(&Vector::from(s.value().clone()), dtype),
            _ => false,
        },
        DType::FixedSizeList(..) => match scalar {
            Scalar::FixedSizeList(s) => {
                vector_matches_dtype(&Vector::from(s.value().clone()), dtype)
            }
            _ => false,
        },
        DType::Struct(_, _) => match scalar {
            Scalar::Struct(s) => vector_matches_dtype(&Vector::from(s.value().clone()), dtype),
            _ => false,
        },
        DType::Extension(ext_dtype) => {
            // For extension types, we check the storage type.
            scalar_matches_dtype(scalar, ext_dtype.storage_dtype())
        }
    }
}
