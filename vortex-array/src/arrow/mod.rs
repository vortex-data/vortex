//! Utilities to work with `Arrow` data and types.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;

mod array;
pub mod compute;
mod convert;
mod datum;
mod record_batch;

pub use datum::*;

use crate::arrow::compute::ToArrowOptions;
use crate::validity::Validity;

/// Describes the desired nullability of a Vortex array created from an Arrow array.
///
/// In Arrow, non-nullable struct or list fields may contain nulls, if-and-only-if the nulls are
/// "masked" by the outer array.
#[derive(Clone, Copy, Debug)]
pub enum ArrowNullability {
    Nullable,
    NonNullable,
    NonNullableField,
}

impl From<Nullability> for ArrowNullability {
    fn from(value: Nullability) -> Self {
        match value {
            Nullability::NonNullable => Self::NonNullable,
            Nullability::Nullable => Self::Nullable,
        }
    }
}

impl From<ArrowNullability> for Nullability {
    fn from(value: ArrowNullability) -> Nullability {
        match value {
            ArrowNullability::Nullable => Nullability::Nullable,
            ArrowNullability::NonNullable => Nullability::NonNullable,
            ArrowNullability::NonNullableField => Nullability::NonNullable,
        }
    }
}

impl From<&arrow_schema::Field> for ArrowNullability {
    fn from(value: &arrow_schema::Field) -> Self {
        match value.is_nullable() {
            true => Self::Nullable,
            false => Self::NonNullableField,
        }
    }
}

impl From<&DType> for ArrowNullability {
    fn from(value: &DType) -> Self {
        match value.is_nullable() {
            true => Self::Nullable,
            false => Self::NonNullable,
        }
    }
}

impl ArrowNullability {
    pub fn from_top_level_is_nullable(is_nullable: bool) -> Self {
        match is_nullable {
            true => Self::Nullable,
            false => Self::NonNullable,
        }
    }

    pub fn is_nullable(self) -> bool {
        match self {
            ArrowNullability::Nullable => true,
            ArrowNullability::NonNullable => false,
            ArrowNullability::NonNullableField => false,
        }
    }

    pub fn into_validity(self, nulls: Option<&NullBuffer>) -> Validity {
        match self {
            ArrowNullability::Nullable => nulls
                .map(|nulls| {
                    if nulls.null_count() == nulls.len() {
                        Validity::AllInvalid
                    } else {
                        Validity::from(nulls.inner().clone())
                    }
                })
                .unwrap_or_else(|| Validity::AllValid),
            ArrowNullability::NonNullable => {
                assert!(nulls.map(|x| x.null_count() == 0).unwrap_or(true));
                Validity::NonNullable
            }
            ArrowNullability::NonNullableField => {
                // Non-nullable fields may contain "masked" nulls, so we do not check the null
                // count. We assume Arrow constructed a valid struct array.
                Validity::NonNullable
            }
        }
    }
}

pub trait FromArrowArray<A> {
    fn from_arrow(array: A, nullable: ArrowNullability) -> Self;
}

pub trait IntoArrowArray {
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef>;

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef>;
}

impl IntoArrowArray for crate::ArrayRef {
    /// Convert this [`crate::ArrayRef`] into an Arrow [`crate::ArrayRef`] by using the array's preferred
    /// Arrow [`DataType`].
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef> {
        compute::to_arrow_opts(&self, &ToArrowOptions { arrow_type: None })
    }

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef> {
        compute::to_arrow_opts(
            &self,
            &ToArrowOptions {
                arrow_type: Some(data_type.clone()),
            },
        )
    }
}
