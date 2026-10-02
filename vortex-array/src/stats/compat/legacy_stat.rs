// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Historical field identifiers and their scalar types.
//!
//! Footer values are finalized results. Legacy zone tables expose only fields whose values also
//! have the semantics of aggregate states; boolean sortedness and constantness flags do not.

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use enum_iterator::Sequence;
use enum_iterator::all;
use num_enum::IntoPrimitive;
use num_enum::TryFromPrimitive;
use vortex_error::VortexExpect;

use crate::aggregate_fn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;

/// Field identifiers used by historical array, footer, and zone-map metadata.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Sequence,
    IntoPrimitive,
    TryFromPrimitive,
)]
#[repr(u8)]
pub enum LegacyStat {
    /// Whether all values are equal, including nulls.
    IsConstant = 0,
    /// Ascending sortedness flag.
    IsSorted = 1,
    /// Strict ascending sortedness flag.
    IsStrictSorted = 2,
    /// The maximum value in the array, ignoring nulls.
    Max = 3,
    /// The minimum value in the array, ignoring nulls.
    Min = 4,
    /// The sum of the non-null values of the array.
    Sum = 5,
    /// The number of null values in the array.
    NullCount = 6,
    /// The uncompressed size of the array in bytes.
    UncompressedSizeInBytes = 7,
    /// The number of NaN values in the array.
    NaNCount = 8,
}

impl LegacyStat {
    /// Scalar type used by historical zone tables before making each field nullable.
    ///
    /// Footer conversion separately retains metadata for types that current kernels cannot handle.
    pub fn dtype(&self, data_type: &DType) -> Option<DType> {
        Some(match self {
            Self::IsConstant => DType::Bool(NonNullable),
            Self::IsSorted => DType::Bool(NonNullable),
            Self::IsStrictSorted => DType::Bool(NonNullable),
            Self::Max if matches!(data_type, DType::Null) => return None,
            Self::Max => data_type.clone(),
            Self::Min if matches!(data_type, DType::Null) => return None,
            Self::Min => data_type.clone(),
            Self::NullCount => {
                return aggregate_fn::fns::null_count::NullCount
                    .return_dtype(&EmptyOptions, data_type);
            }
            Self::UncompressedSizeInBytes => {
                return aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes
                    .return_dtype(&EmptyOptions, data_type);
            }
            Self::NaNCount => {
                return aggregate_fn::fns::nan_count::NanCount
                    .return_dtype(&EmptyOptions, data_type);
            }
            Self::Sum => {
                // Legacy statistics follow NaN-skipping semantics; request it explicitly.
                return aggregate_fn::fns::sum::Sum
                    .return_dtype(&NumericalAggregateOpts::skip_nans(), data_type);
            }
        })
    }

    /// Aggregate whose state can be supplied by this legacy field.
    ///
    /// Sortedness and constantness flags are finalized results without the required boundary state.
    /// They are available only through [`Self::finalized_fn`].
    pub fn aggregate_fn(&self) -> Option<AggregateFnRef> {
        // Request the historical NaN-skipping semantics explicitly.
        Some(match self {
            Self::Max => aggregate_fn::fns::max::Max.bind(NumericalAggregateOpts::skip_nans()),
            Self::Min => aggregate_fn::fns::min::Min.bind(NumericalAggregateOpts::skip_nans()),
            Self::Sum => aggregate_fn::fns::sum::Sum.bind(NumericalAggregateOpts::skip_nans()),
            Self::NullCount => aggregate_fn::fns::null_count::NullCount.bind(EmptyOptions),
            Self::NaNCount => aggregate_fn::fns::nan_count::NanCount.bind(EmptyOptions),
            Self::UncompressedSizeInBytes => {
                aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes
                    .bind(EmptyOptions)
            }
            Self::IsConstant | Self::IsSorted | Self::IsStrictSorted => return None,
        })
    }

    /// Return the statistic represented by `aggregate_fn`, if it has a legacy stat slot.
    ///
    /// Min/max/sum statistics skip NaN values, so NaN-including configurations of those
    /// aggregates have no stat slot.
    pub fn from_aggregate_fn(aggregate_fn: &AggregateFnRef) -> Option<Self> {
        if let Some(options) = aggregate_fn.as_opt::<aggregate_fn::fns::sum::Sum>() {
            return options.skip_nans.then_some(Self::Sum);
        }
        if aggregate_fn.is::<aggregate_fn::fns::nan_count::NanCount>() {
            return Some(Self::NaNCount);
        }
        if aggregate_fn.is::<aggregate_fn::fns::null_count::NullCount>() {
            return Some(Self::NullCount);
        }
        if let Some(options) = aggregate_fn.as_opt::<aggregate_fn::fns::min::Min>() {
            return options.skip_nans.then_some(Self::Min);
        }
        if let Some(options) = aggregate_fn.as_opt::<aggregate_fn::fns::max::Max>() {
            return options.skip_nans.then_some(Self::Max);
        }
        if aggregate_fn
            .is::<aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes>()
        {
            return Some(Self::UncompressedSizeInBytes);
        }

        None
    }

    /// Aggregate whose finalized result occupies this historical field.
    pub fn finalized_fn(self) -> AggregateFnRef {
        match self {
            Self::IsConstant => aggregate_fn::fns::is_constant::IsConstant.bind(EmptyOptions),
            Self::IsSorted | Self::IsStrictSorted => aggregate_fn::fns::is_sorted::IsSorted.bind(
                aggregate_fn::fns::is_sorted::IsSortedOptions {
                    strict: self == Self::IsStrictSorted,
                },
            ),
            _ => self
                .aggregate_fn()
                .vortex_expect("numeric legacy fields have aggregate functions"),
        }
    }

    /// Scalar type of a finalized footer field, even when current kernels cannot compute it.
    pub(super) fn finalized_dtype(self, input_dtype: &DType) -> Option<DType> {
        match self {
            Self::Min | Self::Max => {
                (!matches!(input_dtype, DType::Null)).then(|| input_dtype.as_nullable())
            }
            Self::IsConstant | Self::IsSorted | Self::IsStrictSorted => {
                Some(DType::Bool(NonNullable))
            }
            Self::NullCount | Self::NaNCount | Self::UncompressedSizeInBytes => {
                Some(DType::Primitive(PType::U64, NonNullable))
            }
            Self::Sum => self.dtype(input_dtype).or_else(|| {
                // Older writers also summed extension storage values.
                if let DType::Extension(ext) = input_dtype {
                    self.finalized_dtype(ext.storage_dtype())
                } else {
                    None
                }
            }),
        }
    }

    /// Historical field for a finalized aggregate result, including its options.
    pub fn from_finalized_fn(aggregate_fn: &AggregateFnRef) -> Option<Self> {
        Self::all().find(|stat| &stat.finalized_fn() == aggregate_fn)
    }

    /// Historical field name in a zone-map table.
    pub fn name(&self) -> &str {
        match self {
            Self::IsConstant => "is_constant",
            Self::IsSorted => "is_sorted",
            Self::IsStrictSorted => "is_strict_sorted",
            Self::Max => "max",
            Self::Min => "min",
            Self::NullCount => "null_count",
            Self::UncompressedSizeInBytes => "uncompressed_size_in_bytes",
            Self::Sum => "sum",
            Self::NaNCount => "nan_count",
        }
    }

    /// Iterate over historical fields in wire-ID order.
    pub fn all() -> impl Iterator<Item = LegacyStat> {
        all::<Self>()
    }
}

impl Display for LegacyStat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
