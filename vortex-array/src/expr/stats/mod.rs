// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use enum_iterator::Sequence;
use enum_iterator::all;
use num_enum::IntoPrimitive;
use num_enum::TryFromPrimitive;

use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;

mod bound;
mod precision;
mod provider;
mod stat_bound;

pub use bound::*;
pub use precision::*;
pub use provider::*;
pub use stat_bound::*;

use crate::aggregate_fn;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;

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
pub enum Stat {
    /// Whether all values are the same (nulls are not equal to other non-null values,
    /// so this is true iff all values are null or all values are the same non-null value)
    IsConstant = 0,
    /// Whether the non-null values in the array are sorted in ascending order (i.e., we skip nulls)
    /// This may later be extended to support descending order, but for now we only support ascending order.
    IsSorted = 1,
    /// Whether the non-null values in the array are strictly sorted in ascending order (i.e., sorted with no duplicates)
    /// This may later be extended to support descending order, but for now we only support ascending order.
    IsStrictSorted = 2,
    /// The maximum value in the array (ignoring nulls, unless all values are null)
    Max = 3,
    /// The minimum value in the array (ignoring nulls, unless all values are null)
    Min = 4,
    /// The sum of the non-null values of the array.
    Sum = 5,
    /// The number of null values in the array
    NullCount = 6,
    /// The uncompressed size of the array in bytes
    UncompressedSizeInBytes = 7,
    /// The number of NaN values in the array
    NaNCount = 8,
}

/// These structs allow the extraction of the bound from the `Precision` value.
/// They tie together the Stat and the StatBound, which allows the bound to be extracted.
pub struct Max;

pub struct Min;

pub struct Sum;

pub struct IsConstant;

pub struct IsSorted;

pub struct IsStrictSorted;

pub struct NullCount;

pub struct UncompressedSizeInBytes;

pub struct NaNCount;

impl StatType<bool> for IsConstant {
    type Bound = Precision<bool>;

    const STAT: Stat = Stat::IsConstant;
}

impl StatType<bool> for IsSorted {
    type Bound = Precision<bool>;

    const STAT: Stat = Stat::IsSorted;
}

impl StatType<bool> for IsStrictSorted {
    type Bound = Precision<bool>;

    const STAT: Stat = Stat::IsStrictSorted;
}

impl<T: PartialOrd + Clone> StatType<T> for NullCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::NullCount;
}

impl<T: PartialOrd + Clone> StatType<T> for UncompressedSizeInBytes {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::UncompressedSizeInBytes;
}

impl<T: PartialOrd + Clone + Debug> StatType<T> for Max {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::Max;
}

impl<T: PartialOrd + Clone + Debug> StatType<T> for Min {
    type Bound = LowerBound<T>;

    const STAT: Stat = Stat::Min;
}

impl<T: PartialOrd + Clone + Debug> StatType<T> for Sum {
    type Bound = Precision<T>;

    const STAT: Stat = Stat::Sum;
}

impl<T: PartialOrd + Clone> StatType<T> for NaNCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::NaNCount;
}

impl Stat {
    /// Whether the statistic is commutative (i.e., whether merging can be done independently of ordering)
    /// e.g., min/max are commutative, but is_sorted is not
    pub fn is_commutative(&self) -> bool {
        // NOTE: we prefer this syntax to force a compile error if we add a new stat
        match self {
            Self::IsConstant
            | Self::Max
            | Self::Min
            | Self::NullCount
            | Self::Sum
            | Self::NaNCount
            | Self::UncompressedSizeInBytes => true,
            Self::IsSorted | Self::IsStrictSorted => false,
        }
    }

    /// Whether the statistic has the same dtype as the array it's computed on
    pub fn has_same_dtype_as_array(&self) -> bool {
        matches!(self, Stat::Min | Stat::Max)
    }

    /// Return the [`DType`] of the statistic scalar assuming the array is of the given [`DType`].
    pub fn dtype(&self, data_type: &DType) -> Option<DType> {
        Some(match self {
            Self::IsConstant => DType::Bool(NonNullable),
            Self::IsSorted => DType::Bool(NonNullable),
            Self::IsStrictSorted => DType::Bool(NonNullable),
            Self::Max if matches!(data_type, DType::Null) => return None,
            Self::Max => data_type.clone(),
            Self::Min if matches!(data_type, DType::Null) => return None,
            Self::Min => data_type.clone(),
            Self::NullCount => DType::Primitive(PType::U64, NonNullable),
            Self::UncompressedSizeInBytes => DType::Primitive(PType::U64, NonNullable),
            Self::NaNCount => {
                return aggregate_fn::fns::nan_count::NanCount
                    .return_dtype(&EmptyOptions, data_type);
            }
            Self::Sum => {
                return aggregate_fn::fns::sum::Sum.return_dtype(&EmptyOptions, data_type);
            }
        })
    }

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

    pub fn all() -> impl Iterator<Item = Stat> {
        all::<Self>()
    }
}

impl Display for Stat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod test {
    use enum_iterator::all;

    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::expr::stats::Stat;

    #[test]
    fn min_of_nulls_is_not_panic() {
        let min = PrimitiveArray::from_option_iter::<i32, _>([None, None, None, None])
            .statistics()
            .compute_as::<i64>(Stat::Min, &mut LEGACY_SESSION.create_execution_ctx());

        assert_eq!(min, None);
    }

    #[test]
    fn has_same_dtype_as_array() {
        assert!(Stat::Min.has_same_dtype_as_array());
        assert!(Stat::Max.has_same_dtype_as_array());
        for stat in all::<Stat>().filter(|s| !matches!(s, Stat::Min | Stat::Max)) {
            assert!(!stat.has_same_dtype_as_array());
        }
    }
}
