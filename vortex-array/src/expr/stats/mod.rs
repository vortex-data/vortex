// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::LazyLock;

use enum_iterator::Sequence;
use enum_iterator::all;
use num_enum::IntoPrimitive;
use num_enum::TryFromPrimitive;

use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;

mod bound;
mod precision;
mod provider;
mod stat_bound;

pub use bound::*;
pub use precision::*;
pub use provider::*;
pub use stat_bound::*;

use crate::aggregate_fn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;

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

/// The aggregate function of each [`Stat`], indexed by its discriminant.
static STAT_AGGREGATE_FNS: LazyLock<[AggregateFnRef; 9]> = LazyLock::new(|| {
    // Statistics follow NaN-skipping semantics; request it explicitly rather than the default.
    let is_sorted = |strict| {
        aggregate_fn::fns::is_sorted::IsSorted
            .bind(aggregate_fn::fns::is_sorted::IsSortedOptions { strict })
    };
    [
        aggregate_fn::fns::is_constant::IsConstant.bind(EmptyOptions),
        is_sorted(false),
        is_sorted(true),
        aggregate_fn::fns::max::Max.bind(NumericalAggregateOpts::skip_nans()),
        aggregate_fn::fns::min::Min.bind(NumericalAggregateOpts::skip_nans()),
        aggregate_fn::fns::sum::Sum.bind(NumericalAggregateOpts::skip_nans()),
        aggregate_fn::fns::null_count::NullCount.bind(EmptyOptions),
        aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes.bind(EmptyOptions),
        aggregate_fn::fns::nan_count::NanCount.bind(EmptyOptions),
    ]
});

impl Stat {
    /// Whether the statistic is stored in zone maps and used for pruning.
    ///
    /// `IsConstant` and `IsSorted` are array stats only: their stored results cannot be
    /// combined across zones, so zone maps and predicate rewrites never reference them.
    pub fn is_zone_stat(&self) -> bool {
        !matches!(
            self,
            Self::IsConstant | Self::IsSorted | Self::IsStrictSorted
        )
    }

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
                // Statistics follow NaN-skipping semantics; request it explicitly.
                return aggregate_fn::fns::sum::Sum
                    .return_dtype(&NumericalAggregateOpts::skip_nans(), data_type);
            }
        })
    }

    /// Return the built-in aggregate function this statistic stores the result of.
    ///
    /// Returns a static reference, so hot stat lookups never touch a shared reference count.
    pub fn aggregate_fn(&self) -> &'static AggregateFnRef {
        &STAT_AGGREGATE_FNS[usize::from(u8::from(*self))]
    }

    /// Return the statistic whose static key is `aggregate_fn`, comparing pointers only.
    ///
    /// Most callers pass the keys from [`Self::aggregate_fn`], so this avoids downcasting.
    pub(crate) fn static_from_aggregate_fn(aggregate_fn: &AggregateFnRef) -> Option<Self> {
        let index = STAT_AGGREGATE_FNS
            .iter()
            .position(|key| key.ptr_eq(aggregate_fn))?;
        u8::try_from(index)
            .ok()
            .and_then(|i| Self::try_from(i).ok())
    }

    /// Return the statistic represented by `aggregate_fn`, if it has a legacy stat slot.
    ///
    /// Min/max/sum statistics skip NaN values, so NaN-including configurations of those
    /// aggregates have no stat slot.
    pub fn from_aggregate_fn(aggregate_fn: &AggregateFnRef) -> Option<Self> {
        if let Some(stat) = Self::static_from_aggregate_fn(aggregate_fn) {
            return Some(stat);
        }

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
        if aggregate_fn.is::<aggregate_fn::fns::is_constant::IsConstant>() {
            return Some(Self::IsConstant);
        }
        if let Some(options) = aggregate_fn.as_opt::<aggregate_fn::fns::is_sorted::IsSorted>() {
            return Some(if options.strict {
                Self::IsStrictSorted
            } else {
                Self::IsSorted
            });
        }
        None
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

    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::expr::stats::Stat;

    #[test]
    fn min_of_nulls_is_not_panic() {
        let min = PrimitiveArray::from_option_iter::<i32, _>([None, None, None, None])
            .statistics()
            .get_as::<i64>(
                Stat::Min.aggregate_fn(),
                &mut array_session().create_execution_ctx(),
            );

        assert_eq!(min, None);
    }

    #[test]
    fn aggregate_fn_round_trips() {
        for stat in all::<Stat>() {
            assert_eq!(Stat::from_aggregate_fn(stat.aggregate_fn()), Some(stat));
        }
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
