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

mod bound;
mod precision;
mod provider;
mod stat_bound;

pub use bound::*;
pub use precision::*;
pub use provider::*;
pub use stat_bound::*;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::aggregate_fn;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::aggregate_fn::stat_aggregates;
use crate::aggregate_fn::stat_aggregates::IsSortedOptions;
use crate::aggregate_fn::stat_aggregates::UncompressedSizeOpts;
use crate::dtype::PType;

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
            Self::UncompressedSizeInBytes => {
                return stat_aggregates::supports_uncompressed_size_in_bytes(data_type)
                    .then_some(DType::Primitive(PType::U64, NonNullable));
            }
            Self::NaNCount => {
                return matches!(data_type, DType::Primitive(ptype, _) if ptype.is_float())
                    .then_some(DType::Primitive(PType::U64, NonNullable));
            }
            Self::Sum => return stat_aggregates::sum_return_dtype(data_type),
        })
    }

    /// The ID of the aggregate function that computes this statistic.
    pub fn aggregate_fn_id(&self) -> AggregateFnId {
        **match self {
            Self::IsConstant => &stat_aggregates::IS_CONSTANT_ID,
            Self::IsSorted | Self::IsStrictSorted => &stat_aggregates::IS_SORTED_ID,
            Self::Max => &stat_aggregates::MAX_ID,
            Self::Min => &stat_aggregates::MIN_ID,
            Self::NullCount => &stat_aggregates::NULL_COUNT_ID,
            Self::UncompressedSizeInBytes => &stat_aggregates::UNCOMPRESSED_SIZE_IN_BYTES_ID,
            Self::Sum => &stat_aggregates::SUM_ID,
            Self::NaNCount => &stat_aggregates::NAN_COUNT_ID,
        }
    }

    /// The serialized options the aggregate function computing this statistic is bound to.
    ///
    /// Min/max/sum statistics skip NaN values, and the uncompressed size statistic is exact.
    pub fn aggregate_fn_options(&self) -> Vec<u8> {
        match self {
            Self::Max | Self::Min | Self::Sum => NumericalAggregateOpts::skip_nans().serialize(),
            Self::UncompressedSizeInBytes => UncompressedSizeOpts::exact().serialize(),
            Self::IsSorted => IsSortedOptions { strict: false }.serialize(),
            Self::IsStrictSorted => IsSortedOptions { strict: true }.serialize(),
            Self::IsConstant | Self::NullCount | Self::NaNCount => vec![],
        }
    }

    /// Resolve the aggregate function computing this statistic from the session's registry.
    ///
    /// Returns `None` if no aggregate function with this statistic's ID is registered.
    pub fn resolve_aggregate_fn(
        &self,
        session: &VortexSession,
    ) -> VortexResult<Option<AggregateFnRef>> {
        let Some(plugin) = session.aggregate_fns().find_plugin(&self.aggregate_fn_id()) else {
            return Ok(None);
        };
        plugin
            .deserialize(&self.aggregate_fn_options(), session)
            .map(Some)
    }

    /// Return the built-in aggregate function corresponding to this statistic, if one exists.
    pub fn aggregate_fn(&self) -> Option<AggregateFnRef> {
        // Statistics follow NaN-skipping semantics; request it explicitly rather than the default.
        Some(match self {
            Self::Max => aggregate_fn::fns::max::Max.bind(NumericalAggregateOpts::skip_nans()),
            Self::Min => aggregate_fn::fns::min::Min.bind(NumericalAggregateOpts::skip_nans()),
            Self::Sum => aggregate_fn::fns::sum::Sum.bind(NumericalAggregateOpts::skip_nans()),
            Self::NullCount => aggregate_fn::fns::null_count::NullCount.bind(EmptyOptions),
            Self::NaNCount => aggregate_fn::fns::nan_count::NanCount.bind(EmptyOptions),
            // The stat slot holds the exact size, so the statistic measures referenced bytes.
            Self::UncompressedSizeInBytes => {
                aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes
                    .bind(UncompressedSizeOpts::exact())
            }
            Self::IsConstant | Self::IsSorted | Self::IsStrictSorted => return None,
        })
    }

    /// Return the statistic represented by `aggregate_fn`, if it has a legacy stat slot.
    ///
    /// The aggregate must have this statistic's ID and options. Min/max/sum statistics skip NaN
    /// values, so NaN-including configurations of those aggregates have no stat slot. The
    /// uncompressed size statistic is exact, so the inexact configuration of that aggregate has
    /// no stat slot either. Sortedness and constancy have no stat slot.
    pub fn from_aggregate_fn(aggregate_fn: &AggregateFnRef) -> Option<Self> {
        let id = aggregate_fn.id();
        let stat = [
            Self::Min,
            Self::Max,
            Self::Sum,
            Self::NullCount,
            Self::NaNCount,
            Self::UncompressedSizeInBytes,
        ]
        .into_iter()
        .find(|stat| stat.aggregate_fn_id() == id)?;
        let options = aggregate_fn.options().serialize().ok()??;
        (options == stat.aggregate_fn_options()).then_some(stat)
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
    use std::sync::LazyLock;

    use enum_iterator::all;
    use vortex_session::VortexSession;

    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::expr::stats::Stat;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn min_of_nulls_is_not_panic() {
        let min = PrimitiveArray::from_option_iter::<i32, _>([None, None, None, None])
            .statistics()
            .compute_as::<i64>(Stat::Min, &mut SESSION.create_execution_ctx());

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
