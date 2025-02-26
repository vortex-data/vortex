//! Traits and utilities to compute and access array statistics.

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use arrow_buffer::bit_iterator::BitIterator;
use arrow_buffer::{BooleanBufferBuilder, MutableBuffer};
use enum_iterator::{Sequence, cardinality};
use itertools::Itertools;
use log::debug;
use num_enum::{IntoPrimitive, TryFromPrimitive};
pub use stats_set::*;
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_panic};
use vortex_scalar::{Scalar, ScalarValue};

use crate::Array;

mod bound;
pub mod flatbuffers;
mod precision;
mod stat_bound;
mod stats_set;

pub use bound::{LowerBound, UpperBound};
pub use precision::Precision;
pub use stat_bound::*;

/// Statistics that are used for pruning files (i.e., we want to ensure they are computed when compressing/writing).
/// Sum is included for boolean arrays.
pub const PRUNING_STATS: &[Stat] = &[Stat::Min, Stat::Max, Stat::Sum, Stat::NullCount];

/// Stats to keep when serializing arrays to layouts
pub const STATS_TO_WRITE: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::NullCount,
    Stat::Sum,
    Stat::IsConstant,
    Stat::IsSorted,
    Stat::IsStrictSorted,
    Stat::UncompressedSizeInBytes,
];

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
    /// Frequency of each bit width (nulls are treated as 0)
    BitWidthFreq = 0,
    /// Frequency of each trailing zero (nulls are treated as 0)
    TrailingZeroFreq = 1,
    /// Whether all values are the same (nulls are not equal to other non-null values,
    /// so this is true iff all values are null or all values are the same non-null value)
    IsConstant = 2,
    /// Whether the non-null values in the array are sorted (i.e., we skip nulls)
    IsSorted = 3,
    /// Whether the non-null values in the array are strictly sorted (i.e., sorted with no duplicates)
    IsStrictSorted = 4,
    /// The maximum value in the array (ignoring nulls, unless all values are null)
    Max = 5,
    /// The minimum value in the array (ignoring nulls, unless all values are null)
    Min = 6,
    /// The sum of the non-null values of the array.
    Sum = 8,
    /// The number of null values in the array
    NullCount = 9,
    /// The uncompressed size of the array in bytes
    UncompressedSizeInBytes = 10,
}

/// These structs allow the extraction of the bound from the `Precision` value.
/// They tie together the Stat and the StatBound, which allows the bound to be extracted.
pub struct Max;
pub struct Min;
pub struct Sum;
pub struct BitWidthFreq;
pub struct TrailingZeroFreq;
pub struct IsConstant;
pub struct IsSorted;
pub struct IsStrictSorted;
pub struct NullCount;
pub struct UncompressedSizeInBytes;

impl<T: PartialOrd + Clone> StatType<T> for BitWidthFreq {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::BitWidthFreq;
}

impl<T: PartialOrd + Clone> StatType<T> for TrailingZeroFreq {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::TrailingZeroFreq;
}

impl StatType<bool> for IsConstant {
    type Bound = Precision<bool>;

    const STAT: Stat = Stat::IsConstant;
}

impl<T: PartialOrd + Clone> StatType<T> for IsSorted {
    type Bound = Precision<T>;

    const STAT: Stat = Stat::IsSorted;
}

impl<T: PartialOrd + Clone> StatType<T> for IsStrictSorted {
    type Bound = Precision<T>;

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

impl Stat {
    /// Whether the statistic is commutative (i.e., whether merging can be done independently of ordering)
    /// e.g., min/max are commutative, but is_sorted is not
    pub fn is_commutative(&self) -> bool {
        // NOTE: we prefer this syntax to force a compile error if we add a new stat
        match self {
            Stat::BitWidthFreq
            | Stat::TrailingZeroFreq
            | Stat::IsConstant
            | Stat::Max
            | Stat::Min
            | Stat::NullCount
            | Stat::Sum
            | Stat::UncompressedSizeInBytes => true,
            Stat::IsSorted | Stat::IsStrictSorted => false,
        }
    }

    /// Whether the statistic has the same dtype as the array it's computed on
    pub fn has_same_dtype_as_array(&self) -> bool {
        matches!(self, Stat::Min | Stat::Max)
    }

    pub fn dtype(&self, data_type: &DType) -> Option<DType> {
        Some(match self {
            Stat::BitWidthFreq => DType::List(
                Arc::new(DType::Primitive(PType::U64, NonNullable)),
                NonNullable,
            ),
            Stat::TrailingZeroFreq => DType::List(
                Arc::new(DType::Primitive(PType::U64, NonNullable)),
                NonNullable,
            ),
            Stat::IsConstant => DType::Bool(NonNullable),
            Stat::IsSorted => DType::Bool(NonNullable),
            Stat::IsStrictSorted => DType::Bool(NonNullable),
            Stat::Max => data_type.clone(),
            Stat::Min => data_type.clone(),
            Stat::NullCount => DType::Primitive(PType::U64, NonNullable),
            Stat::UncompressedSizeInBytes => DType::Primitive(PType::U64, NonNullable),
            Stat::Sum => {
                // Any array that cannot be summed has a sum DType of null.
                // Any array that can be summed, but overflows, has a sum _value_ of null.
                // Therefore, we make integer sum stats nullable.
                match data_type {
                    DType::Bool(_) => DType::Primitive(PType::U64, Nullable),
                    DType::Primitive(ptype, _) => match ptype {
                        PType::U8 | PType::U16 | PType::U32 | PType::U64 => {
                            DType::Primitive(PType::U64, Nullable)
                        }
                        PType::I8 | PType::I16 | PType::I32 | PType::I64 => {
                            DType::Primitive(PType::I64, Nullable)
                        }
                        PType::F16 | PType::F32 | PType::F64 => {
                            // Float sums cannot overflow, so it's non-nullable
                            DType::Primitive(PType::F64, NonNullable)
                        }
                    },
                    DType::Extension(ext_dtype) => self.dtype(ext_dtype.storage_dtype())?,
                    // Unsupported types
                    DType::Null
                    | DType::Utf8(_)
                    | DType::Binary(_)
                    | DType::Struct(..)
                    | DType::List(..) => return None,
                }
            }
        })
    }

    pub fn name(&self) -> &str {
        match self {
            Self::BitWidthFreq => "bit_width_frequency",
            Self::TrailingZeroFreq => "trailing_zero_frequency",
            Self::IsConstant => "is_constant",
            Self::IsSorted => "is_sorted",
            Self::IsStrictSorted => "is_strict_sorted",
            Self::Max => "max",
            Self::Min => "min",
            Self::NullCount => "null_count",
            Self::UncompressedSizeInBytes => "uncompressed_size_in_bytes",
            Stat::Sum => "sum",
        }
    }
}

pub fn as_stat_bitset_bytes(stats: &[Stat]) -> Vec<u8> {
    let stat_count = cardinality::<Stat>();
    let mut stat_bitset = BooleanBufferBuilder::new_from_buffer(
        MutableBuffer::from_len_zeroed(stat_count.div_ceil(8)),
        stat_count,
    );
    for stat in stats {
        stat_bitset.set_bit(u8::from(*stat) as usize, true);
    }

    stat_bitset
        .finish()
        .into_inner()
        .into_vec()
        .unwrap_or_else(|b| b.to_vec())
}

pub fn stats_from_bitset_bytes(bytes: &[u8]) -> Vec<Stat> {
    BitIterator::new(bytes, 0, bytes.len() * 8)
        .enumerate()
        .filter_map(|(i, b)| b.then_some(i))
        // Filter out indices failing conversion, these are stats written by newer version of library
        .filter_map(|i| {
            let Ok(stat) = u8::try_from(i) else {
                debug!("invalid stat encountered: {i}");
                return None;
            };
            Stat::try_from(stat).ok()
        })
        .collect::<Vec<_>>()
}

impl Display for Stat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

pub trait Statistics {
    /// Returns the value of the statistic only if it's present
    fn get_stat(&self, stat: Stat) -> Option<Precision<ScalarValue>>;

    /// Get all existing statistics
    fn stats_set(&self) -> StatsSet;

    /// Set the value of the statistic
    fn set_stat(&self, stat: Stat, value: Precision<ScalarValue>);

    /// Clear the value of the statistic
    fn clear_stat(&self, stat: Stat);

    /// Computes the value of the stat if it's not present and inexact.
    ///
    /// Returns the scalar if compute succeeded, or `None` if the stat is not supported
    /// for this array.
    fn compute_stat(&self, stat: Stat) -> VortexResult<Option<ScalarValue>>;

    /// Compute all the requested statistics (if not already present)
    /// Returns a StatsSet with the requested stats and any additional available stats
    // [deprecated]
    // TODO(joe): replace with `compute_statistics`
    fn compute_all(&self, stats: &[Stat]) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        for stat in stats {
            if let Some(s) = self.compute_stat(*stat)? {
                stats_set.set(*stat, Precision::exact(s))
            }
        }
        Ok(stats_set)
    }

    fn retain_only(&self, stats: &[Stat]);

    fn inherit(&self, parent: &dyn Statistics) {
        let parent_stats_set = parent.stats_set();
        for (stat, value) in parent_stats_set.into_iter() {
            // TODO(ngates): we may need a set_all(&[(Stat, Precision<ScalarValue>)]) method
            //  so we don't have to take lots of write locks.
            // TODO(ngates): depending on statistic, this should choose the more precise one.
            self.set_stat(stat, value);
        }
    }
}

impl dyn Statistics + '_ {
    /// Get the provided stat if present in the underlying array, converting the `ScalarValue` into a typed value.
    /// If the stored `ScalarValue` is of different type then the primitive typed value this function will perform a cast.
    ///
    /// # Panics
    ///
    /// This function will panic if the conversion fails.
    pub fn get_as<U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<Precision<U>> {
        self.get_stat(stat)
            .map(|s| s.try_map(|s| U::try_from(&s)))
            .transpose()
            .unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to cast stat {} to {}",
                    stat,
                    std::any::type_name::<U>()
                )
            })
    }

    pub fn get_as_bound<S, U>(&self) -> Option<S::Bound>
    where
        S: StatType<U>,
        U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>,
    {
        self.get_as::<U>(S::STAT).map(|v| v.bound::<S>())
    }

    pub fn get_scalar(&self, stat: Stat, dtype: &DType) -> Option<Precision<Scalar>> {
        self.get_stat(stat).map(|s| s.into_scalar(dtype.clone()))
    }

    /// Get or calculate the provided stat, converting the `ScalarValue` into a typed value.
    /// If the stored `ScalarValue` is of different type then the primitive typed value this function will perform a cast.
    ///
    /// # Panics
    ///
    /// This function will panic if the conversion fails.
    pub fn compute_as<U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<U> {
        self.compute_stat(stat)
            .inspect_err(|e| log::warn!("Failed to compute stat {}: {}", stat, e))
            .ok()
            .flatten()
            .map(|s| U::try_from(&s))
            .transpose()
            .unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to compute stat {} as {}",
                    stat,
                    std::any::type_name::<U>()
                )
            })
    }

    /// Get or calculate the minimum value in the array, returning as a typed value.
    ///
    /// This function will panic if the conversion fails.
    pub fn compute_min<U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
    ) -> Option<U> {
        self.compute_as(Stat::Min)
    }

    /// Get or calculate the maximum value in the array, returning as a typed value.
    ///
    /// This function will panic if the conversion fails.
    pub fn compute_max<U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
    ) -> Option<U> {
        self.compute_as(Stat::Max)
    }

    pub fn compute_is_strict_sorted(&self) -> Option<bool> {
        self.compute_as(Stat::IsStrictSorted)
    }

    pub fn compute_is_sorted(&self) -> Option<bool> {
        self.compute_as(Stat::IsSorted)
    }

    pub fn compute_is_constant(&self) -> Option<bool> {
        self.compute_as(Stat::IsConstant)
    }

    pub fn compute_null_count(&self) -> Option<usize> {
        self.compute_as(Stat::NullCount)
    }

    pub fn compute_bit_width_freq(&self) -> Option<Vec<usize>> {
        self.compute_as::<Vec<usize>>(Stat::BitWidthFreq)
    }

    pub fn compute_trailing_zero_freq(&self) -> Option<Vec<usize>> {
        self.compute_as::<Vec<usize>>(Stat::TrailingZeroFreq)
    }

    pub fn compute_uncompressed_size_in_bytes(&self) -> Option<usize> {
        self.compute_as(Stat::UncompressedSizeInBytes)
    }
}

pub fn trailing_zeros(array: &dyn Array) -> u8 {
    let tz_freq = array
        .statistics()
        .compute_trailing_zero_freq()
        .unwrap_or_else(|| vec![0]);
    tz_freq
        .iter()
        .enumerate()
        .find_or_first(|&(_, &v)| v > 0)
        .map(|(i, _)| i)
        .unwrap_or(0)
        .try_into()
        .vortex_expect("tz_freq must fit in u8")
}

#[cfg(test)]
mod test {
    use enum_iterator::all;

    use crate::array::Array;
    use crate::arrays::PrimitiveArray;
    use crate::stats::Stat;

    #[test]
    fn min_of_nulls_is_not_panic() {
        let min = PrimitiveArray::from_option_iter::<i32, _>([None, None, None, None])
            .statistics()
            .compute_as::<i64>(Stat::Min);

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
