//! Traits and utilities to compute and access array statistics.

use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use arrow_buffer::bit_iterator::BitIterator;
use arrow_buffer::{BooleanBufferBuilder, MutableBuffer};
use enum_iterator::{cardinality, Sequence};
use itertools::Itertools;
use log::debug;
use num_enum::{IntoPrimitive, TryFromPrimitive};
pub use statsset::*;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, NativePType, PType};
use vortex_error::{vortex_panic, VortexError, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::ArrayData;

pub mod flatbuffers;
mod statsset;

/// Statistics that are used for pruning files (i.e., we want to ensure they are computed when compressing/writing).
pub const PRUNING_STATS: &[Stat] = &[Stat::Min, Stat::Max, Stat::TrueCount, Stat::NullCount];

/// Stats to keep when serializing arrays to layouts
pub const STATS_TO_WRITE: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::TrueCount,
    Stat::NullCount,
    Stat::RunCount,
    Stat::IsConstant,
    Stat::IsSorted,
    Stat::IsStrictSorted,
    Stat::UncompressedSizeInBytes,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Sequence, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Stat {
    /// Frequency of each bit width (nulls are treated as 0)
    BitWidthFreq,
    /// Frequency of each trailing zero (nulls are treated as 0)
    TrailingZeroFreq,
    /// Whether all values are the same (nulls are not equal to other non-null values,
    /// so this is true iff all values are null or all values are the same non-null value)
    IsConstant,
    /// Whether the non-null values in the array are sorted (i.e., we skip nulls)
    IsSorted,
    /// Whether the non-null values in the array are strictly sorted (i.e., sorted with no duplicates)
    IsStrictSorted,
    /// The maximum value in the array (ignoring nulls, unless all values are null)
    Max,
    /// The minimum value in the array (ignoring nulls, unless all values are null)
    Min,
    /// The number of runs in the array (ignoring nulls)
    RunCount,
    /// The number of true values in the array (nulls are treated as false)
    TrueCount,
    /// The number of null values in the array
    NullCount,
    /// The uncompressed size of the array in bytes
    UncompressedSizeInBytes,
}

impl Stat {
    /// Whether the statistic is commutative (i.e., whether merging can be done independently of ordering)
    /// e.g., min/max are commutative, but is_sorted is not
    pub fn is_commutative(&self) -> bool {
        matches!(
            self,
            Stat::BitWidthFreq
                | Stat::TrailingZeroFreq
                | Stat::IsConstant
                | Stat::Max
                | Stat::Min
                | Stat::TrueCount
                | Stat::NullCount
                | Stat::UncompressedSizeInBytes
        )
    }

    /// Whether the statistic has the same dtype as the array it's computed on
    pub fn has_same_dtype_as_array(&self) -> bool {
        matches!(self, Stat::Min | Stat::Max)
    }

    pub fn dtype(&self, data_type: &DType) -> DType {
        match self {
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
            Stat::RunCount => DType::Primitive(PType::U64, NonNullable),
            Stat::TrueCount => DType::Primitive(PType::U64, NonNullable),
            Stat::NullCount => DType::Primitive(PType::U64, NonNullable),
            Stat::UncompressedSizeInBytes => DType::Primitive(PType::U64, NonNullable),
        }
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
            Self::RunCount => "run_count",
            Self::TrueCount => "true_count",
            Self::NullCount => "null_count",
            Self::UncompressedSizeInBytes => "uncompressed_size_in_bytes",
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
    fn get(&self, stat: Stat) -> Option<Scalar>;

    /// Get all existing statistics
    fn to_set(&self) -> StatsSet;

    /// Set the value of the statistic
    fn set(&self, stat: Stat, value: Scalar);

    /// Clear the value of the statistic
    fn clear(&self, stat: Stat);

    /// Computes the value of the stat if it's not present.
    ///
    /// Returns the scalar if compute succeeded, or `None` if the stat is not supported
    /// for this array.
    fn compute(&self, stat: Stat) -> Option<Scalar>;

    /// Compute all the requested statistics (if not already present)
    /// Returns a StatsSet with the requested stats and any additional available stats
    fn compute_all(&self, stats: &[Stat]) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        for stat in stats {
            if let Some(s) = self.compute(*stat) {
                stats_set.set(*stat, s)
            }
        }
        Ok(stats_set)
    }

    fn retain_only(&self, stats: &[Stat]);
}

pub trait ArrayStatistics {
    fn statistics(&self) -> &dyn Statistics;

    fn inherit_statistics(&self, parent: &dyn Statistics);
}

/// Encoding VTable for computing array statistics.
pub trait StatisticsVTable<Array: ?Sized> {
    /// Compute the requested statistic. Can return additional stats.
    fn compute_statistics(&self, _array: &Array, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

impl<E: Encoding + 'static> StatisticsVTable<ArrayData> for E
where
    E: StatisticsVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn compute_statistics(&self, array: &ArrayData, stat: Stat) -> VortexResult<StatsSet> {
        let (array_ref, encoding) = array.downcast_array_ref::<E>()?;
        StatisticsVTable::compute_statistics(encoding, array_ref, stat)
    }
}

impl dyn Statistics + '_ {
    pub fn get_as<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<U> {
        self.get(stat)
            .map(|s| U::try_from(&s))
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

    pub fn get_as_cast<U: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<U> {
        self.get(stat)
            .filter(|s| s.is_valid())
            .map(|s| s.cast(&DType::Primitive(U::PTYPE, NonNullable)))
            .transpose()
            .and_then(|maybe| maybe.as_ref().map(U::try_from).transpose())
            .unwrap_or_else(|err| {
                vortex_panic!(err, "Failed to cast stat {} to {}", stat, U::PTYPE)
            })
    }

    /// Get or calculate the provided stat, converting the `Scalar` into a typed value.
    ///
    /// This function will panic if the conversion fails.
    pub fn compute_as<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<U> {
        self.compute(stat)
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

    pub fn compute_as_cast<U: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<U> {
        self.compute(stat)
            .filter(|s| s.is_valid())
            .map(|s| s.cast(&DType::Primitive(U::PTYPE, NonNullable)))
            .transpose()
            .and_then(|maybe| maybe.as_ref().map(U::try_from).transpose())
            .unwrap_or_else(|err| {
                vortex_panic!(err, "Failed to compute stat {} as cast {}", stat, U::PTYPE)
            })
    }

    /// Get or calculate the minimum value in the array, returning as a typed value.
    ///
    /// This function will panic if the conversion fails.
    pub fn compute_min<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(&self) -> Option<U> {
        self.compute_as(Stat::Min)
    }

    /// Get or calculate the maximum value in the array, returning as a typed value.
    ///
    /// This function will panic if the conversion fails.
    pub fn compute_max<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(&self) -> Option<U> {
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

    pub fn compute_true_count(&self) -> Option<usize> {
        self.compute_as(Stat::TrueCount)
    }

    pub fn compute_null_count(&self) -> Option<usize> {
        self.compute_as(Stat::NullCount)
    }

    pub fn compute_run_count(&self) -> Option<usize> {
        self.compute_as(Stat::RunCount)
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

pub fn trailing_zeros(array: &ArrayData) -> u8 {
    let tz_freq = array
        .statistics()
        .compute_trailing_zero_freq()
        .unwrap_or_else(|| vec![0]);
    tz_freq
        .iter()
        .enumerate()
        .find_or_first(|(_, &v)| v > 0)
        .map(|(i, _)| i)
        .unwrap_or(0)
        .try_into()
        .vortex_expect("tz_freq must fit in u8")
}

#[cfg(test)]
mod test {
    use enum_iterator::all;

    use crate::array::PrimitiveArray;
    use crate::stats::{ArrayStatistics, Stat};

    #[test]
    fn min_of_nulls_is_not_panic() {
        let min = PrimitiveArray::from_option_iter::<i32, _>([None, None, None, None])
            .statistics()
            .compute_as_cast::<i64>(Stat::Min);

        assert_eq!(min, None);
    }

    #[test]
    fn commutativity() {
        assert!(Stat::BitWidthFreq.is_commutative());
        assert!(Stat::TrailingZeroFreq.is_commutative());
        assert!(Stat::IsConstant.is_commutative());
        assert!(Stat::Min.is_commutative());
        assert!(Stat::Max.is_commutative());
        assert!(Stat::TrueCount.is_commutative());
        assert!(Stat::NullCount.is_commutative());

        assert!(!Stat::IsStrictSorted.is_commutative());
        assert!(!Stat::IsSorted.is_commutative());
        assert!(!Stat::RunCount.is_commutative());
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
