//! Traits and utilities to compute and access array statistics.

use std::fmt::{Display, Formatter};
use std::hash::Hash;

use enum_iterator::Sequence;
use enum_map::Enum;
use itertools::Itertools;
pub use statsset::*;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, NativePType};
use vortex_error::{vortex_err, vortex_panic, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::ArrayData;

pub mod flatbuffers;
mod statsset;

/// Statistics that are used for pruning files (i.e., we want to ensure they are computed when compressing/writing).
pub const PRUNING_STATS: &[Stat] = &[Stat::Min, Stat::Max, Stat::TrueCount, Stat::NullCount];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Sequence, Enum)]
#[non_exhaustive]
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
}

impl Display for Stat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BitWidthFreq => write!(f, "bit_width_frequency"),
            Self::TrailingZeroFreq => write!(f, "trailing_zero_frequency"),
            Self::IsConstant => write!(f, "is_constant"),
            Self::IsSorted => write!(f, "is_sorted"),
            Self::IsStrictSorted => write!(f, "is_strict_sorted"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
            Self::RunCount => write!(f, "run_count"),
            Self::TrueCount => write!(f, "true_count"),
            Self::NullCount => write!(f, "null_count"),
            Self::UncompressedSizeInBytes => write!(f, "uncompressed_size_in_bytes"),
        }
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

    /// Computes the value of the stat if it's not present
    fn compute(&self, stat: Stat) -> Option<Scalar>;

    /// Compute all the requested statistics (if not already present)
    /// Returns a StatsSet with the requested stats and any additional available stats
    fn compute_all(&self, stats: &[Stat]) -> VortexResult<StatsSet> {
        let mut stats_set = self.to_set();
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
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
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

    pub fn compute_min<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(&self) -> Option<U> {
        self.compute_as(Stat::Min)
    }

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
        .unwrap_or(0) as u8
}

#[cfg(test)]
mod test {
    use enum_iterator::all;

    use crate::array::PrimitiveArray;
    use crate::stats::{ArrayStatistics, Stat};

    #[test]
    fn min_of_nulls_is_not_panic() {
        let min = PrimitiveArray::from_nullable_vec::<i32>(vec![None, None, None, None])
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
