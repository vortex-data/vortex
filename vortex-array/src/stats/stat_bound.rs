use std::cmp::Ordering;

use crate::stats::bound::{max, min, JoinResult};
use crate::stats::{LowerBound, Precision, Stat, UpperBound};

/// `StatType` define the bound of a given statistic. (e.g. `Max` is an upper bound),
/// this is used to extract the bound from a `Precision` value, (e.g. p::bound<Max>()).
pub trait StatType<T> {
    type Bound: StatBound<T>;

    const STAT: Stat;
}

pub trait StatBound<T>: Sized {
    fn lift(value: Precision<T>) -> Self;

    fn union(&self, other: &Self) -> Option<Self>;

    fn intersection(&self, other: &Self) -> Option<JoinResult<Self>>;
}

/// This allows a stat with a `Precision` to be interpreted as a bound.
impl<T> Precision<T> {
    pub fn bound<S: StatType<T>>(self) -> S::Bound {
        S::Bound::lift(self)
    }
}

pub struct Max;
pub struct Min;
pub struct BitWidthFreq;
pub struct TrailingZeroFreq;
pub struct IsConstant;
pub struct IsSorted;
pub struct IsStrictSorted;
pub struct RunCount;
pub struct TrueCount;
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

impl<T: PartialOrd + Clone> StatType<T> for IsConstant {
    type Bound = Precision<T>;

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

impl<T: PartialOrd + Clone> StatType<T> for RunCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::RunCount;
}

impl<T: PartialOrd + Clone> StatType<T> for TrueCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::TrueCount;
}

impl<T: PartialOrd + Clone> StatType<T> for NullCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::NullCount;
}

impl<T: PartialOrd + Clone> StatType<T> for UncompressedSizeInBytes {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::UncompressedSizeInBytes;
}

impl<T: PartialOrd + Clone> StatType<T> for Max {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::Max;
}

impl<T: PartialOrd + Clone> StatType<T> for Min {
    type Bound = LowerBound<T>;

    const STAT: Stat = Stat::Min;
}

impl<T: PartialOrd + Clone> LowerBound<T> {
    pub fn into_value(self) -> Precision<T> {
        self.0
    }
}

impl<T: PartialOrd + Clone> StatBound<T> for Precision<T> {
    fn lift(value: Precision<T>) -> Self {
        value
    }

    fn union(&self, other: &Self) -> Option<Self> {
        self.clone()
            .zip(other.clone())
            .map(|(lhs, rhs)| min(lhs, rhs))
            .transpose()
    }

    fn intersection(&self, other: &Self) -> Option<JoinResult<Self>> {
        Some(match (self, other) {
            (Precision::Exact(lhs), Precision::Exact(rhs)) => {
                if lhs.partial_cmp(rhs)? == Ordering::Equal {
                    JoinResult::Join(Precision::Exact(lhs.clone()))
                } else {
                    JoinResult::None
                }
            }
            (Precision::Exact(exact), Precision::Inexact(_))
            | (Precision::Inexact(_), Precision::Exact(exact)) => {
                JoinResult::Join(Precision::Inexact(exact.clone()))
            }
            (Precision::Inexact(lhs), Precision::Inexact(rhs)) => {
                JoinResult::Join(Precision::Inexact(max(lhs, rhs)?.clone()))
            }
        })
    }
}
