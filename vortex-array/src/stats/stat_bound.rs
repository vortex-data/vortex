use crate::stats::{LowerBound, Precision, Stat, UpperBound};

/// `StatType` define the bound of a given statistic. (e.g. `Max` is an upper bound)
pub trait StatType<T> {
    type Bound: StatBound<T>;

    const STAT: Stat;
}

pub trait StatBound<T> {
    fn lift(value: Precision<T>) -> Self;
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

impl<T> StatType<T> for BitWidthFreq {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::BitWidthFreq;
}

impl<T> StatType<T> for TrailingZeroFreq {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::TrailingZeroFreq;
}

impl<T> StatType<T> for IsConstant {
    type Bound = Precision<T>;

    const STAT: Stat = Stat::IsConstant;
}

impl<T> StatType<T> for IsSorted {
    type Bound = Precision<T>;

    const STAT: Stat = Stat::IsSorted;
}

impl<T> StatType<T> for IsStrictSorted {
    type Bound = Precision<T>;

    const STAT: Stat = Stat::IsStrictSorted;
}

impl<T> StatType<T> for RunCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::RunCount;
}

impl<T> StatType<T> for TrueCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::TrueCount;
}

impl<T> StatType<T> for NullCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::NullCount;
}

impl<T> StatType<T> for UncompressedSizeInBytes {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::UncompressedSizeInBytes;
}

impl<T> StatType<T> for Max {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::Max;
}

impl<T> StatType<T> for Min {
    type Bound = LowerBound<T>;

    const STAT: Stat = Stat::Min;
}

impl<T> LowerBound<T> {
    pub fn into_value(self) -> Precision<T> {
        self.0
    }
}

impl<T> StatBound<T> for LowerBound<T> {
    fn lift(value: Precision<T>) -> Self {
        LowerBound(value)
    }
}

impl<T> StatBound<T> for UpperBound<T> {
    fn lift(value: Precision<T>) -> Self {
        UpperBound(value)
    }
}

impl<T> StatBound<T> for Precision<T> {
    fn lift(value: Precision<T>) -> Self {
        value
    }
}
