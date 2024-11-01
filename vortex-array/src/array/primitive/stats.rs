use std::cmp::Ordering;
use std::mem::size_of;

use arrow_buffer::buffer::BooleanBuffer;
use num_traits::PrimInt;
use vortex_dtype::half::f16;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::aliases::hash_map::HashMap;
use crate::array::primitive::PrimitiveArray;
use crate::stats::{ArrayStatisticsCompute, Stat, StatsSet};
use crate::validity::{ArrayValidity, LogicalValidity};
use crate::{ArrayDType, IntoArrayVariant};

trait PStatsType: NativePType + Into<Scalar> + BitWidth {}

impl<T: NativePType + Into<Scalar> + BitWidth> PStatsType for T {}

impl ArrayStatisticsCompute for PrimitiveArray {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        match_each_native_ptype!(self.ptype(), |$P| {
            match self.logical_validity() {
                LogicalValidity::AllValid(_) => self.maybe_null_slice::<$P>().compute_statistics(stat),
                LogicalValidity::AllInvalid(v) => Ok(StatsSet::nulls(v, self.dtype())),
                LogicalValidity::Array(a) => NullableValues(
                    self.maybe_null_slice::<$P>(),
                    &a.clone().into_bool()?.boolean_buffer(),
                )
                .compute_statistics(stat),
            }
        })
    }
}

impl<T: PStatsType> ArrayStatisticsCompute for &[T] {
    fn compute_statistics(&self, _stat: Stat) -> VortexResult<StatsSet> {
        if self.is_empty() {
            return Ok(StatsSet::new());
        }
        let mut stats = StatsAccumulator::new(self[0]);
        self.iter().skip(1).for_each(|next| stats.next(*next));
        Ok(stats.finish())
    }
}

struct NullableValues<'a, T: PStatsType>(&'a [T], &'a BooleanBuffer);

impl<T: PStatsType> ArrayStatisticsCompute for NullableValues<'_, T> {
    fn compute_statistics(&self, _stat: Stat) -> VortexResult<StatsSet> {
        let values = self.0;
        if values.is_empty() {
            return Ok(StatsSet::new());
        }

        let first_non_null_idx = self
            .1
            .iter()
            .enumerate()
            .skip_while(|(_, valid)| !*valid)
            .map(|(idx, _)| idx)
            .next();

        if let Some(first_non_null) = first_non_null_idx {
            let mut acc = StatsAccumulator::new(values[first_non_null]);
            acc.n_nulls(first_non_null);
            self.0
                .iter()
                .zip(self.1.iter())
                .skip(first_non_null + 1)
                .map(|(next, valid)| valid.then_some(*next))
                .for_each(|next| acc.nullable_next(next));
            Ok(acc.finish())
        } else {
            Ok(StatsSet::nulls(
                self.0.len(),
                &DType::Primitive(T::PTYPE, Nullability::Nullable),
            ))
        }
    }
}

trait BitWidth {
    fn bit_width(self) -> u32;
    fn trailing_zeros(self) -> u32;
}

macro_rules! int_bit_width {
    ($T:ty) => {
        impl BitWidth for $T {
            fn bit_width(self) -> u32 {
                Self::BITS - PrimInt::leading_zeros(self)
            }

            fn trailing_zeros(self) -> u32 {
                PrimInt::trailing_zeros(self)
            }
        }
    };
}

int_bit_width!(u8);
int_bit_width!(u16);
int_bit_width!(u32);
int_bit_width!(u64);
int_bit_width!(i8);
int_bit_width!(i16);
int_bit_width!(i32);
int_bit_width!(i64);

// TODO(ngates): just skip counting this in the implementation.
macro_rules! float_bit_width {
    ($T:ty) => {
        impl BitWidth for $T {
            fn bit_width(self) -> u32 {
                (size_of::<Self>() * 8) as u32
            }

            fn trailing_zeros(self) -> u32 {
                0
            }
        }
    };
}

float_bit_width!(f16);
float_bit_width!(f32);
float_bit_width!(f64);

struct StatsAccumulator<T: PStatsType> {
    prev: T,
    min: T,
    max: T,
    is_sorted: bool,
    is_strict_sorted: bool,
    run_count: usize,
    null_count: usize,
    nan_count: usize,
    bit_widths: Vec<usize>,
    trailing_zeros: Vec<usize>,
    len: usize,
}

impl<T: PStatsType> StatsAccumulator<T> {
    fn new(first_value: T) -> Self {
        let mut stats = Self {
            prev: first_value,
            min: first_value,
            max: first_value,
            is_sorted: true,
            is_strict_sorted: true,
            run_count: 1,
            null_count: 0,
            bit_widths: vec![0; size_of::<T>() * 8 + 1],
            trailing_zeros: vec![0; size_of::<T>() * 8 + 1],
            len: 1,
            nan_count: first_value.is_nan().then_some(1).unwrap_or_default(),
        };
        stats.bit_widths[first_value.bit_width() as usize] += 1;
        stats.trailing_zeros[first_value.trailing_zeros() as usize] += 1;
        stats
    }

    fn n_nulls(&mut self, n_nulls: usize) {
        self.null_count += n_nulls;
        self.bit_widths[0] += n_nulls;
        self.trailing_zeros[T::PTYPE.bit_width()] += n_nulls;
        self.len += n_nulls;
    }

    pub fn nullable_next(&mut self, next: Option<T>) {
        match next {
            Some(n) => self.next(n),
            None => {
                self.bit_widths[0] += 1;
                self.trailing_zeros[T::PTYPE.bit_width()] += 1;
                self.null_count += 1;
                self.len += 1;
            }
        }
    }

    pub fn next(&mut self, next: T) {
        self.bit_widths[next.bit_width() as usize] += 1;
        self.trailing_zeros[next.trailing_zeros() as usize] += 1;
        self.len += 1;

        if next.is_nan() {
            self.nan_count += 1;
        }

        if next.is_eq(self.prev) {
            self.is_strict_sorted = false;
        } else {
            if matches!(next.compare(self.prev), Ordering::Less) {
                self.is_sorted = false;
            }
            self.run_count += 1;
        }
        if matches!(next.compare(self.min), Ordering::Less) {
            self.min = next;
        } else if matches!(next.compare(self.max), Ordering::Greater) {
            self.max = next;
        }
        self.prev = next;
    }

    pub fn finish(self) -> StatsSet {
        let is_constant = (self.min == self.max && self.null_count == 0 && self.nan_count == 0)
            || self.null_count == self.len;

        StatsSet::from(HashMap::from([
            (Stat::Min, self.min.into()),
            (Stat::Max, self.max.into()),
            (Stat::NullCount, self.null_count.into()),
            (Stat::IsConstant, is_constant.into()),
            (Stat::BitWidthFreq, self.bit_widths.into()),
            (Stat::TrailingZeroFreq, self.trailing_zeros.into()),
            (Stat::IsSorted, self.is_sorted.into()),
            (
                Stat::IsStrictSorted,
                (self.is_sorted && self.is_strict_sorted).into(),
            ),
            (Stat::RunCount, self.run_count.into()),
        ]))
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::array::primitive::PrimitiveArray;
    use crate::stats::{ArrayStatistics, Stat};

    #[test]
    fn stats() {
        let arr = PrimitiveArray::from(vec![1, 2, 3, 4, 5]);
        let min: i32 = arr.statistics().compute_min().unwrap();
        let max: i32 = arr.statistics().compute_max().unwrap();
        let is_sorted = arr.statistics().compute_is_sorted().unwrap();
        let is_strict_sorted = arr.statistics().compute_is_strict_sorted().unwrap();
        let is_constant = arr.statistics().compute_is_constant().unwrap();
        let bit_width_freq = arr.statistics().compute_bit_width_freq().unwrap();
        let trailing_zeros_freq = arr.statistics().compute_trailing_zero_freq().unwrap();
        let run_count = arr.statistics().compute_run_count().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 5);
        assert!(is_sorted);
        assert!(is_strict_sorted);
        assert!(!is_constant);
        assert_eq!(
            bit_width_freq,
            vec![
                0usize, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(
            trailing_zeros_freq,
            vec![
                // 1, 3, 5 have 0 trailing zeros
                // 2 has 1 trailing zero, 4 has 2 trailing zeros
                3usize, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(run_count, 5);
    }

    #[test]
    fn stats_u8() {
        let arr = PrimitiveArray::from(vec![1u8, 2, 3, 4, 5]);
        let min: u8 = arr.statistics().compute_min().unwrap();
        let max: u8 = arr.statistics().compute_max().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 5);
    }

    #[test]
    fn nullable_stats_u8() {
        let arr = PrimitiveArray::from_nullable_vec(vec![None, None, Some(1i32), Some(2), None]);
        let min: i32 = arr.statistics().compute_min().unwrap();
        let max: i32 = arr.statistics().compute_max().unwrap();
        let null_count: usize = arr.statistics().compute_null_count().unwrap();
        let is_strict_sorted: bool = arr.statistics().compute_is_strict_sorted().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 2);
        assert_eq!(null_count, 3);
        assert!(is_strict_sorted);
    }

    #[test]
    fn all_null() {
        let arr = PrimitiveArray::from_nullable_vec(vec![Option::<i32>::None, None, None]);
        let min: Option<Scalar> = arr.statistics().compute(Stat::Min);
        let max: Option<Scalar> = arr.statistics().compute(Stat::Max);
        let null_i32 = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        assert_eq!(min, Some(null_i32.clone()));
        assert_eq!(max, Some(null_i32));
    }
}
