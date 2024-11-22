use std::cmp::Ordering;

use itertools::{Itertools, MinMaxResult};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexResult};
use vortex_scalar::Scalar;

use crate::accessor::ArrayAccessor;
use crate::array::varbin::{varbin_scalar, VarBinArray};
use crate::array::VarBinEncoding;
use crate::nbytes::ArrayNBytes;
use crate::stats::{Stat, StatisticsVTable, StatsSet};
use crate::{ArrayDType, ArrayLen};

impl StatisticsVTable<VarBinArray> for VarBinEncoding {
    fn compute_statistics(&self, array: &VarBinArray, stat: Stat) -> VortexResult<StatsSet> {
        if stat == Stat::UncompressedSizeInBytes {
            return Ok(StatsSet::of(stat, array.nbytes()));
        }

        if array.is_empty()
            || stat == Stat::TrueCount
            || stat == Stat::RunCount
            || stat == Stat::BitWidthFreq
            || stat == Stat::TrailingZeroFreq
        {
            return Ok(StatsSet::default());
        }

        Ok(match stat {
            Stat::IsConstant | Stat::NullCount => {
                let null_count = array.validity().null_count(array.len())?;
                if null_count == array.len() {
                    return Ok(StatsSet::nulls(array.len(), array.dtype()));
                }

                let mut stats = StatsSet::of(Stat::NullCount, null_count);
                if stat == Stat::NullCount {
                    return Ok(stats);
                }

                let is_constant = if null_count > 0 {
                    // we know that there is at least one null, but not all nulls, so it's not constant
                    false
                } else {
                    array.with_iterator(|iter| compute_is_constant(&mut iter.flatten()))?
                };
                stats.set(Stat::IsConstant, is_constant);
                stats
            }
            Stat::Min | Stat::Max => {
                // handle min and max in the same loop
                let Some((min, max)) = array
                    .with_iterator(|iter| compute_min_max(&mut iter.flatten(), array.dtype()))?
                else {
                    vortex_panic!(
                        "Unreachable: already checked that array has at least one non-null element"
                    );
                };
                StatsSet::from_iter([(Stat::Min, min), (Stat::Max, max)])
            }
            Stat::IsSorted => {
                let is_sorted = array.with_iterator(|iter| iter.flatten().is_sorted())?;
                let mut stats = StatsSet::of(Stat::IsSorted, is_sorted);
                if !is_sorted {
                    stats.set(Stat::IsStrictSorted, false);
                }
                stats
            }
            Stat::IsStrictSorted => {
                let is_strict_sorted = array.with_iterator(|iter| {
                    iter.flatten()
                        .is_sorted_by(|a, b| matches!(a.cmp(b), Ordering::Less))
                })?;
                let mut stats = StatsSet::of(Stat::IsStrictSorted, is_strict_sorted);
                if is_strict_sorted {
                    stats.set(Stat::IsSorted, true);
                }
                stats
            }
            Stat::UncompressedSizeInBytes
            | Stat::TrueCount
            | Stat::RunCount
            | Stat::BitWidthFreq
            | Stat::TrailingZeroFreq => {
                vortex_panic!(
                    "Unreachable, stat {} should have already been handled",
                    stat
                )
            }
        })
    }
}

fn compute_is_constant(iter: &mut dyn Iterator<Item = &[u8]>) -> bool {
    let Some(first_value) = iter.next() else {
        return true;
    };
    for v in iter {
        if v != first_value {
            return false;
        }
    }
    true
}

fn compute_min_max(
    iter: &mut dyn Iterator<Item = &[u8]>,
    dtype: &DType,
) -> Option<(Scalar, Scalar)> {
    Some(match iter.minmax() {
        MinMaxResult::NoElements => return None,
        MinMaxResult::OneElement(v) => {
            let scalar = varbin_scalar(Buffer::from(v), dtype);
            (scalar.clone(), scalar)
        }
        MinMaxResult::MinMax(min, max) => (
            varbin_scalar(Buffer::from(min), dtype),
            varbin_scalar(Buffer::from(max), dtype),
        ),
    })
}

pub fn compute_stats(iter: &mut dyn Iterator<Item = Option<&[u8]>>, dtype: &DType) -> StatsSet {
    let mut leading_nulls: usize = 0;
    let mut first_value: Option<&[u8]> = None;
    for v in &mut *iter {
        if v.is_none() {
            leading_nulls += 1;
        } else {
            first_value = v;
            break;
        }
    }

    if let Some(first_non_null) = first_value {
        let mut acc = VarBinAccumulator::new(first_non_null);
        acc.n_nulls(leading_nulls);
        iter.for_each(|n| acc.nullable_next(n));
        acc.finish(dtype)
    } else {
        StatsSet::nulls(leading_nulls, dtype)
    }
}

pub struct VarBinAccumulator<'a> {
    min: &'a [u8],
    max: &'a [u8],
    is_sorted: bool,
    is_strict_sorted: bool,
    last_value: &'a [u8],
    null_count: usize,
    runs: usize,
    len: usize,
}

impl<'a> VarBinAccumulator<'a> {
    pub fn new(value: &'a [u8]) -> Self {
        Self {
            min: value,
            max: value,
            is_sorted: true,
            is_strict_sorted: true,
            last_value: value,
            runs: 1,
            null_count: 0,
            len: 1,
        }
    }

    pub fn nullable_next(&mut self, val: Option<&'a [u8]>) {
        match val {
            None => {
                self.null_count += 1;
                self.len += 1;
            }
            Some(v) => self.next(v),
        }
    }

    pub fn n_nulls(&mut self, null_count: usize) {
        self.len += null_count;
        self.null_count += null_count;
    }

    pub fn next(&mut self, val: &'a [u8]) {
        self.len += 1;

        if val < self.min {
            self.min.clone_from(&val);
        } else if val > self.max {
            self.max.clone_from(&val);
        }

        match val.cmp(self.last_value) {
            Ordering::Less => {
                self.is_sorted = false;
                self.is_strict_sorted = false;
            }
            Ordering::Equal => {
                self.is_strict_sorted = false;
                return;
            }
            Ordering::Greater => {}
        }
        self.last_value = val;
        self.runs += 1;
    }

    pub fn finish(&self, dtype: &DType) -> StatsSet {
        let is_constant =
            (self.min == self.max && self.null_count == 0) || self.null_count == self.len;

        StatsSet::from_iter([
            (Stat::Min, varbin_scalar(Buffer::from(self.min), dtype)),
            (Stat::Max, varbin_scalar(Buffer::from(self.max), dtype)),
            (Stat::RunCount, self.runs.into()),
            (Stat::IsSorted, self.is_sorted.into()),
            (Stat::IsStrictSorted, self.is_strict_sorted.into()),
            (Stat::IsConstant, is_constant.into()),
            (Stat::NullCount, self.null_count.into()),
        ])
    }
}

#[cfg(test)]
mod test {
    use std::ops::Deref;

    use vortex_buffer::{Buffer, BufferString};
    use vortex_dtype::{DType, Nullability};

    use crate::array::varbin::VarBinArray;
    use crate::stats::{ArrayStatistics, Stat};

    fn array(dtype: DType) -> VarBinArray {
        VarBinArray::from_vec(
            vec!["hello world", "hello world this is a long string"],
            dtype,
        )
    }

    #[test]
    fn utf8_stats() {
        let arr = array(DType::Utf8(Nullability::NonNullable));
        assert_eq!(
            arr.statistics().compute_min::<BufferString>().unwrap(),
            BufferString::from("hello world".to_string())
        );
        assert_eq!(
            arr.statistics().compute_max::<BufferString>().unwrap(),
            BufferString::from("hello world this is a long string".to_string())
        );
        assert_eq!(arr.statistics().compute_run_count().unwrap(), 2);
        assert!(!arr.statistics().compute_is_constant().unwrap());
        assert!(arr.statistics().compute_is_sorted().unwrap());
    }

    #[test]
    fn binary_stats() {
        let arr = array(DType::Binary(Nullability::NonNullable));
        assert_eq!(
            arr.statistics().compute_min::<Buffer>().unwrap().deref(),
            b"hello world"
        );
        assert_eq!(
            arr.statistics().compute_max::<Buffer>().unwrap().deref(),
            "hello world this is a long string".as_bytes()
        );
        assert_eq!(arr.statistics().compute_run_count().unwrap(), 2);
        assert!(!arr.statistics().compute_is_constant().unwrap());
        assert!(arr.statistics().compute_is_sorted().unwrap());
    }

    #[test]
    fn some_nulls() {
        let array = VarBinArray::from_iter(
            vec![
                Some("hello world"),
                None,
                Some("hello world this is a long string"),
                None,
            ],
            DType::Utf8(Nullability::Nullable),
        );
        assert_eq!(
            array.statistics().compute_min::<BufferString>().unwrap(),
            BufferString::from("hello world".to_string())
        );
        assert_eq!(
            array.statistics().compute_max::<BufferString>().unwrap(),
            BufferString::from("hello world this is a long string".to_string())
        );
    }

    #[test]
    fn all_nulls() {
        let array = VarBinArray::from_iter(
            vec![Option::<&str>::None, None, None],
            DType::Utf8(Nullability::Nullable),
        );
        assert!(array.statistics().get(Stat::Min).is_none());
        assert!(array.statistics().get(Stat::Max).is_none());
    }
}
