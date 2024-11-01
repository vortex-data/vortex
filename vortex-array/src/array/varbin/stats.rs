use std::cmp::Ordering;

use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::aliases::hash_map::HashMap;
use crate::array::varbin::{varbin_scalar, VarBinArray};
use crate::stats::{ArrayStatisticsCompute, Stat, StatsSet};
use crate::ArrayDType;

impl ArrayStatisticsCompute for VarBinArray {
    fn compute_statistics(&self, _stat: Stat) -> VortexResult<StatsSet> {
        if self.is_empty() {
            return Ok(StatsSet::new());
        }
        self.with_iterator(|iter| compute_stats(iter, self.dtype()))
    }
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

        StatsSet::from(HashMap::from([
            (Stat::Min, varbin_scalar(Buffer::from(self.min), dtype)),
            (Stat::Max, varbin_scalar(Buffer::from(self.max), dtype)),
            (Stat::RunCount, self.runs.into()),
            (Stat::IsSorted, self.is_sorted.into()),
            (Stat::IsStrictSorted, self.is_strict_sorted.into()),
            (Stat::IsConstant, is_constant.into()),
            (Stat::NullCount, self.null_count.into()),
        ]))
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
