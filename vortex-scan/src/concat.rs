// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One partition that executes other partitions one after another.

use std::any::Any;

use futures::TryStreamExt;
use futures::stream;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;

use crate::Partition;
use crate::PartitionRef;

/// A partition that yields the rows of its partitions in their order, with one row limit for all
/// of them.
///
/// Use it where rows must come out in order but each part of the source is its own partition,
/// such as the files of one table fragment. A partition is executed only when the one before it
/// is done, and none is executed once the limit is reached.
pub struct ConcatPartition {
    index: usize,
    dtype: DType,
    partitions: Vec<PartitionRef>,
    limit: Option<u64>,
}

impl ConcatPartition {
    /// A partition with index `index` that yields the rows of `partitions`, each of dtype `dtype`,
    /// in order, and at most `limit` rows in all.
    pub fn new(
        index: usize,
        dtype: DType,
        partitions: Vec<PartitionRef>,
        limit: Option<u64>,
    ) -> Self {
        Self {
            index,
            dtype,
            partitions,
            limit,
        }
    }
}

impl Partition for ConcatPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.index
    }

    fn row_count(&self) -> Precision<u64> {
        let rows = sum(self
            .partitions
            .iter()
            .map(|partition| partition.row_count()));
        rows.map(|rows| self.limit.map_or(rows, |limit| rows.min(limit)))
    }

    fn byte_size(&self) -> Precision<u64> {
        sum(self
            .partitions
            .iter()
            .map(|partition| partition.byte_size()))
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let partitions = stream::iter(self.partitions.into_iter().map(Partition::execute));
        let arrays = ArrayStreamAdapter::new(self.dtype, partitions.try_flatten());

        Ok(match self.limit {
            Some(limit) => arrays.boxed().limit(limit).boxed(),
            None => arrays.boxed(),
        })
    }
}

/// The sum of `values`: exact only when every value is exact, and absent when any is absent.
fn sum(values: impl Iterator<Item = Precision<u64>>) -> Precision<u64> {
    values.fold(Precision::exact(0u64), |sum, value| {
        sum.zip(value)
            .and_then(|(sum, value)| sum.checked_add(value))
    })
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::executor::block_on;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::stats::Precision;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::ConcatPartition;
    use crate::Partition;
    use crate::PartitionRef;

    struct Rows {
        rows: std::ops::Range<u64>,
        executed: Arc<AtomicUsize>,
    }

    impl Partition for Rows {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn index(&self) -> usize {
            0
        }

        fn row_count(&self) -> Precision<u64> {
            Precision::exact(self.rows.end - self.rows.start)
        }

        fn byte_size(&self) -> Precision<u64> {
            Precision::Absent
        }

        fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
            self.executed.fetch_add(1, Ordering::Relaxed);
            Ok(Buffer::from_iter(self.rows)
                .into_array()
                .to_array_stream()
                .boxed())
        }
    }

    fn concat(limit: Option<u64>, executed: &Arc<AtomicUsize>) -> ConcatPartition {
        let partitions = [0..3, 3..5, 5..9]
            .into_iter()
            .map(|rows| -> PartitionRef {
                Box::new(Rows {
                    rows,
                    executed: Arc::clone(executed),
                })
            })
            .collect();
        let dtype = Buffer::<u64>::empty().into_array().dtype().clone();
        ConcatPartition::new(7, dtype, partitions, limit)
    }

    #[rstest]
    #[case::all(None, 9, 3)]
    #[case::within_first(Some(2), 2, 1)]
    #[case::across(Some(4), 4, 2)]
    fn yields_rows_in_order_up_to_the_limit(
        #[case] limit: Option<u64>,
        #[case] rows: u64,
        #[case] executed: usize,
    ) -> VortexResult<()> {
        let counter = Arc::new(AtomicUsize::new(0));
        let partition = Box::new(concat(limit, &counter));
        assert_eq!(partition.index(), 7);
        assert_eq!(partition.row_count(), Precision::exact(rows));

        let array = block_on(partition.execute()?.read_all())?;
        assert_eq!(counter.load(Ordering::Relaxed), executed);

        let mut ctx = VortexSession::empty().create_execution_ctx();
        assert_arrays_eq!(array, Buffer::from_iter(0..rows).into_array(), &mut ctx);
        Ok(())
    }
}
