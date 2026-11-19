// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`ScanPool`] over the partitions of a [`DataSourceScan`].

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_lock::Mutex;
use futures::StreamExt;
use futures::stream;
use futures::stream::Fuse;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;

use crate::DataSourceScanRef;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::ScanPool;

/// A [`ScanPool`] whose units are a scan's partitions.
///
/// Workers share the scan's partition stream: each takes the next partition and polls its
/// [`execute`](crate::Partition::execute) stream inline until it ends. The pool itself never
/// spawns, though a partition's stream may.
///
/// The pool shares no row limit between workers beyond what the partitions themselves share, so
/// an ordered scan with a limit may return more rows than the limit, as its partitions do.
pub struct PartitionPool {
    dtype: DType,
    partitions: Mutex<Fuse<PartitionStream>>,
    partition_count: Option<usize>,
    handed_out: AtomicUsize,
    is_exhausted: AtomicBool,
}

impl PartitionPool {
    /// Opens a pool over the partitions of `scan`.
    pub fn new(scan: DataSourceScanRef) -> Self {
        let dtype = scan.dtype().clone();
        let partition_count = scan.partition_count().into_inner();
        Self {
            dtype,
            partitions: Mutex::new(scan.partitions().fuse()),
            partition_count,
            handed_out: AtomicUsize::new(0),
            is_exhausted: AtomicBool::new(false),
        }
    }

    /// Takes the next partition, waiting for the stream if another worker is polling it.
    async fn next_partition(&self) -> Option<VortexResult<PartitionRef>> {
        let mut partitions = self.partitions.lock().await;
        let partition = partitions.next().await;
        match &partition {
            Some(_) => {
                self.handed_out.fetch_add(1, Ordering::Relaxed);
            }
            None => self.is_exhausted.store(true, Ordering::Relaxed),
        }
        partition
    }
}

impl ScanPool for PartitionPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let arrays = stream::unfold(
            (self, None::<SendableArrayStream>),
            |(pool, mut current)| async move {
                loop {
                    if let Some(mut arrays) = current.take() {
                        if let Some(item) = arrays.next().await {
                            return Some((item, (pool, Some(arrays))));
                        }
                    }
                    let partition = match pool.next_partition().await? {
                        Ok(partition) => partition,
                        Err(err) => return Some((Err(err), (pool, None))),
                    };
                    match partition.execute() {
                        Ok(arrays) => current = Some(arrays),
                        Err(err) => return Some((Err(err), (pool, None))),
                    }
                }
            },
        );
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays))
    }

    fn is_exhausted(&self) -> bool {
        self.is_exhausted.load(Ordering::Relaxed)
    }

    fn progress(&self) -> f64 {
        if self.is_exhausted() {
            return 1.0;
        }
        match self.partition_count {
            Some(count) if count > 0 => {
                (self.handed_out.load(Ordering::Relaxed) as f64 / count as f64).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }
}
