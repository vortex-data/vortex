// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt as _;
use futures::StreamExt as _;
use futures::TryStreamExt as _;
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::stream;
use vortex_array::IntoArray as _;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::ScanDriver;
use crate::DataSourceScan;
use crate::Partition;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::ScanRequest;

struct TestScan(Vec<PartitionRef>);

impl DataSourceScan for TestScan {
    fn dtype(&self) -> &DType {
        &DType::Primitive(PType::U64, Nullability::NonNullable)
    }
    fn partition_count(&self) -> Precision<usize> {
        Precision::Exact(self.0.len())
    }
    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter(self.0.into_iter().map(Ok)).boxed()
    }
}

struct Fragment {
    index: usize,
    started: Arc<AtomicUsize>,
    gate: Option<oneshot::Receiver<()>>,
    fail: bool,
}

impl Partition for Fragment {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn index(&self) -> usize {
        self.index
    }
    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(3)
    }
    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }
    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        self.started.fetch_add(1, Ordering::Relaxed);
        if self.fail {
            vortex_bail!("test partition failed");
        }
        let array =
            PrimitiveArray::from_iter((self.index * 3..self.index * 3 + 3).map(|x| x as u64))
                .into_array();
        Ok(Box::pin(ArrayStreamAdapter::new(
            array.dtype().clone(),
            stream::once(async move {
                if let Some(gate) = self.gate {
                    let _ = gate.await;
                }
                Ok(array)
            }),
        )))
    }
}

fn fragments(
    started: &Arc<AtomicUsize>,
    gate: Option<oneshot::Receiver<()>>,
    fail: bool,
) -> TestScan {
    let mut gate = gate;
    TestScan(
        (0..10)
            .map(|index| {
                Box::new(Fragment {
                    index,
                    started: Arc::clone(started),
                    gate: gate.take(),
                    fail: fail && index == 0,
                }) as PartitionRef
            })
            .collect(),
    )
}

#[test]
fn outputs_poll_independently_and_cancel_on_drop() {
    let started = Arc::new(AtomicUsize::new(0));
    let (release, gate) = oneshot::channel();
    let mut outputs = ScanDriver::new(
        Box::new(fragments(&started, Some(gate), false)),
        &ScanRequest::default(),
    )
    .into_streams(2.try_into().unwrap(), 1.try_into().unwrap());
    assert_eq!(started.load(Ordering::Relaxed), 0);
    assert!(outputs[0].next().now_or_never().is_none());
    assert_eq!(
        block_on(outputs[1].next())
            .unwrap()
            .unwrap()
            .as_::<Primitive>()
            .as_slice::<u64>(),
        &[3, 4, 5]
    );
    assert_eq!(started.load(Ordering::Relaxed), 2);
    drop(outputs.remove(1));
    assert!(block_on(outputs[0].next()).unwrap().is_err());
    assert!(block_on(outputs[0].next()).is_none());
    assert!(release.send(()).is_err());

    let outputs = ScanDriver::new(
        Box::new(fragments(&started, None, false)),
        &ScanRequest::default(),
    )
    .into_streams(3.try_into().unwrap(), 2.try_into().unwrap());
    let arrays = block_on(stream::select_all(outputs).try_collect::<Vec<_>>()).unwrap();
    let mut starts = arrays
        .iter()
        .map(|a| a.as_::<Primitive>().as_slice::<u64>()[0])
        .collect::<Vec<_>>();
    starts.sort_unstable();
    assert_eq!(starts, (0..30).step_by(3).collect::<Vec<_>>());
}

#[test]
fn ordered_limit_preserves_sequence_and_stops_work() {
    let started = Arc::new(AtomicUsize::new(0));
    let (release, gate) = oneshot::channel();
    let request = ScanRequest {
        ordered: true,
        limit: Some(4),
        ..Default::default()
    };
    let mut outputs = ScanDriver::new(Box::new(fragments(&started, Some(gate), false)), &request)
        .into_streams(4.try_into().unwrap(), 4.try_into().unwrap());
    assert_eq!(outputs.len(), 1);
    let stream = &mut outputs[0];
    assert!(stream.next().now_or_never().is_none());
    assert_eq!(started.load(Ordering::Relaxed), 1);
    release.send(()).unwrap();
    let arrays = block_on(stream.try_collect::<Vec<_>>()).unwrap();
    assert_eq!(arrays.len(), 2);
    assert_eq!(arrays[0].as_::<Primitive>().as_slice::<u64>(), &[0, 1, 2]);
    assert_eq!(arrays[1].as_::<Primitive>().as_slice::<u64>(), &[3]);
    assert_eq!(started.load(Ordering::Relaxed), 2);

    let request = ScanRequest {
        limit: Some(0),
        ..Default::default()
    };
    let mut empty = ScanDriver::new(Box::new(fragments(&started, None, false)), &request)
        .into_stream(4.try_into().unwrap());
    assert!(block_on(empty.next()).is_none());
    assert_eq!(started.load(Ordering::Relaxed), 2);
}

#[test]
fn partition_error_cancels_siblings() {
    let started = Arc::new(AtomicUsize::new(0));
    let mut outputs = ScanDriver::new(
        Box::new(fragments(&started, None, true)),
        &ScanRequest::default(),
    )
    .into_streams(2.try_into().unwrap(), 1.try_into().unwrap());
    let error = block_on(outputs[0].next()).unwrap().unwrap_err();
    assert!(error.to_string().contains("test partition failed"));
    assert!(block_on(outputs[1].next()).unwrap().is_err());
    assert!(block_on(outputs[0].next()).is_none());
    assert!(block_on(outputs[1].next()).is_none());
    assert_eq!(started.load(Ordering::Relaxed), 1);
}
