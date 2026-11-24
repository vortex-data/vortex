// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Source-independent fixtures shared by plan and engine tests.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::ready;
use futures::stream;
use parking_lot::Mutex;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use crate::DataSourceScan;
use crate::DataSourceScanRef;
use crate::Partition;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::execution::SegmentLocation;
use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::execution::operator::SourceBlocked;
use crate::execution::operator::SourceReady;
use crate::execution::operator::Step;
use crate::execution::pipeline::Chain;
use crate::execution::pipeline::Compiler;
use crate::plan::ConcatPlan;
use crate::plan::FilterPlan;
use crate::plan::LazyPlanRef;
use crate::plan::PackPlan;
use crate::plan::Plan;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanSlots;
use crate::plan::PlanVTable;
use crate::plan::TakePlan;
use crate::source::ScanInput;
use crate::source::SegmentFuture;
use crate::source::SegmentId;
use crate::source::SegmentSource;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[tokio::test]
async fn into_array_stream_preserves_dtype_and_partition_order() -> VortexResult<()> {
    let first = buffer![1_u32].into_array();
    let second = buffer![2_u32, 3].into_array();
    let dtype = first.dtype().clone();
    let scan: DataSourceScanRef = Box::new(TestArrayScan {
        dtype: dtype.clone(),
        arrays: vec![first, second],
    });

    let stream = scan.into_array_stream();
    assert_eq!(stream.dtype(), &dtype);
    let lengths: Vec<_> = stream.map_ok(|array| array.len()).try_collect().await?;
    assert_eq!(lengths, [1, 2]);
    Ok(())
}

struct TestArrayScan {
    dtype: DType,
    arrays: Vec<ArrayRef>,
}

impl DataSourceScan for TestArrayScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::Exact(self.arrays.len())
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter(
            self.arrays.into_iter().enumerate().map(|(index, array)| {
                Ok(Box::new(TestArrayPartition(index, array)) as PartitionRef)
            }),
        )
        .boxed()
    }
}

struct TestArrayPartition(usize, ArrayRef);

impl Partition for TestArrayPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.0
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(self.1.len() as u64)
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        Ok(Box::pin(self.1.to_array_stream()))
    }
}

#[derive(Clone, Debug)]
pub struct TestSource;

#[derive(Clone, Debug)]
pub struct TestSourceData {
    segment: SegmentId,
    array_ctx: Option<ReadContext>,
}

fn source(
    dtype: DType,
    row_count: u64,
    segment: SegmentId,
    array_ctx: Option<ReadContext>,
) -> PlanRef {
    FilterPlan::new(
        PlanParts {
            vtable: TestSource,
            dtype,
            row_count,
            slots: PlanSlots::default(),
            data: TestSourceData { segment, array_ctx },
        }
        .into_plan(),
    )
    .into_plan()
}

/// A shape-only source, used by optimizer tests that never execute payloads.
pub fn flat(row_count: u64, dtype: DType, segment: u32) -> PlanRef {
    source(dtype, row_count, segment.into(), None)
}

impl PlanVTable for TestSource {
    type PlanData = TestSourceData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.test.source");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        None
    }

    fn is_data_source(_plan: &Plan<Self>) -> bool {
        true
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        if mask.all_false() {
            return Ok(None);
        }
        compiler.register_segment(plan.data().segment);
        Ok(Some(Chain::new(TestRead {
            segment: Some(plan.data().segment),
            dtype: plan.dtype().clone(),
            row_count: usize::try_from(plan.row_count())?,
            array_ctx: plan.data().array_ctx.clone(),
            rows: usize::try_from(rows.start)?..usize::try_from(rows.end)?,
            mask: mask.clone(),
            done: false,
        })))
    }
}

struct TestRead {
    segment: Option<SegmentId>,
    dtype: DType,
    row_count: usize,
    array_ctx: Option<ReadContext>,
    rows: Range<usize>,
    mask: Mask,
    done: bool,
}

impl Source for TestRead {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if let Some(segment) = self.segment.take() {
            cx.request_read(segment);
            return Ok(SourceReady::Blocked(SourceBlocked::Io));
        }
        Ok(if self.done || cx.has_bytes() {
            SourceReady::Ready
        } else {
            SourceReady::Blocked(SourceBlocked::Io)
        })
    }
}

impl Operator for TestRead {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.done {
            return Ok(Step::End);
        }
        let (_, bytes) = cx
            .take_segment()
            .ok_or_else(|| vortex_err!("test source ran without its read completion"))?;
        self.done = true;
        let array_ctx = self
            .array_ctx
            .as_ref()
            .ok_or_else(|| vortex_err!("shape-only source was executed"))?;
        let array = SerializedArray::try_from(bytes)?.decode(
            &self.dtype,
            self.row_count,
            array_ctx,
            cx.session(),
        )?;
        let array = if self.rows == (0..array.len()) {
            array
        } else {
            array.slice(self.rows.clone())?
        };
        let array = if self.mask.all_true() {
            array
        } else {
            array.filter(self.mask.clone())?
        };
        Ok(Step::Final(array))
    }
}

/// In-memory serialized arrays, decoded from the bytes delivered to each test source.
#[derive(Default)]
pub struct Store {
    segments: Mutex<Vec<BufferHandle>>,
    requests: AtomicUsize,
}

impl Store {
    pub fn flat(&self, array: &ArrayRef) -> VortexResult<PlanRef> {
        let array = array.optimize_ctx(&SESSION)?;
        let ctx = ArrayContext::empty();
        let buffers = array.serialize(
            &ctx,
            &SESSION,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )?;
        let mut bytes = ByteBufferMut::empty_aligned(Alignment::new(64));
        for buffer in buffers {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let mut segments = self.segments.lock();
        let id = SegmentId::try_from(segments.len())?;
        segments.push(BufferHandle::new_host(bytes.freeze()));
        Ok(source(
            array.dtype().clone(),
            array.len() as u64,
            id,
            Some(ReadContext::new(ctx.to_ids())),
        ))
    }

    pub fn chunked(&self, array: &ArrayRef, sizes: &[usize]) -> VortexResult<PlanRef> {
        let mut chunks = Vec::with_capacity(sizes.len());
        let mut start = 0;
        for &size in sizes {
            chunks.push(self.flat(&array.slice(start..start + size)?)?);
            start += size;
        }
        assert_eq!(start, array.len(), "chunk sizes must cover the array");
        Ok(ConcatPlan::try_new(array.dtype().clone(), chunks)?.into_plan())
    }

    pub fn read(&self, segment: SegmentId) -> BufferHandle {
        self.segments.lock()[*segment as usize].clone()
    }

    pub fn len(&self) -> usize {
        self.segments.lock().len()
    }

    pub fn requests(&self) -> usize {
        self.requests.load(Ordering::Relaxed)
    }

    pub fn input(self: &Arc<Self>, plan: PlanRef) -> ScanInput {
        let mut offset = 0;
        let locations = self
            .segments
            .lock()
            .iter()
            .map(|segment| {
                let length = u32::try_from(segment.len()).expect("test segment fits u32");
                let location = SegmentLocation {
                    offset,
                    length,
                    alignment: Alignment::new(64),
                };
                offset += u64::from(length);
                location
            })
            .collect();
        ScanInput {
            dtype: plan.dtype().clone(),
            root: LazyPlanRef::ready(plan),
            locations,
            segments: Arc::clone(self) as Arc<dyn SegmentSource>,
            cache: None,
        }
    }
}

impl SegmentSource for Store {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.requests.fetch_add(1, Ordering::Relaxed);
        ready(Ok(self.read(id))).boxed()
    }
}

pub fn concat(row_count: u64, dtype: DType, children: Vec<PlanRef>) -> PlanRef {
    let plan = ConcatPlan::try_new(dtype, children).expect("valid test concatenation");
    assert_eq!(plan.row_count(), row_count);
    plan.into_plan()
}

pub fn struct_plan(row_count: u64, dtype: DType, mut children: Vec<PlanRef>) -> PlanRef {
    let validity = (dtype.nullability() == Nullability::Nullable).then(|| children.remove(0));
    PackPlan::try_new(
        dtype.as_struct_fields().clone(),
        dtype.nullability(),
        row_count,
        children,
        validity,
    )
    .expect("valid test struct")
    .into_plan()
}

pub fn dict(values: PlanRef, codes: PlanRef) -> PlanRef {
    TakePlan::new(codes, values).into_plan()
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;

    use super::*;
    use crate::execution::pipeline::Scan;
    use crate::execution::pipeline::Split;
    use crate::execution::pipeline::Turn;

    fn read(plan: PlanRef, bytes: BufferHandle) -> VortexResult<ArrayRef> {
        let rows = 0..plan.row_count();
        let mask = Mask::new_true(usize::try_from(plan.row_count())?);
        let mut scan = Scan::try_new(SESSION.clone(), plan, vec![Split { rows, mask }])?;
        loop {
            match scan.step()? {
                Turn::Read(request) => scan.deliver(request.id, bytes.clone())?,
                Turn::Output(_, array) => return Ok(array),
                Turn::Waiting | Turn::Done => panic!("test source did not emit its payload"),
            }
        }
    }

    #[test]
    fn source_decodes_the_delivered_payload() -> VortexResult<()> {
        let store = Store::default();
        let first = buffer![1i32, 2, 3].into_array();
        let second = buffer![4i32, 5, 6].into_array();
        let plan = store.flat(&first)?;
        store.flat(&second)?;
        // Deliver another segment's valid bytes. A plan-captured array would return `first`.
        let actual = read(plan, store.read(1.into()))?;
        assert_arrays_eq!(actual, second, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn source_rejects_corrupt_payload() -> VortexResult<()> {
        let store = Store::default();
        let plan = store.flat(&buffer![1i32, 2, 3].into_array())?;
        let bytes = BufferHandle::new_host(ByteBuffer::from(vec![0u8; 4]));
        assert!(read(plan, bytes).is_err());
        Ok(())
    }
}
