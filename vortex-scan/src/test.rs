// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Source-independent fixtures shared by plan and engine tests.

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

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

#[derive(Clone, Debug)]
pub struct TestSource;

#[derive(Clone, Debug)]
pub struct TestSourceData {
    segment: SegmentId,
    array: Option<ArrayRef>,
}

fn source(dtype: DType, row_count: u64, segment: SegmentId, array: Option<ArrayRef>) -> PlanRef {
    FilterPlan::new(
        PlanParts {
            vtable: TestSource,
            dtype,
            row_count,
            slots: PlanSlots::default(),
            data: TestSourceData { segment, array },
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
            array: plan.data().array.clone(),
            rows: usize::try_from(rows.start)?..usize::try_from(rows.end)?,
            mask: mask.clone(),
            done: false,
        })))
    }
}

struct TestRead {
    segment: Option<SegmentId>,
    array: Option<ArrayRef>,
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
        cx.take_segment()
            .ok_or_else(|| vortex_err!("test source ran without its read completion"))?;
        self.done = true;
        let array = self
            .array
            .as_ref()
            .ok_or_else(|| vortex_err!("shape-only source was executed"))?;
        let array = if self.rows == (0..array.len()) {
            array.clone()
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

/// An in-memory source whose byte requests gate access to arrays held by the test plans.
#[derive(Default)]
pub struct Store {
    segments: Mutex<Vec<BufferHandle>>,
    requests: AtomicUsize,
}

impl Store {
    pub fn flat(&self, array: &ArrayRef) -> VortexResult<PlanRef> {
        let array = array.optimize_ctx(&SESSION)?;
        let mut segments = self.segments.lock();
        let id = SegmentId::try_from(segments.len())?;
        // A nonempty completion exercises retention and read sharing without a format decoder.
        segments.push(BufferHandle::new_host(ByteBuffer::from(vec![0u8; 4])));
        Ok(source(
            array.dtype().clone(),
            array.len() as u64,
            id,
            Some(array),
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
        let locations = (0..self.len())
            .map(|offset| SegmentLocation {
                offset: offset as u64 * 4,
                length: 4,
                alignment: Alignment::none(),
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
        futures::future::ready(Ok(self.read(id))).boxed()
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
