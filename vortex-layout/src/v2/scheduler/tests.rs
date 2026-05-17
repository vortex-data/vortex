use std::hash::Hash;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use super::*;
use crate::segments::SegmentId;
use crate::v2::demand::RowDemand;
use crate::v2::domain::Domain;
use crate::v2::domain::DomainId;
use crate::v2::plans::LayoutPlan;
use crate::v2::plans::LayoutPlanRef;
use crate::v2::plans::PartitionStats;
use crate::v2::plans::chunked::ChunkedPlan;
use crate::v2::plans::lower_to_single_scheduler;
use crate::v2::runtime_info::Coverage;
use crate::v2::runtime_info::OrdinalDemand;
use crate::v2::runtime_info::PermitPolicy;
use crate::v2::runtime_info::PermitReason;
use crate::v2::runtime_info::WorkEstimate;
use crate::v2::scan_ctx::ScanCtx;

const ROWS: DomainId = DomainId::new(0);

fn high_value_estimate() -> WorkEstimate {
    WorkEstimate::new(1.0, 100.0, 0.95, 0.9)
}

fn low_value_estimate() -> WorkEstimate {
    WorkEstimate::new(100.0, 1.0, 0.05, 0.5)
}

fn i64_dtype() -> DType {
    DType::Primitive(PType::I64, Nullability::NonNullable)
}

fn bool_dtype() -> DType {
    DType::Bool(Nullability::NonNullable)
}

struct TestLeaf {
    tag: &'static str,
    dtype: DType,
    row_count: u64,
}

impl TestLeaf {
    fn new(tag: &'static str, dtype: DType, row_count: u64) -> Self {
        Self {
            tag,
            dtype,
            row_count,
        }
    }
}

impl PartialEq for TestLeaf {
    fn eq(&self, other: &Self) -> bool {
        self.tag == other.tag && self.dtype == other.dtype && self.row_count == other.row_count
    }
}

impl Eq for TestLeaf {}

impl Hash for TestLeaf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tag.hash(state);
        self.dtype.hash(state);
        self.row_count.hash(state);
    }
}

impl LayoutPlan for TestLeaf {
    fn schema(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> usize {
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition != 0 {
            vortex_bail!("TestLeaf partition out of range");
        }
        Ok(PartitionStats::for_range(0..self.row_count))
    }

    fn output_ordered(&self) -> bool {
        true
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        &[]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if !children.is_empty() {
            vortex_bail!("TestLeaf has no children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        _row_range: Range<u64>,
        _demand: &RowDemand,
        _ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        unreachable!("TestLeaf is lowering-only")
    }
}

struct TestContainer {
    children: Vec<LayoutPlanRef>,
    dtype: DType,
    row_count: u64,
}

impl TestContainer {
    fn new(children: Vec<LayoutPlanRef>, row_count: u64) -> Self {
        Self {
            children,
            dtype: i64_dtype(),
            row_count,
        }
    }
}

impl PartialEq for TestContainer {
    fn eq(&self, other: &Self) -> bool {
        crate::v2::plans::plan_slices_eq(&self.children, &other.children)
            && self.row_count == other.row_count
    }
}

impl Eq for TestContainer {}

impl Hash for TestContainer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::v2::plans::hash_plan_slice(&self.children, state);
        self.row_count.hash(state);
    }
}

impl LayoutPlan for TestContainer {
    fn schema(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> usize {
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition != 0 {
            vortex_bail!("TestContainer partition out of range");
        }
        Ok(PartitionStats::for_range(0..self.row_count))
    }

    fn output_ordered(&self) -> bool {
        true
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true; self.children.len()]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true; self.children.len()]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        &self.children
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        Ok(Arc::new(Self {
            children,
            dtype: self.dtype.clone(),
            row_count: self.row_count,
        }))
    }

    fn execute(
        &self,
        _row_range: Range<u64>,
        _demand: &RowDemand,
        _ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        unreachable!("TestContainer is lowering-only")
    }
}

#[test]
fn unknown_demand_is_correct_but_waitable() -> VortexResult<()> {
    let demand = OrdinalDemand::new(ROWS, 1_000);
    let target = 100..200;

    assert_eq!(demand.coverage(&target)?, Coverage::Unknown);
    let correctness_mask = demand.mask_for(&target)?;
    assert!(correctness_mask.all_true());

    let policy = PermitPolicy::new(128, 16, 1.0);
    let permit = policy.value_consumer_permit(&demand, &target, high_value_estimate())?;
    assert_eq!(permit.reason(), PermitReason::WaitForDemand);
    assert_eq!(permit.rows_to_poll(), 0);
    Ok(())
}

#[test]
fn demand_producer_runs_to_first_uncovered_range() -> VortexResult<()> {
    let mut demand = OrdinalDemand::new(ROWS, 1_000);
    let policy = PermitPolicy::new(128, 16, 1.0);
    let target = 0..512;

    let permit = policy.demand_producer_permit(&demand, &target)?;
    assert_eq!(permit.reason(), PermitReason::DriveDemandProducer);
    assert_eq!(permit.range(), &(0..128));

    demand.publish(0..128, Mask::new_false(128))?;
    let permit = policy.demand_producer_permit(&demand, &target)?;
    assert_eq!(permit.range(), &(128..256));
    Ok(())
}

#[test]
fn all_false_covered_prefix_skips_value_work() -> VortexResult<()> {
    let mut demand = OrdinalDemand::new(ROWS, 1_000);
    demand.publish(0..128, Mask::new_false(128))?;

    let policy = PermitPolicy::new(128, 16, 1.0);
    let permit = policy.value_consumer_permit(&demand, &(0..512), high_value_estimate())?;

    assert_eq!(permit.reason(), PermitReason::SkipAllFalse);
    assert_eq!(permit.range(), &(0..128));
    assert_eq!(permit.rows_to_poll(), 0);
    Ok(())
}

#[test]
fn known_live_prefix_allows_value_work() -> VortexResult<()> {
    let mut demand = OrdinalDemand::new(ROWS, 1_000);
    demand.publish(0..128, Mask::new_true(128))?;

    let policy = PermitPolicy::new(128, 16, 1.0);
    let permit = policy.value_consumer_permit(&demand, &(0..512), high_value_estimate())?;

    assert_eq!(permit.reason(), PermitReason::ProceedWithKnownDemand);
    assert_eq!(permit.range(), &(0..128));
    assert_eq!(permit.rows_to_poll(), 128);
    Ok(())
}

#[test]
fn low_value_unknown_range_gets_small_speculative_permit() -> VortexResult<()> {
    let demand = OrdinalDemand::new(ROWS, 1_000);
    let policy = PermitPolicy::new(128, 16, 1.0);
    let permit = policy.value_consumer_permit(&demand, &(0..512), low_value_estimate())?;

    assert_eq!(permit.reason(), PermitReason::Speculate);
    assert_eq!(permit.range(), &(0..16));
    assert_eq!(permit.rows_to_poll(), 16);
    Ok(())
}

#[test]
fn sorted_domain_can_advertise_ordinal_lowering() {
    let sorted = Domain::Sorted {
        id: DomainId::new(1),
        key: "event_time",
        ordinal: ROWS,
    };
    let keyed = Domain::Keyed {
        id: DomainId::new(2),
        key: "user_id",
    };

    assert_eq!(sorted.ordinal_mapping(), Some(ROWS));
    assert_eq!(keyed.ordinal_mapping(), None);
}

fn scheduler_morsel(
    id: u64,
    role: MorselRole,
    order_key: Range<u64>,
    stage_count: u16,
    priority: MorselPriority,
) -> SchedulerMorsel {
    SchedulerMorsel::new(
        MorselId::new(id),
        ROWS,
        order_key,
        role,
        stage_count,
        MorselEstimate::new(10_000, 0, 1024),
        priority,
    )
}

#[test]
fn partition_scheduler_prioritizes_information_over_row_offset() {
    let mut scheduler =
        PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::default());
    assert_eq!(scheduler.id(), PartitionSchedulerId::new(0));

    let early_value = scheduler_morsel(
        1,
        MorselRole::ValueProducer,
        0..1024,
        2,
        MorselPriority::value_work(1_000, 10_000, 0),
    );
    let later_information = scheduler_morsel(
        2,
        MorselRole::InformationProducer,
        64 * 1024..65 * 1024,
        2,
        MorselPriority::information_producer(100_000, 1_000, 10_000),
    );

    assert!(scheduler.enqueue(
        SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), early_value)),
        0
    ));
    assert!(scheduler.enqueue(
        SchedulerTask::Work(SchedulerWorkTask::new(
            PipelineId::new(0),
            later_information
        )),
        0
    ));

    assert_eq!(
        scheduler.make_progress(1),
        Some(SchedulerStep::Advanced {
            morsel_id: MorselId::new(2),
            pipeline: PipelineId::new(0),
            from_stage: 0,
            to_stage: 1,
        })
    );
}

#[test]
fn partition_scheduler_advances_one_pipeline_stage_per_step() {
    let mut scheduler =
        PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::default());
    let morsel = scheduler_morsel(
        7,
        MorselRole::InformationConsumer,
        0..1024,
        2,
        MorselPriority::value_work(10_000, 10_000, 0),
    );

    assert!(scheduler.enqueue(
        SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), morsel)),
        0
    ));
    assert_eq!(
        scheduler.make_progress(0),
        Some(SchedulerStep::Advanced {
            morsel_id: MorselId::new(7),
            pipeline: PipelineId::new(0),
            from_stage: 0,
            to_stage: 1,
        })
    );
    assert_eq!(
        scheduler.make_progress(0),
        Some(SchedulerStep::Completed {
            morsel_id: MorselId::new(7),
            pipeline: PipelineId::new(0),
        })
    );
    assert_eq!(scheduler.make_progress(0), None);
}

#[test]
fn partition_scheduler_bounds_queued_memory() {
    let mut scheduler =
        PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::new(8, 1500));
    let first = scheduler_morsel(
        1,
        MorselRole::ValueProducer,
        0..1024,
        1,
        MorselPriority::value_work(1, 1, 0),
    );
    let second = scheduler_morsel(
        2,
        MorselRole::ValueProducer,
        1024..2048,
        1,
        MorselPriority::value_work(1, 1, 0),
    );

    assert!(scheduler.enqueue(
        SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), first)),
        0
    ));
    assert!(!scheduler.enqueue(
        SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), second)),
        0
    ));
    assert_eq!(scheduler.len(), 1);
    assert_eq!(scheduler.queued_memory_bytes(), 1024);
}

#[test]
fn partition_scheduler_queue_holds_io_and_control_events() {
    let mut scheduler =
        PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::default());
    let segment = SchedulerSegmentTask::metadata_only(
        IoRequestId::new(11),
        PipelineId::new(4),
        SegmentId::from(7),
        ROWS,
        0..4096,
        32 * 1024,
        MorselPriority::information_producer(10_000, 1_000, 0),
    );
    let control = SchedulerControlEvent::Rebalance {
        reason: "test rebalance",
    };

    assert!(scheduler.enqueue(SchedulerTask::Control(control.clone()), 0));
    assert!(scheduler.enqueue(SchedulerTask::Segment(segment), 0));

    assert_eq!(
        scheduler.make_progress(0),
        Some(SchedulerStep::CompletedSegment {
            request_id: IoRequestId::new(11),
            pipeline: PipelineId::new(4),
            segment_id: SegmentId::from(7),
            bytes: 32 * 1024,
        })
    );
    assert_eq!(
        scheduler.make_progress(0),
        Some(SchedulerStep::Control { event: control })
    );
}

#[test]
fn partition_scheduler_steals_only_non_critical_data_morsels() {
    let mut scheduler =
        PartitionScheduler::new(PartitionSchedulerId::new(0), SchedulerBudget::default());
    let data = scheduler_morsel(
        1,
        MorselRole::ValueProducer,
        0..1024,
        1,
        MorselPriority::value_work(1, 1, 0),
    );
    let information = scheduler_morsel(
        2,
        MorselRole::InformationProducer,
        1024..2048,
        1,
        MorselPriority::information_producer(100, 1, 0),
    );
    let sink = scheduler_morsel(
        3,
        MorselRole::Sink,
        2048..3072,
        1,
        MorselPriority::value_work(1, 1, 0),
    );

    assert!(scheduler.enqueue(
        SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), data)),
        0
    ));
    assert!(scheduler.enqueue(
        SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), information)),
        0
    ));
    assert!(scheduler.enqueue(
        SchedulerTask::Work(SchedulerWorkTask::new(PipelineId::new(0), sink)),
        0
    ));

    let stolen = scheduler.stealable_morsels(4);
    assert_eq!(stolen.len(), 1);
    assert_eq!(stolen[0].id(), MorselId::new(1));
    assert_eq!(scheduler.len(), 2);
}

#[test]
fn layout_lowering_closes_leaf_pipeline() -> VortexResult<()> {
    let leaf = TestLeaf::new("bool-leaf", bool_dtype(), 100);
    let mut ctx = lower_to_single_scheduler(&leaf, 0..100)?;

    assert_eq!(ctx.pipeline_count(), 1);
    assert_eq!(ctx.initial_work_count(), 1);
    assert_eq!(ctx.queued_event_count(), 1);
    assert!(ctx.initial_work()[0].source_label().contains(":leaf:"));
    assert_eq!(
        ctx.initial_work()[0].role(),
        MorselRole::InformationProducer
    );
    assert_eq!(ctx.initial_work()[0].pipeline().index(), 0);

    let report = ctx.drive_to_completion();
    assert_eq!(report.completed_morsels(), 1);
    assert_eq!(report.steps(), 3);
    Ok(())
}

#[test]
fn unary_layout_lowering_prepends_operator_to_child_pipeline() -> VortexResult<()> {
    let child: LayoutPlanRef = Arc::new(TestLeaf::new("value", i64_dtype(), 100));
    let plan = TestContainer::new(vec![Arc::clone(&child)], 100);
    let ctx = lower_to_single_scheduler(&plan, 0..100)?;

    assert_eq!(ctx.pipeline_count(), 1);
    assert_eq!(ctx.initial_work_count(), 1);

    let pipeline = ctx.initial_work()[0].pipeline();
    let transforms = ctx.pipeline_transforms(pipeline).unwrap();
    assert_eq!(transforms.len(), 1);
    assert!(transforms[0].label().contains(":1children"));
    assert!(
        ctx.pipeline_sink(pipeline)
            .is_some_and(|sink| sink.label().starts_with("root:"))
    );
    Ok(())
}

#[test]
fn multi_input_layout_lowering_starts_side_pipelines() -> VortexResult<()> {
    let value: LayoutPlanRef = Arc::new(TestLeaf::new("value", i64_dtype(), 100));
    let info: LayoutPlanRef = Arc::new(TestLeaf::new("info", bool_dtype(), 100));
    let plan = TestContainer::new(vec![Arc::clone(&value), Arc::clone(&info)], 100);
    let ctx = lower_to_single_scheduler(&plan, 0..100)?;

    assert_eq!(ctx.pipeline_count(), 3);
    assert_eq!(ctx.initial_work_count(), 3);
    assert!(ctx.initial_work()[0].source_label().contains(":leaf:"));
    assert!(ctx.initial_work()[1].source_label().contains(":leaf:"));
    assert!(
        ctx.initial_work()[2]
            .source_label()
            .contains(":node-output:")
    );
    assert!(
        ctx.pipeline_sink(ctx.initial_work()[0].pipeline())
            .is_some_and(|sink| sink.label().contains(":input0:"))
    );
    assert!(
        ctx.pipeline_sink(ctx.initial_work()[1].pipeline())
            .is_some_and(|sink| sink.label().contains(":input1:"))
    );
    assert!(
        ctx.pipeline_sink(PipelineId::new(2))
            .is_some_and(|sink| sink.label().starts_with("root:"))
    );
    Ok(())
}

#[test]
fn layout_lowering_prioritizes_information_leaves() -> VortexResult<()> {
    let value: LayoutPlanRef = Arc::new(TestLeaf::new("value", i64_dtype(), 100));
    let info: LayoutPlanRef = Arc::new(TestLeaf::new("info", bool_dtype(), 100));
    let plan = TestContainer::new(vec![Arc::clone(&value), Arc::clone(&info)], 100);
    let mut ctx = lower_to_single_scheduler(&plan, 0..100)?;

    assert_eq!(ctx.initial_work_count(), 3);
    let info_morsel = ctx.initial_work()[1].morsel();
    let steps = ctx.drain_steps();
    assert!(matches!(
        steps.first(),
        Some(SchedulerStep::Advanced {
            morsel_id,
            ..
        }) if *morsel_id == info_morsel
    ));
    Ok(())
}

#[test]
fn chunked_layout_lowering_preserves_global_order_ranges() -> VortexResult<()> {
    let first: LayoutPlanRef = Arc::new(TestLeaf::new("first", i64_dtype(), 10));
    let second: LayoutPlanRef = Arc::new(TestLeaf::new("second", i64_dtype(), 20));
    let chunked = ChunkedPlan::new(
        vec![Arc::clone(&first), Arc::clone(&second)],
        vec![0, 10, 30],
        i64_dtype(),
    );
    let ctx = lower_to_single_scheduler(&chunked, 5..25)?;

    let leaf_initial_work: Vec<_> = ctx
        .initial_work()
        .iter()
        .filter(|work| work.source_label().contains(":leaf:"))
        .collect();
    assert_eq!(leaf_initial_work.len(), 2);
    assert_eq!(leaf_initial_work[0].local_range(), &(5..10));
    assert_eq!(leaf_initial_work[0].global_range(), &(5..10));
    assert_eq!(leaf_initial_work[1].local_range(), &(0..15));
    assert_eq!(leaf_initial_work[1].global_range(), &(10..25));
    Ok(())
}
