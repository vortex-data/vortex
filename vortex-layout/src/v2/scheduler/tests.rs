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
use crate::v2::dataflow::Coverage;
use crate::v2::dataflow::OrdinalDemand;
use crate::v2::dataflow::PermitPolicy;
use crate::v2::dataflow::PermitReason;
use crate::v2::dataflow::WorkEstimate;
use crate::v2::demand::RowDemand;
use crate::v2::domain::Domain;
use crate::v2::domain::DomainId;
use crate::v2::domain::GrantKey;
use crate::v2::domain::SubplanId;
use crate::v2::plans::LayoutPlan;
use crate::v2::plans::LayoutPlanRef;
use crate::v2::plans::PartitionStats;
use crate::v2::plans::chunked::ChunkedPlan;
use crate::v2::plans::lower_to_single_scheduler;
use crate::v2::scan_ctx::ScanCtx;

const ROWS: DomainId = DomainId::new(0);
const CHEAP_FILTER: SubplanId = SubplanId::new(1);
const EXPENSIVE_FILTER: SubplanId = SubplanId::new(2);
const THIRD_FILTER: SubplanId = SubplanId::new(3);

fn high_value_estimate() -> WorkEstimate {
    WorkEstimate::new(1.0, 100.0, 0.95, 0.9)
}

fn low_value_estimate() -> WorkEstimate {
    WorkEstimate::new(100.0, 1.0, 0.05, 0.5)
}

fn narrow_output(rows: u64) -> OutputEstimate {
    OutputEstimate::new(rows, rows * 8)
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
        _frontier: &OutputFrontier,
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
        _frontier: &OutputFrontier,
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
fn demand_producer_runs_to_first_uncovered_frontier() -> VortexResult<()> {
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

#[test]
fn output_grants_are_scoped_by_domain_and_subplan() -> VortexResult<()> {
    let cheap_key = GrantKey::new(ROWS, CHEAP_FILTER);
    let expensive_key = GrantKey::new(ROWS, EXPENSIVE_FILTER);
    let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
    grantor.register_domain(ROWS, 1_000_000);
    grantor.set_frontier(cheap_key, 128 * 1024)?;
    grantor.set_frontier(expensive_key, 8 * 1024)?;

    let cheap_grant = grantor.grant(OutputGrantRequest::new(
        cheap_key,
        0..1_000_000,
        OutputEstimate::new(1_000_000, 8_000_000),
    ))?;
    let expensive_grant = grantor.grant(OutputGrantRequest::new(
        expensive_key,
        0..1_000_000,
        OutputEstimate::new(1_000_000, 8_000_000),
    ))?;

    assert_eq!(cheap_grant.reason(), OutputGrantReason::Granted);
    assert_eq!(cheap_grant.range(), &(0..64 * 1024));
    assert_eq!(cheap_grant.visible_frontier(), 128 * 1024);
    assert_eq!(expensive_grant.reason(), OutputGrantReason::Granted);
    assert_eq!(expensive_grant.range(), &(0..8 * 1024));
    assert_eq!(expensive_grant.visible_frontier(), 8 * 1024);
    Ok(())
}

#[test]
fn output_grant_blocks_at_visible_frontier() -> VortexResult<()> {
    let key = GrantKey::new(ROWS, EXPENSIVE_FILTER);
    let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
    grantor.register_domain(ROWS, 1_000_000);
    grantor.set_frontier(key, 8 * 1024)?;

    let grant = grantor.grant(OutputGrantRequest::new(
        key,
        8 * 1024..1_000_000,
        OutputEstimate::new(1_000_000, 8_000_000),
    ))?;

    assert_eq!(grant.reason(), OutputGrantReason::BlockedAtFrontier);
    assert_eq!(grant.range(), &(8 * 1024..8 * 1024));
    Ok(())
}

#[test]
fn output_grant_uses_byte_cap_for_wide_rows() -> VortexResult<()> {
    let key = GrantKey::new(ROWS, EXPENSIVE_FILTER);
    let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
    grantor.register_domain(ROWS, 1_000_000);
    grantor.set_frontier(key, 1_000_000)?;

    let grant = grantor.grant(OutputGrantRequest::new(
        key,
        0..1_000_000,
        OutputEstimate::new(1_000_000, 256_000_000),
    ))?;

    assert_eq!(grant.reason(), OutputGrantReason::Granted);
    assert_eq!(grant.range(), &(0..4096));
    assert_eq!(grant.estimate().rows(), 4096);
    assert_eq!(grant.estimate().bytes(), 1024 * 1024);
    Ok(())
}

#[test]
fn output_grant_frontier_can_advance_after_demand_publication() -> VortexResult<()> {
    let key = GrantKey::new(ROWS, EXPENSIVE_FILTER);
    let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
    grantor.register_domain(ROWS, 1_000_000);
    grantor.set_frontier(key, 0)?;

    let blocked = grantor.grant(OutputGrantRequest::new(
        key,
        0..1_000_000,
        OutputEstimate::new(1_000_000, 8_000_000),
    ))?;
    assert_eq!(blocked.reason(), OutputGrantReason::BlockedAtFrontier);

    grantor.advance_frontier(key, 32 * 1024)?;
    let grant = grantor.grant(OutputGrantRequest::new(
        key,
        0..1_000_000,
        OutputEstimate::new(1_000_000, 8_000_000),
    ))?;

    assert_eq!(grant.reason(), OutputGrantReason::Granted);
    assert_eq!(grant.range(), &(0..32 * 1024));
    Ok(())
}

#[test]
fn output_frontier_grants_sequentially() -> VortexResult<()> {
    let mut frontier = OutputFrontier::unbounded(100).clone_with_offset(10..30);

    let first = frontier.grant_next(8, narrow_output(20))?;
    let second = frontier.grant_next(20, narrow_output(20))?;

    assert_eq!(first.range(), &(0..8));
    assert_eq!(second.range(), &(8..20));
    Ok(())
}

#[test]
fn output_frontier_sideways_clones_have_independent_cursors() -> VortexResult<()> {
    let frontier = OutputFrontier::unbounded(100).clone_with_offset(10..50);
    let mut cheap = frontier.clone_sideways(CHEAP_FILTER);
    let mut expensive = frontier.clone_sideways(EXPENSIVE_FILTER);

    let cheap_grant = cheap.grant_next(16, narrow_output(40))?;
    let expensive_grant = expensive.grant_next(4, narrow_output(40))?;

    assert_eq!(cheap_grant.range(), &(0..16));
    assert_eq!(expensive_grant.range(), &(0..4));
    Ok(())
}

#[test]
fn output_frontier_offset_clone_maps_grants_back_to_local_rows() -> VortexResult<()> {
    let key = GrantKey::new(ROWS, CHEAP_FILTER);
    let mut grantor = OutputGrantor::new(64 * 1024, 1024 * 1024);
    grantor.register_domain(ROWS, 100);
    grantor.set_frontier(key, 35)?;
    let source: Arc<dyn FrontierSource> = Arc::new(parking_lot::Mutex::new(grantor));
    let mut frontier = OutputFrontier::new(source, key, 100).clone_with_offset(20..60);

    let grant = frontier.grant_next(40, narrow_output(40))?;

    assert_eq!(grant.reason(), OutputGrantReason::Granted);
    assert_eq!(grant.range(), &(0..15));
    assert_eq!(grant.visible_frontier(), 15);
    Ok(())
}

#[test]
fn conjunct_controller_releases_distinct_initial_frontiers() -> VortexResult<()> {
    let policy = ConjunctFrontierPolicy::new(128 * 1024, 8 * 1024, 64 * 1024, 1024 * 1024);
    let mut controller = ConjunctFrontierController::new(
        ROWS,
        1_000_000,
        vec![CHEAP_FILTER, EXPENSIVE_FILTER],
        policy,
    )?;

    controller.begin_range(&(0..1_000_000))?;

    assert_eq!(controller.stage_frontier(0)?, 128 * 1024);
    assert_eq!(controller.stage_frontier(1)?, 8 * 1024);

    let leader = controller.grant_for_stage(0, 0..1_000_000, narrow_output(1_000_000))?;
    let dependent = controller.grant_for_stage(1, 0..1_000_000, narrow_output(1_000_000))?;

    assert_eq!(leader.range(), &(0..64 * 1024));
    assert_eq!(dependent.range(), &(0..8 * 1024));
    Ok(())
}

#[test]
fn conjunct_controller_releases_next_stage_to_known_demand_prefix() -> VortexResult<()> {
    let policy = ConjunctFrontierPolicy::new(128 * 1024, 0, 128 * 1024, 1024 * 1024);
    let mut controller = ConjunctFrontierController::new(
        ROWS,
        1_000_000,
        vec![CHEAP_FILTER, EXPENSIVE_FILTER],
        policy,
    )?;
    controller.begin_range(&(0..1_000_000))?;

    let blocked = controller.grant_for_stage(1, 0..1_000_000, narrow_output(1_000_000))?;
    assert_eq!(blocked.reason(), OutputGrantReason::BlockedAtFrontier);

    let mut demand = OrdinalDemand::new(ROWS, 1_000_000);
    demand.publish(0..32 * 1024, Mask::new_true(32 * 1024))?;
    controller.release_after_stage(0, &demand, &(0..1_000_000))?;

    assert_eq!(controller.stage_frontier(1)?, 32 * 1024);
    let grant = controller.grant_for_stage(1, 0..1_000_000, narrow_output(1_000_000))?;
    assert_eq!(grant.reason(), OutputGrantReason::Granted);
    assert_eq!(grant.range(), &(0..32 * 1024));
    Ok(())
}

#[test]
fn conjunct_controller_releases_stage_by_stage() -> VortexResult<()> {
    let policy = ConjunctFrontierPolicy::new(128 * 1024, 0, 128 * 1024, 1024 * 1024);
    let mut controller = ConjunctFrontierController::new(
        ROWS,
        1_000_000,
        vec![CHEAP_FILTER, EXPENSIVE_FILTER, THIRD_FILTER],
        policy,
    )?;
    controller.begin_range(&(0..1_000_000))?;

    let mut first_demand = OrdinalDemand::new(ROWS, 1_000_000);
    first_demand.publish(0..64 * 1024, Mask::new_true(64 * 1024))?;
    controller.release_after_stage(0, &first_demand, &(0..1_000_000))?;

    assert_eq!(controller.stage_frontier(1)?, 64 * 1024);
    assert_eq!(controller.stage_frontier(2)?, 0);

    let mut second_demand = OrdinalDemand::new(ROWS, 1_000_000);
    second_demand.publish(0..16 * 1024, Mask::new_true(16 * 1024))?;
    controller.release_after_stage(1, &second_demand, &(0..1_000_000))?;

    assert_eq!(controller.stage_frontier(2)?, 16 * 1024);
    Ok(())
}

#[test]
fn conjunct_controller_allows_bounded_dependent_speculation() -> VortexResult<()> {
    let policy = ConjunctFrontierPolicy::new(128 * 1024, 4 * 1024, 128 * 1024, 1024 * 1024);
    let mut controller = ConjunctFrontierController::new(
        ROWS,
        1_000_000,
        vec![CHEAP_FILTER, EXPENSIVE_FILTER],
        policy,
    )?;
    controller.begin_range(&(0..1_000_000))?;

    let mut demand = OrdinalDemand::new(ROWS, 1_000_000);
    demand.publish(0..16 * 1024, Mask::new_true(16 * 1024))?;
    controller.release_after_stage(0, &demand, &(0..1_000_000))?;

    assert_eq!(controller.stage_frontier(1)?, 20 * 1024);
    let grant = controller.grant_for_stage(1, 16 * 1024..1_000_000, narrow_output(1_000_000))?;
    assert_eq!(grant.reason(), OutputGrantReason::Granted);
    assert_eq!(grant.range(), &(16 * 1024..20 * 1024));
    Ok(())
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
fn layout_lowering_registers_leaf_work() -> VortexResult<()> {
    let leaf = TestLeaf::new("bool-leaf", bool_dtype(), 100);
    let mut ctx = lower_to_single_scheduler(&leaf, 0..100)?;

    assert_eq!(ctx.lowered_node_count(), 1);
    assert_eq!(ctx.pipeline_count(), 1);
    assert_eq!(ctx.leaf_work_count(), 1);
    assert_eq!(ctx.queued_event_count(), 1);
    assert_eq!(ctx.leaf_work()[0].role(), MorselRole::InformationProducer);
    assert_eq!(ctx.leaf_work()[0].pipeline().index(), 0);

    let report = ctx.drive_to_completion();
    assert_eq!(report.completed_morsels(), 1);
    assert_eq!(report.steps(), 3);
    Ok(())
}

#[test]
fn layout_lowering_prioritizes_information_leaves() -> VortexResult<()> {
    let value: LayoutPlanRef = Arc::new(TestLeaf::new("value", i64_dtype(), 100));
    let info: LayoutPlanRef = Arc::new(TestLeaf::new("info", bool_dtype(), 100));
    let plan = TestContainer::new(vec![Arc::clone(&value), Arc::clone(&info)], 100);
    let mut ctx = lower_to_single_scheduler(&plan, 0..100)?;

    assert_eq!(ctx.leaf_work_count(), 2);
    let info_morsel = ctx.leaf_work()[1].morsel();
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

    assert_eq!(ctx.leaf_work_count(), 2);
    assert_eq!(ctx.leaf_work()[0].local_range(), &(5..10));
    assert_eq!(ctx.leaf_work()[0].global_range(), &(5..10));
    assert_eq!(ctx.leaf_work()[1].local_range(), &(0..15));
    assert_eq!(ctx.leaf_work()[1].global_range(), &(10..25));
    Ok(())
}
