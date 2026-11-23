// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executor-level tests of block fusion, with test encodings: a counting source and ops that add
//! one, widen, or pass their child through without fusing.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use parking_lot::Mutex;
use rstest::rstest;
use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::BLOCK_LEN;
use super::Block;
use super::BlockBuf;
use super::BlockOp;
use super::BlockSource;
use super::SourceBlock;
use super::Written;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::vtable::NotSupported;
use crate::array::vtable::ValidityVTable;
use crate::array::vtable::with_empty_buffers;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::require_spine;
use crate::session::ArraySession;
use crate::test_harness::fusion::fusion_lines;
use crate::test_harness::fusion::fusion_trace;
use crate::test_harness::trace::trace_op;
use crate::validity::Validity;

thread_local! {
    /// How many times this thread asked a test op for its block op.
    static BLOCK_OPS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn dtype(ptype: PType) -> DType {
    DType::Primitive(ptype, Nullability::NonNullable)
}

// ---------------------------------------------------------------------------------------------
// A counting source: rows `0..len` hold `start..start + len` as i32.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Counter;

#[derive(Clone, Debug)]
struct CounterData {
    start: i32,
    /// The block offset its source reports.
    block_offset: usize,
    /// The row ranges its sources were asked to fill.
    fills: Arc<Mutex<Vec<Range<usize>>>>,
}

impl Display for CounterData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "start={}", self.start)
    }
}

impl ArrayHash for CounterData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl ArrayEq for CounterData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.start == other.start
    }
}

impl ValidityVTable<Counter> for Counter {
    fn validity(_array: ArrayView<'_, Counter>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

impl VTable for Counter {
    type TypedArrayData = CounterData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.test.counter");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        _len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(dtype, &self::dtype(PType::I32));
        vortex_ensure_eq!(slots.len(), 0);
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("Counter buffer {idx}")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("Counter slot {idx}")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let start = array.as_view().data().start;
        let len = i32::try_from(array.len()).unwrap_or(i32::MAX);
        Ok(ExecutionResult::done(PrimitiveArray::from_iter(
            start..start + len,
        )))
    }

    fn block_source(
        array: ArrayView<'_, Self>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn BlockSource>>> {
        let data = array.data();
        Ok(Some(Box::new(CounterSource {
            start: data.start,
            block_offset: data.block_offset,
            fills: Arc::clone(&data.fills),
        })))
    }
}

struct CounterSource {
    start: i32,
    block_offset: usize,
    fills: Arc<Mutex<Vec<Range<usize>>>>,
}

impl BlockSource for CounterSource {
    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &'s mut BlockBuf,
    ) -> VortexResult<SourceBlock<'s>> {
        self.fills.lock().push(rows.clone());
        let out = scratch.out::<i32>(rows.len());
        for (out, row) in out.iter_mut().zip(rows.clone()) {
            out.write(self.start + i32::try_from(row).unwrap_or(i32::MAX));
        }
        // SAFETY: the loop wrote every value of `out`.
        Ok(unsafe { SourceBlock::scratch(0, rows.len()) })
    }

    fn block_offset(&self) -> usize {
        self.block_offset
    }
}

fn counter(
    start: i32,
    len: usize,
    block_offset: usize,
) -> (ArrayRef, Arc<Mutex<Vec<Range<usize>>>>) {
    let fills = Arc::new(Mutex::new(Vec::new()));
    let array = Array::<Counter>::try_from_parts(ArrayParts::new(
        Counter,
        dtype(PType::I32),
        len,
        CounterData {
            start,
            block_offset,
            fills: Arc::clone(&fills),
        },
        smallvec![],
    ))
    .unwrap_or_else(|e| vortex_panic!("counter: {e}"))
    .into_array();
    (array, fills)
}

// ---------------------------------------------------------------------------------------------
// Test ops over one child.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TestOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestOpKind {
    /// Adds one; may run in place.
    AddOne,
    /// Adds one, but never in place.
    AddOneCopying,
    /// Widens i32 to i64.
    Widen,
    /// Returns its child, asking for it with a plain `ExecuteSlot`, so it never fuses.
    Pass,
}

#[derive(Clone, Debug)]
struct TestOpData(TestOpKind);

impl Display for TestOpData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            TestOpKind::AddOne => "add-one",
            TestOpKind::AddOneCopying => "add-one-copying",
            TestOpKind::Widen => "widen",
            TestOpKind::Pass => "pass",
        })
    }
}

impl ArrayHash for TestOpData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl ArrayEq for TestOpData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.0 == other.0
    }
}

impl ValidityVTable<TestOp> for TestOp {
    fn validity(_array: ArrayView<'_, TestOp>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

impl VTable for TestOp {
    type TypedArrayData = TestOpData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.test.op");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        _dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), 1);
        if let Some(child) = &slots[0] {
            vortex_ensure_eq!(child.len(), len);
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("TestOp buffer {idx}")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, _idx: usize) -> String {
        "child".to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let kind = array.as_view().data().0;
        let child = array.slots()[0]
            .clone()
            .unwrap_or_else(|| vortex_panic!("child"));
        if kind == TestOpKind::Pass {
            if !child.is::<Primitive>() {
                return Ok(ExecutionResult::execute_slot::<Primitive>(array, 0));
            }
            return Ok(ExecutionResult::done(child));
        }
        let array = require_spine!(array, child, 0 => Primitive);
        let child = array.slots()[0]
            .clone()
            .unwrap_or_else(|| vortex_panic!("child"))
            .downcast::<Primitive>();
        Ok(ExecutionResult::done(match (kind, child.ptype()) {
            (TestOpKind::Widen, _) => {
                PrimitiveArray::from_iter(child.as_slice::<i32>().iter().map(|&v| i64::from(v)))
                    .into_array()
            }
            (_, PType::I64) => {
                PrimitiveArray::from_iter(child.as_slice::<i64>().iter().map(|v| v + 1))
                    .into_array()
            }
            _ => PrimitiveArray::from_iter(child.as_slice::<i32>().iter().map(|v| v + 1))
                .into_array(),
        }))
    }

    fn block_op(
        array: ArrayView<'_, Self>,
        _slot_idx: usize,
        child_dtype: &DType,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn BlockOp>>> {
        BLOCK_OPS_BUILT.with(|count| count.set(count.get() + 1));
        let wide = child_dtype == &dtype(PType::I64);
        Ok(match array.data().0 {
            TestOpKind::AddOne => Some(Box::new(AddOne {
                in_place: true,
                wide,
            })),
            TestOpKind::AddOneCopying => Some(Box::new(AddOne {
                in_place: false,
                wide,
            })),
            TestOpKind::Widen => Some(Box::new(Widen)),
            TestOpKind::Pass => None,
        })
    }
}

struct AddOne {
    in_place: bool,
    /// Whether it adds to i64 values rather than i32.
    wide: bool,
}

impl BlockOp for AddOne {
    fn process<'a>(&mut self, _start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        Ok(if self.wide {
            block.map(|v: i64| v + 1)
        } else {
            block.map(|v: i32| v + 1)
        })
    }

    fn supports_in_place(&self) -> bool {
        self.in_place
    }
}

struct Widen;

impl BlockOp for Widen {
    fn process<'a>(&mut self, _start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        Ok(block.map(|v: i32| i64::from(v)))
    }
}

fn op(kind: TestOpKind, child: ArrayRef) -> ArrayRef {
    let dtype = match kind {
        TestOpKind::Widen => dtype(PType::I64),
        _ => child.dtype().clone(),
    };
    let len = child.len();
    Array::<TestOp>::try_from_parts(ArrayParts::new(
        TestOp,
        dtype,
        len,
        TestOpData(kind),
        smallvec![Some(child)],
    ))
    .unwrap_or_else(|e| vortex_panic!("op: {e}"))
    .into_array()
}

fn add(child: ArrayRef) -> ArrayRef {
    op(TestOpKind::AddOne, child)
}

// ---------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------

/// Execute `array` with fusion off and on, check they agree, and return the fused run's fusion
/// steps.
fn fused(array: ArrayRef) -> VortexResult<String> {
    fusion_trace(array, &SESSION)
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

/// Two adds over a counter: down by spines to the counter, which starts a program on the way
/// back up; each add extends it, and the root drives it. The first add runs in place on the
/// source's scratch block, and the last writes the output.
#[test]
fn chain_builds_on_the_way_up_and_drives_at_the_root() -> VortexResult<()> {
    let (counter, _) = counter(10, 3000, 0);
    assert_eq!(
        fused(add(add(counter)))?,
        "ExecuteSpine slot=0 parent=vortex.test.op(i32, len=3000) child=vortex.test.op(i32, len=3000)
ExecuteSpine slot=0 parent=vortex.test.op(i32, len=3000) child=vortex.test.counter(i32, len=3000)
fuse start array=vortex.test.counter(i32, len=3000)
fuse extend parent=vortex.test.op(i32, len=3000) ops=1
fuse extend parent=vortex.test.op(i32, len=3000) ops=2
drive ops=2 len=3000
block_op 0 in_place i32->i32 scratch0 -> scratch0
block_op 1 copy i32->i32 scratch0 -> output"
    );
    Ok(())
}

/// A widening op cannot overwrite its input, so it copies into the other scratch block, and the
/// ops after it work there.
#[test]
fn width_change_copies_between_scratch_blocks() -> VortexResult<()> {
    let (counter, _) = counter(-5, 2500, 0);
    let array = add(op(TestOpKind::Widen, add(counter)));
    let trace = fused(array)?;
    assert!(
        trace.ends_with(
            "drive ops=3 len=2500
block_op 0 in_place i32->i32 scratch0 -> scratch0
block_op 1 copy i32->i64 scratch0 -> scratch1
block_op 2 copy i64->i64 scratch1 -> output"
        ),
        "{trace}"
    );
    Ok(())
}

/// An op that does not support in-place blocks always copies, alternating scratch blocks.
#[test]
fn op_without_in_place_support_copies() -> VortexResult<()> {
    let (counter, _) = counter(0, 2000, 0);
    let copying = |child| op(TestOpKind::AddOneCopying, child);
    let trace = fused(copying(copying(copying(counter))))?;
    assert!(
        trace.ends_with(
            "drive ops=3 len=2000
block_op 0 copy i32->i32 scratch0 -> scratch1
block_op 1 copy i32->i32 scratch1 -> scratch0
block_op 2 copy i32->i32 scratch0 -> output"
        ),
        "{trace}"
    );
    Ok(())
}

/// A parent that asks for its child with a plain `ExecuteSlot` cannot extend the program, so the
/// fused child is driven when it is popped into that parent, before the parent runs.
#[test]
fn non_spine_parent_drives_the_chain_below_it() -> VortexResult<()> {
    let (counter, _) = counter(1, 1500, 0);
    let array = op(TestOpKind::Pass, add(add(counter)));
    assert_eq!(
        fused(array)?,
        "ExecuteSlot slot=0 parent=vortex.test.op(i32, len=1500) child=vortex.test.op(i32, len=1500)
ExecuteSpine slot=0 parent=vortex.test.op(i32, len=1500) child=vortex.test.op(i32, len=1500)
ExecuteSpine slot=0 parent=vortex.test.op(i32, len=1500) child=vortex.test.counter(i32, len=1500)
fuse start array=vortex.test.counter(i32, len=1500)
fuse extend parent=vortex.test.op(i32, len=1500) ops=1
fuse extend parent=vortex.test.op(i32, len=1500) ops=2
drive ops=2 len=1500
block_op 0 in_place i32->i32 scratch0 -> scratch0
block_op 1 copy i32->i32 scratch0 -> output"
    );
    Ok(())
}

/// The executor cuts blocks on the source's grid, so a source whose first chunk starts part way
/// is asked for a short first block and then whole blocks.
#[rstest]
#[case::aligned(0, 3000, vec![0..1024, 1024..2048, 2048..3000])]
#[case::offset(300, 3000, vec![0..724, 724..1748, 1748..2772, 2772..3000])]
#[case::shorter_than_first_block(1000, 20, vec![0..20])]
#[case::exact_blocks(0, 2048, vec![0..1024, 1024..2048])]
#[case::one_row(0, 1, vec![0..1])]
fn blocks_follow_the_source_grid(
    #[case] block_offset: usize,
    #[case] len: usize,
    #[case] expected: Vec<Range<usize>>,
) -> VortexResult<()> {
    let (counter, fills) = counter(7, len, block_offset);
    fused(add(counter))?;
    assert_eq!(*fills.lock(), expected);
    assert!(expected.iter().all(|rows| rows.len() <= BLOCK_LEN));
    Ok(())
}

/// A canonical leaf is a block source like any other. The program decodes in the leaf's own
/// buffer when the executor holds the only reference to it, and copies the leaf into scratch when
/// another owner shares it.
///
/// A primitive child of a spine parent is already canonical, so the executor never descends to
/// it: the innermost op runs its own `execute`, and its result is the leaf of the ops above.
#[test]
fn canonical_leaf_in_place_only_when_owned() -> VortexResult<()> {
    let leaf = || PrimitiveArray::from_iter(0..3000i32).into_array();
    assert_eq!(
        fused(add(add(add(leaf()))))?,
        "ExecuteSpine slot=0 parent=vortex.test.op(i32, len=3000) child=vortex.test.op(i32, len=3000)
ExecuteSpine slot=0 parent=vortex.test.op(i32, len=3000) child=vortex.test.op(i32, len=3000)
fuse start array=vortex.primitive(i32, len=3000)
fuse extend parent=vortex.test.op(i32, len=3000) ops=1
fuse extend parent=vortex.test.op(i32, len=3000) ops=2
drive ops=2 len=3000
block_op 0 in_place i32->i32 leaf -> leaf
block_op 1 in_place i32->i32 leaf -> leaf"
    );

    // `Pass` returns the shared leaf itself, so the leaf reaches the ops above still shared.
    let shared = leaf();
    let mut ctx = SESSION.create_execution_ctx();
    let traced = trace_op(|| {
        add(add(op(TestOpKind::Pass, shared.clone()))).execute::<PrimitiveArray>(&mut ctx)
    })?;
    assert_eq!(
        fusion_lines(&traced.trace.to_string()),
        "ExecuteSpine slot=0 parent=vortex.test.op(i32, len=3000) child=vortex.test.op(i32, len=3000)
ExecuteSpine slot=0 parent=vortex.test.op(i32, len=3000) child=vortex.test.op(i32, len=3000)
fuse start array=vortex.primitive(i32, len=3000)
fuse extend parent=vortex.test.op(i32, len=3000) ops=1
fuse extend parent=vortex.test.op(i32, len=3000) ops=2
drive ops=2 len=3000
block_op 0 copy i32->i32 source -> scratch0
block_op 1 copy i32->i32 scratch0 -> output"
    );
    // The shared leaf still holds its own values.
    assert_arrays_eq!(shared, leaf(), &mut ctx);
    assert_arrays_eq!(
        traced.output,
        PrimitiveArray::from_iter(2..3002i32),
        &mut ctx
    );
    Ok(())
}

#[test]
fn disabled_fusion_runs_level_wise() -> VortexResult<()> {
    let (counter, fills) = counter(0, 2000, 0);
    let mut ctx = SESSION.create_execution_ctx().with_block_fusion(false);
    let traced = trace_op(|| add(add(counter)).execute::<PrimitiveArray>(&mut ctx))?;
    assert!(fills.lock().is_empty());
    assert!(!traced.trace.to_string().contains("fuse "));
    Ok(())
}

#[test]
fn empty_arrays_do_not_fuse() -> VortexResult<()> {
    let (counter, _) = counter(0, 0, 0);
    // Decided on the way down: an empty spine runs as a plain slot.
    assert_eq!(
        fused(add(add(counter)))?,
        "ExecuteSlot slot=0 parent=vortex.test.op(i32, len=0) child=vortex.test.op(i32, len=0)
ExecuteSlot slot=0 parent=vortex.test.op(i32, len=0) child=vortex.test.counter(i32, len=0)"
    );
    Ok(())
}

/// Matches any array but the test op: a target `execute_until` reaches as soon as a test op is
/// fused away.
struct NotTestOp;

impl crate::matcher::Matcher for NotTestOp {
    type Match<'a> = &'a ArrayRef;

    fn matches(array: &ArrayRef) -> bool {
        !array.is::<TestOp>()
    }

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        Self::matches(array).then_some(array)
    }
}

/// A fused array is driven before the target is checked, so it never leaves `execute_until`,
/// even for a target that matches it.
#[test]
fn fused_arrays_never_leave_execute_until() -> VortexResult<()> {
    let (counter, _) = counter(3, 2000, 0);
    let mut ctx = SESSION.create_execution_ctx();
    let output = add(add(counter)).execute_until::<NotTestOp>(&mut ctx)?;
    assert!(output.is::<Primitive>(), "{}", output.encoding_id());
    assert_arrays_eq!(output, PrimitiveArray::from_iter(5..2005i32), &mut ctx);
    Ok(())
}

#[derive(Debug)]
struct DeclineKernel;

impl crate::kernel::ExecuteParentKernel<Primitive> for DeclineKernel {
    type Parent = TestOp;

    fn execute_parent(
        &self,
        _array: ArrayView<'_, Primitive>,
        _parent: <Self::Parent as crate::matcher::Matcher>::Match<'_>,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(None)
    }
}

/// A parent with execute-parent kernels of its own is not fused, so the kernels still probe it
/// once its child is canonical.
#[test]
fn parents_with_kernels_are_not_fused() -> VortexResult<()> {
    use crate::optimizer::kernels::ArrayKernelsExt;

    let session = VortexSession::empty().with::<ArraySession>();
    session
        .kernels()
        .register_execute_parent_kernel(TestOp.id(), Primitive, DeclineKernel);
    let (counter, _) = counter(0, 2000, 0);
    let mut ctx = session.create_execution_ctx();
    let traced = trace_op(|| add(add(counter)).execute::<PrimitiveArray>(&mut ctx))?;
    // Decided on the way down: neither spine request can fuse, so both run as plain slots.
    assert_eq!(
        fusion_lines(&traced.trace.to_string()),
        "ExecuteSlot slot=0 parent=vortex.test.op(i32, len=2000) child=vortex.test.op(i32, len=2000)
ExecuteSlot slot=0 parent=vortex.test.op(i32, len=2000) child=vortex.test.counter(i32, len=2000)"
    );
    assert_arrays_eq!(
        traced.output,
        PrimitiveArray::from_iter(2..2002i32),
        &mut ctx
    );
    Ok(())
}

/// An op is built once per frame. A child that is not a leaf yet executes while the op its parent
/// built waits in the frame, and the canonical result, a leaf, starts the chain with that op.
#[test]
fn op_waits_in_its_frame_for_the_leaf() -> VortexResult<()> {
    let leaf = PrimitiveArray::from_iter(0..3000i32).into_array();
    let array = add(op(TestOpKind::Pass, leaf));
    let before = BLOCK_OPS_BUILT.with(std::cell::Cell::get);
    let trace = fused(array)?;
    // `fused` also runs the array with fusion off, which builds no op.
    assert_eq!(BLOCK_OPS_BUILT.with(std::cell::Cell::get) - before, 1);
    assert_eq!(
        trace,
        "ExecuteSpine slot=0 parent=vortex.test.op(i32, len=3000) child=vortex.test.op(i32, len=3000)
fuse start array=vortex.primitive(i32, len=3000)
fuse extend parent=vortex.test.op(i32, len=3000) ops=1
drive ops=1 len=3000
block_op 0 in_place i32->i32 leaf -> leaf"
    );
    Ok(())
}
