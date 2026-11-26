// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The block runtime, with test sources and ops, checked against level-wise reference values and
//! against the trace of how each op ran over its first block.

use std::marker::PhantomData;
use std::ops::Add;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::BLOCK_LEN;
use super::Block;
use super::BlockBuf;
use super::BlockOp;
use super::BlockPatches;
use super::BlockProgram;
use super::BlockSource;
use super::BlockValidity;
use super::PrimitiveSource;
use super::SourceBlock;
use super::Written;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::patches::Patches;
use crate::session::ArraySession;
use crate::test_harness::trace::trace_op;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn ctx() -> ExecutionCtx {
    SESSION.create_execution_ctx()
}

/// The row ranges a source was asked to fill.
type Fills = Arc<Mutex<Vec<Range<usize>>>>;

/// Emits row `r` as `r as i32` into its scratch, and records the rows it was asked for.
struct Counter {
    block_offset: usize,
    validity: Mask,
    fills: Fills,
}

fn counter(validity: Mask, block_offset: usize) -> (Box<dyn BlockSource>, Fills) {
    let fills = Fills::default();
    let source = Counter {
        block_offset,
        validity,
        fills: Arc::clone(&fills),
    };
    (Box::new(source), fills)
}

impl BlockSource for Counter {
    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        _validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>> {
        self.fills.lock().push(rows.clone());
        for (out, row) in scratch
            .values_mut::<i32>(rows.len())
            .iter_mut()
            .zip(rows.clone())
        {
            *out = i32::try_from(row)?;
        }
        Ok(SourceBlock::Scratch {
            offset: 0,
            len: rows.len(),
        })
    }

    fn block_offset(&self) -> usize {
        self.block_offset
    }
}

/// Adds one to each value of `T`, in place when `in_place`.
struct AddOne<T> {
    in_place: bool,
    _type: PhantomData<fn(T)>,
}

fn add<T: NativePType + Add<Output = T> + From<u8>>(in_place: bool) -> Box<dyn BlockOp> {
    Box::new(AddOne::<T> {
        in_place,
        _type: PhantomData,
    })
}

impl<T: NativePType + Add<Output = T> + From<u8>> BlockOp for AddOne<T> {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        Ok(block.map(|v: T| v + <T as From<u8>>::from(1)))
    }

    fn supports_in_place(&self) -> bool {
        self.in_place
    }
}

struct Widen;

impl BlockOp for Widen {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        Ok(block.map(|v: i32| i64::from(v)))
    }
}

/// Converts i32 to f32, which keeps the width, so it may run in place.
struct ToFloat;

impl BlockOp for ToFloat {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        #[expect(clippy::cast_precision_loss, reason = "the test values are small")]
        Ok(block.map(|v: i32| v as f32))
    }
}

/// Nulls every row whose index is a multiple of its count, keeping values, one row at a time.
struct NullEvery(usize);

impl BlockOp for NullEvery {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        let start = block.start();
        let Block {
            values,
            mut validity,
            ..
        } = block;
        for row in 0..validity.len {
            if (start + row).is_multiple_of(self.0) {
                validity.set(row, false);
            }
        }
        Ok(values.map(|v: i32| v))
    }
}

/// Intersects the block's validity with the same rows of a mask, a word at a time, as an op
/// over a materialized side input does.
struct AndWith(Mask);

impl BlockOp for AndWith {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        let start = block.start();
        let Block {
            values,
            mut validity,
            ..
        } = block;
        let len = validity.len;
        for (i, word) in validity.words_mut().iter_mut().enumerate() {
            let rows = start + i * 64..start + (i * 64 + 64).min(len);
            let side = rows.enumerate().fold(0u64, |side, (bit, row)| {
                side | (u64::from(self.0.value(row)) << bit)
            });
            *word &= side;
        }
        Ok(values.map(|v: i32| v))
    }
}

/// Applies patches to i32 values.
struct Patch(BlockPatches<i32>);

impl BlockOp for Patch {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        let start = block.start();
        let Block {
            values,
            mut validity,
            ..
        } = block;
        let mut written = values.map(|v: i32| v);
        self.0.apply(start, written.as_mut_slice(), &mut validity);
        Ok(written)
    }
}

/// Run `program` into an array of `nullability`, and return it with the program's lines of the
/// trace.
fn run(program: BlockProgram, nullability: Nullability) -> VortexResult<(PrimitiveArray, String)> {
    let mut ctx = ctx();
    let traced = trace_op(|| program.run(nullability, &mut ctx))?;
    let lines = traced
        .trace
        .to_string()
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("drive ") || line.starts_with("block_op"))
        .collect::<Vec<_>>()
        .join("\n");
    Ok((traced.output, lines))
}

fn program(
    source: Box<dyn BlockSource>,
    len: usize,
    ops: Vec<(Box<dyn BlockOp>, PType)>,
) -> BlockProgram {
    let mut program = BlockProgram::new(source, PType::I32, len);
    for (op, ptype) in ops {
        program.push(op, ptype);
    }
    program
}

/// A mask of `len` rows: all valid, all null, or a drawn pattern repeated from a drawn bit offset
/// into its buffer.
fn draw_mask(tc: &TestCase, len: usize) -> Mask {
    match tc.draw(gs::integers::<u8>().max_value(2)) {
        0 => Mask::new_true(len),
        1 => Mask::new_false(len),
        _ => {
            let pattern: Vec<bool> = tc.draw(gs::vecs(gs::booleans()).min_size(1).max_size(70));
            let offset = tc.draw(gs::integers::<usize>().max_value(9));
            let bits = BitBuffer::collect_bool(offset + len, |i| pattern[i % pattern.len()]);
            Mask::from_buffer(bits.slice(offset..offset + len))
        }
    }
}

/// Any chain of adds, in place or not, then optionally a widening or a same-width type change,
/// over any length, block grid and validity, gives the values and validity of applying it row by
/// row, and the source is asked for the program's blocks: consecutive, at most `BLOCK_LEN` rows,
/// and cut on the source's grid.
#[hegel::test]
fn programs_match_row_by_row(tc: TestCase) {
    let len = tc.draw(gs::integers::<usize>().max_value(3 * BLOCK_LEN + 100));
    let block_offset = tc.draw(gs::integers::<usize>().max_value(BLOCK_LEN - 1));
    let mask = draw_mask(&tc, len);
    let adds: Vec<bool> = tc.draw(gs::vecs(gs::booleans()).max_size(4));
    let last = tc.draw(gs::integers::<u8>().max_value(2));

    let (source, fills) = counter(mask.clone(), block_offset);
    let mut ops: Vec<(Box<dyn BlockOp>, PType)> = adds
        .iter()
        .map(|&in_place| (add::<i32>(in_place), PType::I32))
        .collect();
    match last {
        0 => ops.push((Box::new(Widen), PType::I64)),
        1 => ops.push((Box::new(ToFloat), PType::F32)),
        _ => {}
    }
    let (output, _) = run(program(source, len, ops), Nullability::Nullable).vortex_expect("run");

    let value = |row: usize| i32::try_from(row + adds.len()).vortex_expect("small");
    let valid = |row: usize| mask.value(row);
    #[expect(clippy::cast_precision_loss, reason = "the test values are small")]
    let expected = match last {
        0 => PrimitiveArray::from_option_iter(
            (0..len).map(|r| valid(r).then(|| i64::from(value(r)))),
        ),
        1 => PrimitiveArray::from_option_iter((0..len).map(|r| valid(r).then(|| value(r) as f32))),
        _ => PrimitiveArray::from_option_iter((0..len).map(|r| valid(r).then(|| value(r)))),
    };
    assert_arrays_eq!(output, expected, &mut ctx());

    let fills = fills.lock();
    assert_eq!(fills.iter().map(Range::len).sum::<usize>(), len);
    for (i, rows) in fills.iter().enumerate() {
        assert!(rows.len() <= BLOCK_LEN && !rows.is_empty(), "{rows:?}");
        if i > 0 {
            assert_eq!(rows.start, fills[i - 1].end);
            assert_eq!(
                (block_offset + rows.start) % BLOCK_LEN,
                0,
                "{rows:?} off the grid"
            );
        }
    }
}

/// Ops that null rows one at a time, or intersect a block's validity a word at a time, give the
/// source's validity with every op's nulls, whatever the grid and source validity.
#[hegel::test]
fn validity_edits_match_row_by_row(tc: TestCase) {
    let len = tc.draw(gs::integers::<usize>().max_value(3 * BLOCK_LEN + 100));
    let block_offset = tc.draw(gs::integers::<usize>().max_value(BLOCK_LEN - 1));
    let source_mask = draw_mask(&tc, len);
    let side = draw_mask(&tc, len);
    let every = tc.draw(gs::integers::<usize>().min_value(1).max_value(200));

    let (source, _) = counter(source_mask.clone(), block_offset);
    let ops: Vec<(Box<dyn BlockOp>, PType)> = vec![
        (Box::new(NullEvery(every)), PType::I32),
        (add::<i32>(true), PType::I32),
        (Box::new(AndWith(side.clone())), PType::I32),
    ];
    let (output, _) = run(program(source, len, ops), Nullability::Nullable).vortex_expect("run");

    let expected = PrimitiveArray::from_option_iter((0..len).map(|r| {
        (source_mask.value(r) && !r.is_multiple_of(every) && side.value(r))
            .then(|| i32::try_from(r + 1).vortex_expect("small"))
    }));
    assert_arrays_eq!(output, expected, &mut ctx());
}

/// How ops share the two scratch blocks: an op overwrites its block when it keeps the width and
/// allows it, otherwise it copies into the other block, and the last op writes the output.
#[rstest]
#[case::in_place_then_output(
    vec![(add::<i32>(true), PType::I32),
         (add::<i32>(true), PType::I32)],
    "block_op 0 in_place i32->i32 scratch0 -> scratch0
block_op 1 copy i32->i32 scratch0 -> output"
)]
#[case::width_change(
    vec![(add::<i32>(true), PType::I32),
         (Box::new(Widen), PType::I64),
         (add::<i64>(true), PType::I64)],
    "block_op 0 in_place i32->i32 scratch0 -> scratch0
block_op 1 copy i32->i64 scratch0 -> scratch1
block_op 2 copy i64->i64 scratch1 -> output"
)]
#[case::type_change_of_one_width(
    vec![(add::<i32>(true), PType::I32),
         (Box::new(ToFloat), PType::F32),
         (add::<f32>(true), PType::F32)],
    "block_op 0 in_place i32->i32 scratch0 -> scratch0
block_op 1 in_place i32->f32 scratch0 -> scratch0
block_op 2 copy f32->f32 scratch0 -> output"
)]
#[case::no_in_place_support(
    vec![(add::<i32>(false), PType::I32),
         (add::<i32>(false), PType::I32),
         (add::<i32>(false), PType::I32)],
    "block_op 0 copy i32->i32 scratch0 -> scratch1
block_op 1 copy i32->i32 scratch1 -> scratch0
block_op 2 copy i32->i32 scratch0 -> output"
)]
fn ops_share_scratch(
    #[case] ops: Vec<(Box<dyn BlockOp>, PType)>,
    #[case] expected: &str,
) -> VortexResult<()> {
    let n = ops.len();
    let (source, _) = counter(Mask::new_true(2500), 0);
    let (_, trace) = run(program(source, 2500, ops), Nullability::NonNullable)?;
    assert_eq!(trace, format!("drive ops={n} len=2500\n{expected}"));
    Ok(())
}

/// A canonical leaf decodes in its own buffer when the program holds the only reference to it,
/// and copies out of it, leaving it unchanged, when another owner shares it.
#[test]
fn canonical_leaf_decodes_in_place_only_when_owned() -> VortexResult<()> {
    let two_adds = |source| {
        program(
            source,
            3000,
            vec![
                (add::<i32>(true), PType::I32),
                (add::<i32>(true), PType::I32),
            ],
        )
    };
    let owned = PrimitiveSource::boxed(&PrimitiveArray::from_iter(0..3000i32), &mut ctx())?;
    let (output, trace) = run(two_adds(owned), Nullability::NonNullable)?;
    assert_arrays_eq!(output, PrimitiveArray::from_iter(2..3002i32), &mut ctx());
    assert_eq!(
        trace,
        "drive ops=2 len=3000
block_op 0 in_place i32->i32 leaf -> leaf
block_op 1 in_place i32->i32 leaf -> leaf"
    );

    let shared = PrimitiveArray::from_iter(0..3000i32);
    let (output, trace) = run(
        two_adds(PrimitiveSource::boxed(&shared, &mut ctx())?),
        Nullability::NonNullable,
    )?;
    assert_arrays_eq!(output, PrimitiveArray::from_iter(2..3002i32), &mut ctx());
    assert_eq!(
        trace,
        "drive ops=2 len=3000
block_op 0 copy i32->i32 source -> scratch0
block_op 1 copy i32->i32 scratch0 -> output"
    );
    assert_arrays_eq!(shared, PrimitiveArray::from_iter(0..3000i32), &mut ctx());
    Ok(())
}

/// Uniform validity needs no bitmap: the output is non-nullable, all valid or all null.
#[rstest]
#[case::non_nullable(Mask::new_true(2000), Nullability::NonNullable, Validity::NonNullable)]
#[case::all_valid(Mask::new_true(2000), Nullability::Nullable, Validity::AllValid)]
#[case::all_null(Mask::new_false(2000), Nullability::Nullable, Validity::AllInvalid)]
fn uniform_validity_needs_no_bitmap(
    #[case] mask: Mask,
    #[case] nullability: Nullability,
    #[case] expected: Validity,
) -> VortexResult<()> {
    let (source, _) = counter(mask, 0);
    let ops: Vec<(Box<dyn BlockOp>, PType)> = vec![(add::<i32>(true), PType::I32)];
    let (output, _) = run(program(source, 2000, ops), nullability)?;
    assert_eq!(
        std::mem::discriminant(&output.validity()?),
        std::mem::discriminant(&expected)
    );
    Ok(())
}

/// A non-nullable output cannot hold the nulls an op produced.
#[test]
fn non_nullable_output_rejects_nulls() {
    let (source, _) = counter(Mask::new_true(100), 0);
    let ops: Vec<(Box<dyn BlockOp>, PType)> = vec![(Box::new(NullEvery(5)), PType::I32)];
    assert!(run(program(source, 100, ops), Nullability::NonNullable).is_err());
}

/// Patches overwrite their rows' values and validity, block by block, null patch values
/// included.
#[test]
fn patches_set_values_and_validity() -> VortexResult<()> {
    let mut ctx = ctx();
    let len = 3000;
    let patches = Patches::new(
        len,
        0,
        buffer![0u32, 1023, 1024, 2999].into_array(),
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(300), Some(400)]).into_array(),
        None,
        &mut ctx,
    )?;
    let patches = BlockPatches::try_new(&patches, &mut ctx, |_, value: i32| value)?;
    let (source, _) = counter(Mask::new_true(len), 0);
    let ops: Vec<(Box<dyn BlockOp>, PType)> = vec![(Box::new(Patch(patches)), PType::I32)];
    let (output, _) = run(program(source, len, ops), Nullability::Nullable)?;
    let expected = PrimitiveArray::from_option_iter((0..len).map(|row| match row {
        0 => Some(100),
        1023 => None,
        1024 => Some(300),
        2999 => Some(400),
        _ => Some(i32::try_from(row).unwrap_or(i32::MAX)),
    }));
    assert_arrays_eq!(output, expected, &mut ctx);
    Ok(())
}
