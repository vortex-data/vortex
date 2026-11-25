// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The block runtime, with test sources and ops: each program's values and validity, and how
//! each op ran over its first block, from the trace.

use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use parking_lot::Mutex;
use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
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

/// Emits `start + row` as i32 into its scratch, and records the rows it was asked for.
struct Counter {
    start: i32,
    block_offset: usize,
    fills: Fills,
}

/// The row ranges a source was asked to fill.
type Fills = Arc<Mutex<Vec<Range<usize>>>>;

fn counter(start: i32, block_offset: usize) -> (Box<dyn BlockSource>, Fills) {
    let fills = Arc::new(Mutex::new(Vec::new()));
    let source = Counter {
        start,
        block_offset,
        fills: Arc::clone(&fills),
    };
    (Box::new(source), fills)
}

impl BlockSource for Counter {
    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        _validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>> {
        self.fills.lock().push(rows.clone());
        let out = scratch.values_mut::<i32>(rows.len());
        for (out, row) in out.iter_mut().zip(rows.clone()) {
            *out = self.start + i32::try_from(row).unwrap_or(i32::MAX);
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

/// Adds one to each value of `T`.
struct AddOne<T> {
    in_place: bool,
    _type: PhantomData<fn(T)>,
}

fn add_one<T: NativePType + std::ops::Add<Output = T> + From<u8>>(
    in_place: bool,
) -> Box<dyn BlockOp> {
    Box::new(AddOne::<T> {
        in_place,
        _type: PhantomData,
    })
}

impl<T: NativePType + std::ops::Add<Output = T> + From<u8>> BlockOp for AddOne<T> {
    fn process<'a>(&mut self, _start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        Ok(block.map(|v: T| v + <T as From<u8>>::from(1)))
    }

    fn supports_in_place(&self) -> bool {
        self.in_place
    }
}

/// Widens i32 to i64.
struct Widen;

impl BlockOp for Widen {
    fn process<'a>(&mut self, _start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        Ok(block.map(|v: i32| i64::from(v)))
    }
}

/// Nulls every row whose index is a multiple of `every`, keeping values.
struct NullEvery(usize);

impl BlockOp for NullEvery {
    fn process<'a>(&mut self, start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        let Block {
            values,
            mut validity,
        } = block;
        for row in 0..validity.len() {
            if (start + row).is_multiple_of(self.0) {
                validity.set(row, false);
            }
        }
        Ok(values.map(|v: i32| v))
    }
}

/// Applies patches to i32 values.
struct Patch(BlockPatches<i32>);

impl BlockOp for Patch {
    fn process<'a>(&mut self, start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        let Block {
            values,
            mut validity,
        } = block;
        let mut written = values.map(|v: i32| v);
        self.0.apply(start, written.as_mut_slice(), &mut validity);
        Ok(written)
    }
}

/// Run `program` over rows with `validity` into an array of `nullability`, and return it with
/// the program's lines of the trace.
fn run(
    program: BlockProgram,
    validity: &Mask,
    nullability: Nullability,
) -> VortexResult<(PrimitiveArray, String)> {
    let mut ctx = ctx();
    let traced = trace_op(|| program.run(validity, nullability, &mut ctx))?;
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

fn counter_program(start: i32, len: usize, ops: Vec<(Box<dyn BlockOp>, PType)>) -> BlockProgram {
    let (source, _) = counter(start, 0);
    let mut program = BlockProgram::new(source, PType::I32, len);
    for (op, ptype) in ops {
        program.push(op, ptype);
    }
    program
}

/// The first op overwrites the source's scratch block, and the last writes the output.
#[test]
fn ops_run_in_scratch_and_the_last_writes_the_output() -> VortexResult<()> {
    let program = counter_program(
        10,
        3000,
        vec![
            (add_one::<i32>(true), PType::I32),
            (add_one::<i32>(true), PType::I32),
        ],
    );
    let (output, trace) = run(program, &Mask::new_true(3000), Nullability::NonNullable)?;
    assert_arrays_eq!(output, PrimitiveArray::from_iter(12..3012i32), &mut ctx());
    assert_eq!(
        trace,
        "drive ops=2 len=3000
block_op 0 in_place i32->i32 scratch0 -> scratch0
block_op 1 copy i32->i32 scratch0 -> output"
    );
    Ok(())
}

/// An op that changes the width copies into the other scratch block, and the ops after it work
/// there.
#[test]
fn width_change_copies_between_scratch_blocks() -> VortexResult<()> {
    let program = counter_program(
        -5,
        2500,
        vec![
            (add_one::<i32>(true), PType::I32),
            (Box::new(Widen), PType::I64),
            (add_one::<i64>(true), PType::I64),
        ],
    );
    let (output, trace) = run(program, &Mask::new_true(2500), Nullability::NonNullable)?;
    assert_arrays_eq!(output, PrimitiveArray::from_iter(-3..2497i64), &mut ctx());
    assert_eq!(
        trace,
        "drive ops=3 len=2500
block_op 0 in_place i32->i32 scratch0 -> scratch0
block_op 1 copy i32->i64 scratch0 -> scratch1
block_op 2 copy i64->i64 scratch1 -> output"
    );
    Ok(())
}

/// An op that does not support in-place blocks always copies, alternating scratch blocks.
#[test]
fn op_without_in_place_support_copies() -> VortexResult<()> {
    let program = counter_program(
        0,
        2000,
        vec![
            (add_one::<i32>(false), PType::I32),
            (add_one::<i32>(false), PType::I32),
            (add_one::<i32>(false), PType::I32),
        ],
    );
    let (output, trace) = run(program, &Mask::new_true(2000), Nullability::NonNullable)?;
    assert_arrays_eq!(output, PrimitiveArray::from_iter(3..2003i32), &mut ctx());
    assert_eq!(
        trace,
        "drive ops=3 len=2000
block_op 0 copy i32->i32 scratch0 -> scratch1
block_op 1 copy i32->i32 scratch1 -> scratch0
block_op 2 copy i32->i32 scratch0 -> output"
    );
    Ok(())
}

/// Blocks are cut on the source's grid, so a source whose first chunk starts part way is asked
/// for a short first block and then whole blocks.
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
    let (source, fills) = counter(7, block_offset);
    let mut program = BlockProgram::new(source, PType::I32, len);
    program.push(add_one::<i32>(true), PType::I32);
    let (output, _) = run(program, &Mask::new_true(len), Nullability::NonNullable)?;
    let end = 8 + i32::try_from(len)?;
    assert_arrays_eq!(output, PrimitiveArray::from_iter(8..end), &mut ctx());
    assert_eq!(*fills.lock(), expected);
    assert!(expected.iter().all(|rows| rows.len() <= BLOCK_LEN));
    Ok(())
}

/// A source over a canonical array lends its blocks. When nothing else references the buffer,
/// every op runs in it and it becomes the result; when another owner shares it, the first op
/// copies out of it, and the shared values are unchanged.
#[test]
fn canonical_leaf_decodes_in_place_only_when_owned() -> VortexResult<()> {
    let two_adds = |source| {
        let mut program = BlockProgram::new(source, PType::I32, 3000);
        program.push(add_one::<i32>(true), PType::I32);
        program.push(add_one::<i32>(true), PType::I32);
        program
    };
    let owned = PrimitiveSource::boxed(&PrimitiveArray::from_iter(0..3000i32));
    let (output, trace) = run(
        two_adds(owned),
        &Mask::new_true(3000),
        Nullability::NonNullable,
    )?;
    assert_arrays_eq!(output, PrimitiveArray::from_iter(2..3002i32), &mut ctx());
    assert_eq!(
        trace,
        "drive ops=2 len=3000
block_op 0 in_place i32->i32 leaf -> leaf
block_op 1 in_place i32->i32 leaf -> leaf"
    );

    let shared = PrimitiveArray::from_iter(0..3000i32);
    let program = two_adds(PrimitiveSource::boxed(&shared));
    let (output, trace) = run(program, &Mask::new_true(3000), Nullability::NonNullable)?;
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

/// Validity is read from the input mask at any bit offset, and flows through ops that leave it
/// alone into the output.
#[rstest]
#[case::aligned(0)]
#[case::bit_offset(3)]
#[case::byte_offset(16)]
fn validity_streams_through_the_ops(#[case] mask_offset: usize) -> VortexResult<()> {
    let len = 3000;
    let valid = |row: usize| !row.is_multiple_of(7) && row % 1000 != 5;
    let bits = BitBuffer::collect_bool(mask_offset + len, |i| {
        i >= mask_offset && valid(i - mask_offset)
    });
    let mask = Mask::from_buffer(bits.slice(mask_offset..mask_offset + len));
    let (source, _) = counter(0, 300);
    let mut program = BlockProgram::new(source, PType::I32, len);
    program.push(add_one::<i32>(true), PType::I32);
    let (output, _) = run(program, &mask, Nullability::Nullable)?;
    let expected = PrimitiveArray::from_option_iter(
        (0..len).map(|row| valid(row).then(|| i32::try_from(row + 1).unwrap_or(i32::MAX))),
    );
    assert_arrays_eq!(output, expected, &mut ctx());
    Ok(())
}

/// Validity that is uniform needs no bitmap: the output is non-nullable, all valid or all null.
#[rstest]
#[case::non_nullable(Mask::new_true(2000), Nullability::NonNullable, Validity::NonNullable)]
#[case::all_valid(Mask::new_true(2000), Nullability::Nullable, Validity::AllValid)]
#[case::all_null(Mask::new_false(2000), Nullability::Nullable, Validity::AllInvalid)]
fn uniform_validity_needs_no_bitmap(
    #[case] mask: Mask,
    #[case] nullability: Nullability,
    #[case] expected: Validity,
) -> VortexResult<()> {
    let program = counter_program(0, 2000, vec![(add_one::<i32>(true), PType::I32)]);
    let (output, _) = run(program, &mask, nullability)?;
    assert_eq!(
        std::mem::discriminant(&output.validity()?),
        std::mem::discriminant(&expected)
    );
    Ok(())
}

/// An op that nulls rows changes the output's validity, combined with the input's.
#[test]
fn ops_can_null_rows() -> VortexResult<()> {
    let len = 2500;
    let mask = Mask::from_buffer(BitBuffer::collect_bool(len, |row| !row.is_multiple_of(7)));
    let program = counter_program(
        0,
        len,
        vec![
            (Box::new(NullEvery(5)), PType::I32),
            (add_one::<i32>(true), PType::I32),
        ],
    );
    let (output, _) = run(program, &mask, Nullability::Nullable)?;
    let expected = PrimitiveArray::from_option_iter((0..len).map(|row| {
        (!row.is_multiple_of(7) && !row.is_multiple_of(5))
            .then(|| i32::try_from(row + 1).unwrap_or(i32::MAX))
    }));
    assert_arrays_eq!(output, expected, &mut ctx());
    Ok(())
}

/// A non-nullable output cannot hold the nulls an op produced.
#[test]
fn non_nullable_output_rejects_nulls() {
    let program = counter_program(0, 100, vec![(Box::new(NullEvery(5)), PType::I32)]);
    assert!(run(program, &Mask::new_true(100), Nullability::NonNullable).is_err());
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
    let program = counter_program(0, len, vec![(Box::new(Patch(patches)), PType::I32)]);
    let (output, _) = run(program, &Mask::new_true(len), Nullability::Nullable)?;
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

/// With no ops, a program emits its source's values.
#[test]
fn no_ops_emit_the_source() -> VortexResult<()> {
    let program = counter_program(4, 1500, vec![]);
    let (output, _) = run(program, &Mask::new_true(1500), Nullability::NonNullable)?;
    assert_arrays_eq!(output, PrimitiveArray::from_iter(4..1504i32), &mut ctx());
    Ok(())
}

/// An empty program emits an empty array and asks its source for nothing.
#[test]
fn empty_programs_fill_nothing() -> VortexResult<()> {
    let (source, fills) = counter(0, 0);
    let mut program = BlockProgram::new(source, PType::I32, 0);
    program.push(add_one::<i32>(true), PType::I32);
    let (output, _) = run(program, &Mask::new_true(0), Nullability::NonNullable)?;
    assert!(output.is_empty());
    assert!(fills.lock().is_empty());
    Ok(())
}
