// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A chain of add-constant ops over a primitive array, applied level by level as decoding does
//! today and block by block with a [`BlockProgram`].
//!
//! Level by level, the first op copies the input into a new buffer and each op after it rewrites
//! that buffer in place, as FoR's decode does. A program reads each block of at most 1024 rows
//! once and runs every op over it while it stays in L1. The nullable case adds validity with nulls
//! at a bit offset.
//!
//! Run with `cargo bench -p vortex-array --bench block_program`.

#![expect(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::block::Block;
use vortex_array::block::BlockOp;
use vortex_array::block::BlockProgram;
use vortex_array::block::PrimitiveSource;
use vortex_array::block::Written;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// A scan split's rows, whose intermediates fit in L2.
const LEN: usize = 64 * 1024;

/// Ops in the chain.
const DEPTH: usize = 4;

fn values() -> Buffer<u32> {
    (0..LEN as u32).map(|i| (i * 7) % 1000).collect()
}

/// The constant op `level` adds.
fn constant(level: usize) -> u32 {
    1_000 + level as u32
}

struct AddConstant(u32);

impl BlockOp for AddConstant {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        let constant = self.0;
        Ok(block.map::<u32, u32>(move |v| v.wrapping_add(constant)))
    }
}

/// Apply the chain level by level, as FoR's decode does: copy the input with the first op into a
/// buffer from the context's allocator, then rewrite the buffer in place with each op after it.
fn level_wise(input: &Buffer<u32>, ctx: &mut ExecutionCtx) -> PrimitiveArray {
    let first = constant(0);
    let mut buffer = BufferMut::<u32>::with_capacity_in(LEN, ctx.allocator().clone());
    input
        .as_slice()
        .map_into(&mut buffer.spare_capacity_mut()[..LEN], move |v| {
            v.wrapping_add(first)
        });
    // SAFETY: `map_into` wrote every one of the `LEN` values.
    unsafe { buffer.set_len(LEN) };
    for level in 1..DEPTH {
        let constant = constant(level);
        buffer
            .as_mut_slice()
            .map_into_in_place(move |v| v.wrapping_add(constant));
    }
    PrimitiveArray::new(buffer.freeze(), Validity::NonNullable)
}

/// Run the chain as one program over the input's blocks.
fn program(
    input: Buffer<u32>,
    validity: &Mask,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> PrimitiveArray {
    let source = PrimitiveSource::new(input, validity.clone());
    let mut program = BlockProgram::new(Box::new(source), PType::U32, LEN);
    for level in 0..DEPTH {
        program.push(Box::new(AddConstant(constant(level))), PType::U32);
    }
    program.run(nullability, ctx).unwrap()
}

#[divan::bench]
fn level_wise_chain(bencher: Bencher) {
    let input = values();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_values(|mut ctx| level_wise(&input, &mut ctx));
}

#[divan::bench]
fn program_chain(bencher: Bencher) {
    let (input, validity) = (values(), Mask::new_true(LEN));
    bencher
        .with_inputs(|| (input.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(input, mut ctx)| {
            program(input, &validity, Nullability::NonNullable, &mut ctx)
        });
}

/// Validity with every seventh row null, starting 3 bits into its buffer.
#[divan::bench]
fn program_chain_nullable(bencher: Bencher) {
    let input = values();
    let bits: BitBuffer = (0..LEN + 3).map(|i| i % 7 != 0).collect();
    let validity = Mask::from_buffer(bits.slice(3..));
    bencher
        .with_inputs(|| (input.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(input, mut ctx)| {
            program(input, &validity, Nullability::Nullable, &mut ctx)
        });
}
