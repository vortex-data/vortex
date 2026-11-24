// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding FoR stacks over BitPacked level by level with `execute`, and block by block with a
//! program: FoR's fused unpack of BitPacked as the leaf, and an add-reference op for each FoR
//! above it.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench block_fusion`.

#![expect(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::block::BlockProgram;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::block::for_block_op;
use vortex_fastlanes::block::for_block_source;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<ArraySession>();
    vortex_fastlanes::initialize(&session);
    session
});

/// A scan split's rows, whose intermediates fit in L2.
const LEN: usize = 64 * 1024;

/// FoR levels over the BitPacked leaf.
const DEPTHS: [usize; 3] = [2, 4, 8];

/// `depth` FoR levels over values bit-packed at width 10.
fn for_stack(depth: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..LEN as u32).map(|i| (i * 7) % 1000));
    let mut array = bitpack_encode(&values, 10, None, &mut ctx)
        .unwrap()
        .into_array();
    for level in 0..depth {
        array = FoR::try_new(array, Scalar::from(1_000u32 + level as u32))
            .unwrap()
            .into_array();
    }
    array
}

/// Decode a FoR stack with a program: the innermost FoR over BitPacked as the leaf, then an op
/// for each FoR above it.
fn run_program(array: &ArrayRef, ctx: &mut ExecutionCtx) -> PrimitiveArray {
    let mut levels = Vec::new();
    let mut current = array.clone();
    while let Some(encoded) = current.as_opt::<FoR>().map(|for_| for_.encoded().clone()) {
        levels.push(current);
        current = encoded;
    }
    let (leaf, above) = levels.split_last().unwrap();
    let source = for_block_source(leaf.as_::<FoR>(), ctx).unwrap().unwrap();
    let mut program = BlockProgram::new(source, leaf.dtype().as_ptype(), leaf.len());
    for level in above.iter().rev() {
        let op = for_block_op(level.as_::<FoR>()).unwrap().unwrap();
        program.push(op, level.dtype().as_ptype());
    }
    let validity = leaf
        .validity()
        .unwrap()
        .execute_mask(leaf.len(), ctx)
        .unwrap();
    program
        .run(&validity, array.dtype().nullability(), ctx)
        .unwrap()
}

#[divan::bench(args = DEPTHS)]
fn for_stack_execute(bencher: Bencher, depth: usize) {
    let array = for_stack(depth);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench(args = DEPTHS)]
fn for_stack_program(bencher: Bencher, depth: usize) {
    let array = for_stack(depth);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| run_program(&array, &mut ctx));
}
