// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `turboquant_encode` and `turboquant_decode` across different validity-mask
//! shapes.
//!
//! The four mask shapes (`AllTrue`, `AllFalse`, dense `Values`, sparse `Values`) exercise the
//! variant-specialized paths added in the mask refactor in `vector/normalize.rs`,
//! `vector/quantize.rs`, and `scalar_fns/decode.rs`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::extension::EmptyMetadata;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;
use vortex_tensor::vector::Vector;
use vortex_turboquant::TQDecode;
use vortex_turboquant::TQEncode;
use vortex_turboquant::TurboQuantConfig;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<ArraySession>();
    vortex_turboquant::initialize(&session);
    session
});

/// Shape of the validity mask used to drive the variant-specialized paths.
#[derive(Copy, Clone)]
enum MaskShape {
    AllValid,
    AllInvalid,
    DenseValues,
    SparseValues,
}

impl std::fmt::Debug for MaskShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MaskShape::AllValid => "all_valid",
            MaskShape::AllInvalid => "all_invalid",
            MaskShape::DenseValues => "dense_95pct",
            MaskShape::SparseValues => "sparse_5pct",
        })
    }
}

impl MaskShape {
    fn build(self, rows: usize, rng: &mut StdRng) -> Validity {
        match self {
            MaskShape::AllValid => Validity::NonNullable,
            MaskShape::AllInvalid => Validity::AllInvalid,
            MaskShape::DenseValues => Validity::from_iter((0..rows).map(|_| rng.random_bool(0.95))),
            MaskShape::SparseValues => {
                Validity::from_iter((0..rows).map(|_| rng.random_bool(0.05)))
            }
        }
    }
}

const MASK_SHAPES: &[MaskShape] = &[
    MaskShape::AllValid,
    MaskShape::AllInvalid,
    MaskShape::DenseValues,
    MaskShape::SparseValues,
];

const ROWS: usize = 4096;
const DIMENSIONS: u32 = 128;

fn build_vector_array(shape: MaskShape) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let dim = DIMENSIONS as usize;
    let values: Buffer<f32> = (0..ROWS * dim).map(|_| rng.random::<f32>()).collect();
    let elements = PrimitiveArray::new::<f32>(values, Validity::NonNullable);
    let validity = shape.build(ROWS, &mut rng);
    let fsl =
        FixedSizeListArray::try_new(elements.into_array(), DIMENSIONS, validity, ROWS).unwrap();

    ExtensionArray::try_new_from_vtable(Vector, EmptyMetadata, fsl.into_array())
        .unwrap()
        .into_array()
}

fn encode(vec: ArrayRef, config: &TurboQuantConfig, ctx: &mut ExecutionCtx) -> ArrayRef {
    TQEncode::try_new_array(vec, config)
        .unwrap()
        .into_array()
        .execute(ctx)
        .unwrap()
}

fn decode(encoded: ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    TQDecode::try_new_array(encoded)
        .unwrap()
        .into_array()
        .execute(ctx)
        .unwrap()
}

fn config() -> TurboQuantConfig {
    // 4 bits, 4 SORF rounds, fixed seed: representative defaults from the test fixtures.
    TurboQuantConfig::try_new(4, 0xDEADBEEF, 4).unwrap()
}

#[divan::bench(args = MASK_SHAPES)]
fn turboquant_encode(bencher: Bencher, shape: &MaskShape) {
    let shape = *shape;
    let cfg = config();
    bencher
        .with_inputs(|| (build_vector_array(shape), SESSION.create_execution_ctx()))
        .input_counter(|_| divan::counter::ItemsCount::new(ROWS))
        .bench_values(|(arr, mut ctx)| encode(arr, &cfg, &mut ctx))
}

#[divan::bench(args = MASK_SHAPES)]
fn turboquant_decode(bencher: Bencher, shape: &MaskShape) {
    let shape = *shape;
    let cfg = config();
    bencher
        .with_inputs(|| {
            let arr = build_vector_array(shape);
            let mut ctx = SESSION.create_execution_ctx();
            let encoded = encode(arr, &cfg, &mut ctx);
            (encoded, SESSION.create_execution_ctx())
        })
        .input_counter(|_| divan::counter::ItemsCount::new(ROWS))
        .bench_values(|(encoded, mut ctx)| decode(encoded, &mut ctx))
}
