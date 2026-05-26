// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::UInt32Array;
use arrow_buffer::NullBuffer;
use arrow_cast::CastOptions;
use arrow_schema::DataType as ArrowDataType;
use divan::Bencher;
use rand::prelude::*;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Stat;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const N: usize = 100_000;

#[divan::bench]
fn cast_u16_to_u32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);
    #[expect(clippy::cast_possible_truncation)]
    let arr = PrimitiveArray::from_option_iter((0..N).map(|i| {
        if rng.random_bool(0.5) {
            None
        } else {
            Some(i as u16)
        }
    }))
    .into_array();
    // Pre-compute min/max so values_fit_in is a cache hit during the benchmark.
    arr.statistics()
        .compute_all(
            &[Stat::Min, Stat::Max],
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .ok();
    bencher.with_inputs(|| arr.clone()).bench_refs(|a| {
        #[expect(clippy::unwrap_used)]
        a.cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
    });
}

// Slow-path inputs: u32 -> u8 with mixed validity, all values in-range (so the cast succeeds),
// no precomputed min/max stats — forces `cast_values` and the `Mask::Values` arm.
fn slow_path_inputs() -> (Vec<u32>, BitBuffer) {
    let mut rng = StdRng::seed_from_u64(42);
    let values: Vec<u32> = (0..N).map(|_| rng.random_range(0..=200u32)).collect();
    let validity: BitBuffer = (0..N).map(|_| rng.random_bool(0.7)).collect();
    (values, validity)
}

#[divan::bench]
fn cast_u32_u8_vortex(bencher: Bencher) {
    let (values, validity) = slow_path_inputs();
    let arr = PrimitiveArray::new(Buffer::from(values), Validity::from(validity)).into_array();
    bencher.with_inputs(|| arr.clone()).bench_refs(|a| {
        #[expect(clippy::unwrap_used)]
        a.cast(DType::Primitive(PType::U8, Nullability::Nullable))
            .unwrap()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
    });
}

#[divan::bench]
fn cast_u32_u8_arrow(bencher: Bencher) {
    let (values, validity) = slow_path_inputs();
    let nulls = NullBuffer::from(validity.iter().collect::<Vec<_>>());
    let arr: Arc<UInt32Array> = Arc::new(UInt32Array::new(values.into(), Some(nulls)));
    let opts = CastOptions { safe: false, ..Default::default() };
    bencher.with_inputs(|| Arc::clone(&arr)).bench_refs(|a| {
        #[expect(clippy::unwrap_used)]
        arrow_cast::cast_with_options(a.as_ref(), &ArrowDataType::UInt8, &opts).unwrap()
    });
}

// Pure scalar baseline: no validity mask at all, checked cast on every element. Bails on
// the first overflow (which never happens for our in-range inputs).
#[divan::bench]
fn cast_u32_u8_checked_no_validity(bencher: Bencher) {
    let (values, _) = slow_path_inputs();
    bencher.with_inputs(|| values.clone()).bench_refs(|vs| {
        let mut out = Vec::with_capacity(vs.len());
        for &v in vs.iter() {
            #[expect(clippy::expect_used)]
            out.push(u8::try_from(v).expect("in-range"));
        }
        out
    });
}

