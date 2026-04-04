// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::dict_test::gen_primitive_for_dict;
use vortex_array::builders::dict::dict_encode;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const LEN: usize = 1_000_000;

#[derive(Clone, Copy, Debug)]
struct BenchArgs {
    unique_count: usize,
    invalid_stride: usize,
}

impl fmt::Display for BenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unique_{}_invalid_every_{}",
            self.unique_count, self.invalid_stride
        )
    }
}

const BENCH_ARGS: &[BenchArgs] = &[
    BenchArgs {
        unique_count: 32,
        invalid_stride: 4,
    },
    BenchArgs {
        unique_count: 32,
        invalid_stride: 16,
    },
    BenchArgs {
        unique_count: 512,
        invalid_stride: 4,
    },
    BenchArgs {
        unique_count: 512,
        invalid_stride: 16,
    },
];

#[divan::bench(args = BENCH_ARGS)]
fn compare_to_constant(bencher: Bencher, args: BenchArgs) {
    let (masked, target) = masked_dict_fixture(args);
    let compare_value = ConstantArray::new(target, masked.len()).into_array();
    let session = VortexSession::empty();

    bencher
        .with_inputs(|| {
            (
                masked.clone(),
                compare_value.clone(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(masked, compare_value, ctx)| {
            let result = masked
                .binary(compare_value.clone(), Operator::Eq)
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap();
            divan::black_box(result);
        });
}

fn masked_dict_fixture(args: BenchArgs) -> (vortex_array::ArrayRef, i32) {
    let primitive = gen_primitive_for_dict::<i32>(LEN, args.unique_count);
    let target = primitive.as_slice::<i32>()[0];
    let dict = dict_encode(&primitive.into_array()).unwrap();
    let validity = Validity::from_iter((0..LEN).map(|idx| idx % args.invalid_stride != 0));
    let masked = MaskedArray::try_new(dict.into_array(), validity)
        .unwrap()
        .into_array();

    (masked, target)
}
