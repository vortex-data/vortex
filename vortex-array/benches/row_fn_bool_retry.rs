// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures packed Boolean dense attempts and nullable retry through public RowFn execution.
//!
//! Every case uses the multiversioned collector over partially valid input. An accepted attempt
//! keeps the dense result, and a null-only failure retries the valid rows.

use std::array;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::VecExecutionArgs;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_array::scalar_fn::unstable::row::execute_rows;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const SIZES: &[usize] = &[16_384];
const BATCHES_PER_ITER: usize = 8;
const CASES: &[(InputShape, Scenario)] = &[
    (InputShape::Columns, Scenario::PartialAccepted),
    (InputShape::ConstantLhs, Scenario::PartialAccepted),
    (InputShape::Columns, Scenario::NullOnlyFailure),
    (InputShape::ConstantLhs, Scenario::NullOnlyFailure),
];

#[derive(Clone, Copy, Debug)]
enum InputShape {
    Columns,
    ConstantLhs,
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    PartialAccepted,
    NullOnlyFailure,
}

#[derive(Clone)]
struct Predicate;

impl RowFn for Predicate {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_fn_bool_retry");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &EmptyOptions,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred_bool::<(i64, i64), bool, true>(
            |(lhs, rhs)| (lhs < rhs, lhs == i64::MAX || rhs == i64::MAX),
            |failed| {
                vortex_ensure!(!failed, "predicate rejected sentinel");
                Ok(())
            },
        )
    }
}

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = CASES, consts = SIZES)]
fn deferred_bool<const ROWS: usize>(
    bencher: Bencher,
    &(shape, scenario): &(InputShape, Scenario),
) {
    let validity = Validity::Array(
        BoolArray::from_iter((0..ROWS).map(|index| !index.is_multiple_of(8))).into_array(),
    );
    let rejected = matches!(scenario, Scenario::NullOnlyFailure);
    let column = PrimitiveArray::new(
        (0..ROWS)
            .map(|index| {
                if rejected && index.is_multiple_of(8) {
                    i64::MAX
                } else {
                    i64::try_from(index % 1024).vortex_expect("fixture value is less than 1024")
                }
            })
            .collect::<Vec<_>>(),
        validity,
    )
    .into_array();
    let (lhs, rhs) = match shape {
        InputShape::Columns => (
            column,
            PrimitiveArray::from_iter((0..ROWS).map(|index| {
                i64::try_from(index % 512 + 1).vortex_expect("fixture value is at most 512")
            }))
            .into_array(),
        ),
        InputShape::ConstantLhs => (ConstantArray::new(512_i64, ROWS).into_array(), column),
    };
    let args = VecExecutionArgs::new(vec![lhs, rhs], ROWS);
    let function = Predicate;
    drop(
        execute_rows(
            &function,
            &EmptyOptions,
            &args,
            &mut SESSION.create_execution_ctx(),
        )
        .vortex_expect("every rejected row is null"),
    );

    bencher
        .counter(ItemsCount::new(ROWS * BATCHES_PER_ITER))
        .with_inputs(|| {
            (
                &args,
                array::from_fn::<_, BATCHES_PER_ITER, _>(|_| SESSION.create_execution_ctx()),
            )
        })
        .bench_refs(|(args, contexts)| {
            contexts
                .each_mut()
                .map(|ctx| execute_rows(&function, &EmptyOptions, *args, ctx))
        });
}
