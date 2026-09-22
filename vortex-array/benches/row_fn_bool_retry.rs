// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures packed Boolean dense attempts and nullable retry through public RowFn execution.

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

const SIZES: &[usize] = &[64, 16_384];
const CASES: &[(InputShape, Scenario)] = &[
    (InputShape::Columns, Scenario::AllValid),
    (InputShape::ConstantLhs, Scenario::AllValid),
    (InputShape::ConstantRhs, Scenario::AllValid),
    (InputShape::Columns, Scenario::PartialAccepted),
    (InputShape::ConstantLhs, Scenario::PartialAccepted),
    (InputShape::ConstantRhs, Scenario::PartialAccepted),
    (InputShape::Columns, Scenario::NullOnlyFailure),
    (InputShape::ConstantLhs, Scenario::NullOnlyFailure),
    (InputShape::ConstantRhs, Scenario::NullOnlyFailure),
    (InputShape::Columns, Scenario::ObservableFailure),
    (InputShape::ConstantLhs, Scenario::ObservableFailure),
    (InputShape::ConstantRhs, Scenario::ObservableFailure),
];

#[derive(Clone, Copy, Debug)]
enum InputShape {
    Columns,
    ConstantLhs,
    ConstantRhs,
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    AllValid,
    PartialAccepted,
    NullOnlyFailure,
    ObservableFailure,
}

#[derive(Clone)]
struct Predicate<const MULTIVERSIONED: bool>;

impl<const MULTIVERSIONED: bool> RowFn for Predicate<MULTIVERSIONED> {
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
        visitor.visit_deferred_bool::<(i64, i64), bool, MULTIVERSIONED>(
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
fn plain<const ROWS: usize>(bencher: Bencher, &(shape, scenario): &(InputShape, Scenario)) {
    bench_predicate::<false>(bencher, ROWS, shape, scenario);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = CASES, consts = SIZES)]
fn multiversioned<const ROWS: usize>(
    bencher: Bencher,
    &(shape, scenario): &(InputShape, Scenario),
) {
    bench_predicate::<true>(bencher, ROWS, shape, scenario);
}

fn bench_predicate<const MULTIVERSIONED: bool>(
    bencher: Bencher,
    rows: usize,
    shape: InputShape,
    scenario: Scenario,
) {
    let validity = match scenario {
        Scenario::AllValid | Scenario::ObservableFailure => Validity::NonNullable,
        Scenario::PartialAccepted | Scenario::NullOnlyFailure => Validity::Array(
            BoolArray::from_iter((0..rows).map(|index| !index.is_multiple_of(8))).into_array(),
        ),
    };
    let rejected = matches!(
        scenario,
        Scenario::NullOnlyFailure | Scenario::ObservableFailure
    );
    let column = PrimitiveArray::new(
        (0..rows)
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
            PrimitiveArray::from_iter((0..rows).map(|index| {
                i64::try_from(index % 512 + 1).vortex_expect("fixture value is at most 512")
            }))
            .into_array(),
        ),
        InputShape::ConstantLhs => (ConstantArray::new(512_i64, rows).into_array(), column),
        InputShape::ConstantRhs => (column, ConstantArray::new(512_i64, rows).into_array()),
    };
    let args = VecExecutionArgs::new(vec![lhs, rhs], rows);
    let function = Predicate::<MULTIVERSIONED>;
    let result = execute_rows(
        &function,
        &EmptyOptions,
        &args,
        &mut SESSION.create_execution_ctx(),
    );
    assert_eq!(
        result.is_err(),
        matches!(scenario, Scenario::ObservableFailure)
    );
    drop(result);

    bencher
        .counter(ItemsCount::new(rows))
        .with_inputs(|| (&args, SESSION.create_execution_ctx()))
        .bench_refs(|(args, ctx)| {
            let result = execute_rows(&function, &EmptyOptions, *args, ctx);
            drop(divan::black_box(result));
        });
}
