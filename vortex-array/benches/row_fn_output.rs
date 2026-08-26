// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(
    clippy::cast_possible_truncation,
    reason = "benchmark fixtures reduce values below 1024 before narrowing"
)]

use std::marker::PhantomData;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::VecExecutionArgs;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_array::scalar_fn::unstable::row::execute_rows;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const ROWS: usize = 1 << 14;

const INPUT_SHAPES: &[InputShape] = &[
    InputShape::PerRowPerRow,
    InputShape::PerRowConstant,
    InputShape::ConstantPerRow,
];

#[derive(Clone, Copy, Debug)]
enum InputShape {
    PerRowPerRow,
    PerRowConstant,
    ConstantPerRow,
}

trait BenchPrimitive: NativePType {
    fn per_row(offset: usize) -> ArrayRef;

    fn constant() -> ArrayRef;
}

impl BenchPrimitive for i32 {
    fn per_row(offset: usize) -> ArrayRef {
        PrimitiveArray::from_iter((0..ROWS).map(|index| ((index + offset) % 1024) as i32))
            .into_array()
    }

    fn constant() -> ArrayRef {
        ConstantArray::new(512_i32, ROWS).into_array()
    }
}

impl BenchPrimitive for i64 {
    fn per_row(offset: usize) -> ArrayRef {
        PrimitiveArray::from_iter((0..ROWS).map(|index| ((index + offset) % 1024) as i64))
            .into_array()
    }

    fn constant() -> ArrayRef {
        ConstantArray::new(512_i64, ROWS).into_array()
    }
}

#[derive(Clone)]
struct InfallibleBool<T>(PhantomData<T>);

impl<T: BenchPrimitive> RowFn for InfallibleBool<T> {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_fn_output.infallible_bool");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(T, T), bool>(|(lhs, rhs)| lhs.is_lt(rhs))
    }
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [i32, i64], args = INPUT_SHAPES)]
fn infallible_bool<T: BenchPrimitive>(bencher: Bencher, &shape: &InputShape) {
    let function = InfallibleBool::<T>(PhantomData);
    let args = match shape {
        InputShape::PerRowPerRow => vec![T::per_row(0), T::per_row(1)],
        InputShape::PerRowConstant => vec![T::per_row(0), T::constant()],
        InputShape::ConstantPerRow => vec![T::constant(), T::per_row(1)],
    };
    let args = VecExecutionArgs::new(args, ROWS);

    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&args, SESSION.create_execution_ctx()))
        .bench_refs(|(args, ctx)| {
            execute_rows(&function, &EmptyOptions, *args, ctx)
                .vortex_expect("row execution should succeed in benchmark")
        });
}
