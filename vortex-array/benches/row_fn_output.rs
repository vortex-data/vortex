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
use vortex_array::ExecutionCtx;
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
use vortex_array::scalar_fn::unstable::row::InitializedElement;
use vortex_array::scalar_fn::unstable::row::InputElement;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_array::scalar_fn::unstable::row::UninitElementSink;
use vortex_array::scalar_fn::unstable::row::execute_rows;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
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

/// The shape every row function is measured on: one per-row operand against another.
const INPUT_SHAPES: &[InputShape] = &[InputShape::PerRowPerRow];

/// Constant operands are measured on `i64` alone, by the `*_constant` benchmarks below.
///
/// The row executor decodes and orients a constant operand the same way whatever the row kernel
/// does with the value, so running both constant orientations under every element type and every
/// output mode measures one path repeatedly, three walltime legs at a time. One infallible pair and
/// one deferred pair keep it covered: those two differ in how a constant row reaches the kernel,
/// and that is the difference worth watching.
const CONSTANT_SHAPES: &[InputShape] = &[InputShape::PerRowConstant, InputShape::ConstantPerRow];

const VALIDITY_PATTERNS: &[ValidityPattern] = &[
    ValidityPattern::AllValid,
    ValidityPattern::OneNullInEight,
    ValidityPattern::NineNullsInTen,
];

#[derive(Clone, Copy, Debug)]
enum InputShape {
    PerRowPerRow,
    PerRowConstant,
    ConstantPerRow,
}

#[derive(Clone, Copy, Debug)]
enum ValidityPattern {
    AllValid,
    OneNullInEight,
    NineNullsInTen,
}

impl ValidityPattern {
    fn is_valid(self, index: usize) -> bool {
        match self {
            Self::AllValid => true,
            Self::OneNullInEight => !index.is_multiple_of(8),
            Self::NineNullsInTen => index.is_multiple_of(10),
        }
    }
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

/// Primitive input that declines null-tolerant decoding to exercise the filtered fallback.
struct FilteredI64;

// SAFETY: the view is a native slice, and its reported length is the slice length.
unsafe impl InputElement for FilteredI64 {
    type Column = Buffer<i64>;
    type Constant = i64;
    type View<'a> = &'a [i64];
    type Elem<'a> = i64;

    const DENSE_SAFE: bool = false;
    const DECODE_INFALLIBLE: bool = true;

    fn validate(dtype: &DType) -> VortexResult<()> {
        <i64 as InputElement>::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        <i64 as InputElement>::decode(array, ctx)
    }

    fn decode_constant(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Constant> {
        <i64 as InputElement>::decode_constant(array, ctx)
    }

    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_> {
        column[index]
    }

    fn get_constant(constant: &Self::Constant) -> Self::Elem<'_> {
        *constant
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column.as_slice()
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a> {
        view[index]
    }

    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a> {
        // SAFETY: forwarded from this method's contract.
        unsafe { *view.get_unchecked(index) }
    }
}

#[derive(Clone)]
struct InfallibleBool<T>(PhantomData<T>);

#[derive(Clone)]
struct DeferredBool<T>(PhantomData<T>);

#[derive(Clone)]
struct DeferredI64;

#[derive(Clone)]
struct FilteredOwnedI64;

#[derive(Clone)]
struct FilteredSinkI64;

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

impl<T: BenchPrimitive> RowFn for DeferredBool<T> {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_fn_output.deferred_bool");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred_bool::<(T, T), bool, false>(
            |(lhs, rhs)| (lhs.is_lt(rhs), lhs.is_lt(T::default())),
            |negative| {
                vortex_ensure!(
                    !negative,
                    "deferred-bool benchmark inputs must be nonnegative"
                );
                Ok(())
            },
        )
    }
}

impl RowFn for DeferredI64 {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_fn_output.deferred_i64");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred::<(i64, i64), i64, bool>(
            |(lhs, rhs)| lhs.overflowing_add(rhs),
            |overflowed| {
                vortex_ensure!(
                    !overflowed,
                    "deferred-i64 benchmark inputs must not overflow"
                );
                Ok(())
            },
        )
    }
}

impl RowFn for FilteredOwnedI64 {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_fn_output.filtered_owned_i64");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(FilteredI64,), i64>(|(value,)| value.wrapping_add(1))
    }
}

impl RowFn for FilteredSinkI64 {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_fn_output.filtered_sink_i64");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_into::<(FilteredI64,), UninitElementSink<i64>, _>((), |(value,), output| {
            // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
            unsafe { InitializedElement::write(output, value.wrapping_add(1)) }
        })
    }
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [i32, i64], args = INPUT_SHAPES)]
fn infallible_bool<T: BenchPrimitive>(bencher: Bencher, &shape: &InputShape) {
    let function = InfallibleBool::<T>(PhantomData);
    bench_row_fn(bencher, &function, make_args::<T>(shape));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = CONSTANT_SHAPES)]
fn infallible_bool_constant(bencher: Bencher, &shape: &InputShape) {
    let function = InfallibleBool::<i64>(PhantomData);
    bench_row_fn(bencher, &function, make_args::<i64>(shape));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [i32, i64], args = INPUT_SHAPES)]
fn deferred_bool<T: BenchPrimitive>(bencher: Bencher, &shape: &InputShape) {
    let function = DeferredBool::<T>(PhantomData);
    bench_row_fn(bencher, &function, make_args::<T>(shape));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = CONSTANT_SHAPES)]
fn deferred_bool_constant(bencher: Bencher, &shape: &InputShape) {
    let function = DeferredBool::<i64>(PhantomData);
    bench_row_fn(bencher, &function, make_args::<i64>(shape));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = INPUT_SHAPES)]
fn deferred_i64(bencher: Bencher, &shape: &InputShape) {
    bench_row_fn(bencher, &DeferredI64, make_args::<i64>(shape));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = VALIDITY_PATTERNS)]
fn filtered_owned_i64(bencher: Bencher, &validity: &ValidityPattern) {
    bench_row_fn(bencher, &FilteredOwnedI64, make_filtered_args(validity));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = VALIDITY_PATTERNS)]
fn filtered_sink_i64(bencher: Bencher, &validity: &ValidityPattern) {
    bench_row_fn(bencher, &FilteredSinkI64, make_filtered_args(validity));
}

fn make_args<T: BenchPrimitive>(shape: InputShape) -> VecExecutionArgs {
    let args = match shape {
        InputShape::PerRowPerRow => vec![T::per_row(0), T::per_row(1)],
        InputShape::PerRowConstant => vec![T::per_row(0), T::constant()],
        InputShape::ConstantPerRow => vec![T::constant(), T::per_row(1)],
    };

    VecExecutionArgs::new(args, ROWS)
}

fn make_filtered_args(validity: ValidityPattern) -> VecExecutionArgs {
    let values = PrimitiveArray::new(
        (0..ROWS).map(|index| index as i64).collect::<Buffer<_>>(),
        Validity::from_iter((0..ROWS).map(|index| validity.is_valid(index))),
    )
    .into_array();

    VecExecutionArgs::new(vec![values], ROWS)
}

fn bench_row_fn<F: RowFn<Options = EmptyOptions>>(
    bencher: Bencher,
    function: &F,
    args: VecExecutionArgs,
) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (&args, SESSION.create_execution_ctx()))
        .bench_refs(|(args, ctx)| {
            execute_rows(function, &EmptyOptions, *args, ctx)
                .vortex_expect("row execution should succeed in benchmark")
        });
}
