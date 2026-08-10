// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks how the row lifting applies validity to a dense kernel.
//!
//! Both arms run the same kernel over the same nullable input and differ only in how the output's
//! nulls are restored:
//!
//! - `lazy` is what the lifting behind [`RowFn`] does: conjoin the input validities (itself lazy)
//!   and hand the resulting boolean array straight to `mask`.
//! - `eager` is the strategy it replaced: materialize the conjunction into a `Mask`, copy it into a
//!   `BitBuffer`, wrap that in a `BoolArray`, and mask with it.
//!
//! `chain` composes three of the same function, which is where a per-call materialization
//! compounds.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::union_child_validities;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::RowExecution;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

const SIZES: &[usize] = &[65_536, 1 << 20];

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

/// The shared kernel: double every lane, ignoring validity.
fn doubled(input: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let input = input.clone().execute::<PrimitiveArray>(ctx)?;
    let values: Buffer<i32> = input
        .as_slice::<i32>()
        .iter()
        .map(|value| value.wrapping_mul(2))
        .collect();
    Ok(PrimitiveArray::new(values, Validity::NonNullable).into_array())
}

/// Dense row function: the lifting decides how validity is applied.
///
/// Its [`reduce_encoded`](RowFn::reduce_encoded) answers with [`doubled`] before the row loop is
/// reached, so this arm runs exactly the kernel [`EagerDouble`] runs and the two differ only in
/// validity. The row closure below is what makes it a [`RowFn`] at all, and never executes.
#[derive(Clone)]
struct LazyDouble;

impl RowFn for LazyDouble {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.lazy_double");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i32,), i32>(|(value,)| value.wrapping_mul(2))
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        doubled(&args[0], ctx).map(|output| Some(RowExecution::Output(output)))
    }
}

/// The same function, applying validity the way the adapter used to: materialize a mask first.
#[derive(Clone)]
struct EagerDouble;

impl ScalarFnVTable for EagerDouble {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.eager_double");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn return_dtype(&self, _options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::I32, args[0].nullability()))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let valid = input.validity()?.execute_mask(args.row_count(), ctx)?;
        let values = doubled(&input, ctx)?;

        if valid.all_true() {
            return values.cast(DType::Primitive(PType::I32, input.dtype().nullability()));
        }

        let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
        values.mask(mask)
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// A nullable i32 column with array-backed validity (~10% nulls).
fn nullable_input(len: usize) -> ArrayRef {
    PrimitiveArray::new(
        (0..len as i32).collect::<Buffer<i32>>(),
        Validity::from_iter((0..len).map(|index| !index.is_multiple_of(10))),
    )
    .into_array()
}

fn bench_depth<F>(bencher: Bencher, function: F, len: usize, depth: usize)
where
    F: ScalarFnVTable<Options = EmptyOptions> + Clone,
{
    bencher
        .with_inputs(|| nullable_input(len))
        .bench_local_values(|input| {
            let mut ctx = SESSION.create_execution_ctx();
            let mut array = input;
            for _ in 0..depth {
                array = function
                    .clone()
                    .try_new_array(len, EmptyOptions, [array])
                    .unwrap()
                    .into_array();
            }
            array.execute::<Canonical>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = SIZES)]
fn lazy(bencher: Bencher, len: usize) {
    bench_depth(bencher, LazyDouble, len, 1);
}

#[divan::bench(args = SIZES)]
fn eager(bencher: Bencher, len: usize) {
    bench_depth(bencher, EagerDouble, len, 1);
}

#[divan::bench(args = SIZES)]
fn lazy_chain(bencher: Bencher, len: usize) {
    bench_depth(bencher, LazyDouble, len, 3);
}

#[divan::bench(args = SIZES)]
fn eager_chain(bencher: Bencher, len: usize) {
    bench_depth(bencher, EagerDouble, len, 3);
}
