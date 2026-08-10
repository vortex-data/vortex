// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares owned-output, sink-writing, and hand-written primitive row loops.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::OutputSink;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

const ROWS: usize = 65_536;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

#[derive(Clone)]
struct RowWrappingAdd;

impl RowFn for RowWrappingAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_wrapping_add");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64, i64), i64>(|(lhs, rhs)| lhs.wrapping_add(rhs))
    }
}

#[derive(Clone)]
struct RowCheckedAdd;

impl RowFn for RowCheckedAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_checked_add");
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
            |failed| {
                if failed {
                    return Err(checked_add_error());
                }
                Ok(())
            },
        )
    }
}

/// Keep error construction out of the benchmarked success path.
#[cold]
#[inline(never)]
fn checked_add_error() -> VortexError {
    vortex_err!("integer overflow in row checked add")
}

/// A benchmark sink that writes one `i64` per row.
struct I64Sink(
    /// The output values written by the row loop.
    BufferMut<i64>,
);

impl OutputSink for I64Sink {
    type Rows<'a> = &'a mut [i64];
    type Row<'a> = &'a mut i64;
    type WriteToken = ();

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
        Ok(DType::from(i64::PTYPE))
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Ok(Self(BufferMut::zeroed(rows)))
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        self.0.as_mut_slice()
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.len() == row_count
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        &mut rows[index]
    }

    unsafe fn finish(self) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::new(self.0.freeze(), Validity::NonNullable).into_array())
    }
}

#[derive(Clone)]
struct RowSinkWrappingAdd;

impl RowFn for RowSinkWrappingAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_sink_wrapping_add");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_into::<(i64, i64), I64Sink, _>(|(lhs, rhs), out| {
            *out = lhs.wrapping_add(rhs);
        })
    }
}

fn inputs() -> (ArrayRef, ArrayRef) {
    let lhs = (0..ROWS)
        .map(|index| index as i64)
        .collect::<Buffer<_>>()
        .into_array();
    let rhs = (0..ROWS)
        .map(|index| (index % 17) as i64)
        .collect::<Buffer<_>>()
        .into_array();
    (lhs, rhs)
}

fn constant_inputs() -> (ArrayRef, ArrayRef) {
    let (lhs, _) = inputs();
    let rhs = ConstantArray::new(Scalar::from(7i64), ROWS).into_array();
    (lhs, rhs)
}

fn nullable_inputs() -> (ArrayRef, ArrayRef) {
    let lhs = PrimitiveArray::new(
        (0..ROWS).map(|index| index as i64).collect::<Buffer<_>>(),
        Validity::from_iter((0..ROWS).map(|index| !index.is_multiple_of(5))),
    )
    .into_array();
    let rhs = PrimitiveArray::new(
        (0..ROWS)
            .map(|index| (index % 17) as i64)
            .collect::<Buffer<_>>(),
        Validity::from_iter((0..ROWS).map(|index| !index.is_multiple_of(7))),
    )
    .into_array();
    (lhs, rhs)
}

fn bench_row_fn<F>(bencher: Bencher, function: F, make_inputs: fn() -> (ArrayRef, ArrayRef))
where
    F: RowFn<Options = EmptyOptions>,
{
    bencher
        .with_inputs(make_inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            function
                .clone()
                .try_new_array(ROWS, EmptyOptions, [lhs, rhs])
                .unwrap()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn row_wrapping_add(bencher: Bencher) {
    bench_row_fn(bencher, RowWrappingAdd, inputs);
}

#[divan::bench]
fn row_sink_wrapping_add(bencher: Bencher) {
    bench_row_fn(bencher, RowSinkWrappingAdd, inputs);
}

#[divan::bench]
fn handrolled_sink_wrapping_add(bencher: Bencher) {
    bencher
        .with_inputs(inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            let lhs = lhs
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
                .into_buffer::<i64>();
            let rhs = rhs
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
                .into_buffer::<i64>();
            let mut output = BufferMut::zeroed(ROWS);
            for ((out, lhs), rhs) in output
                .as_mut_slice()
                .iter_mut()
                .zip(lhs.as_slice())
                .zip(rhs.as_slice())
            {
                *out = lhs.wrapping_add(*rhs);
            }
            PrimitiveArray::new(output.freeze(), Validity::NonNullable).into_array()
        });
}

#[divan::bench]
fn row_checked_add(bencher: Bencher) {
    bench_row_fn(bencher, RowCheckedAdd, inputs);
}

#[divan::bench]
fn row_wrapping_add_constant(bencher: Bencher) {
    bench_row_fn(bencher, RowWrappingAdd, constant_inputs);
}

#[divan::bench]
fn row_checked_add_constant(bencher: Bencher) {
    bench_row_fn(bencher, RowCheckedAdd, constant_inputs);
}

#[divan::bench]
fn row_checked_add_nullable(bencher: Bencher) {
    bench_row_fn(bencher, RowCheckedAdd, nullable_inputs);
}

#[divan::bench]
fn row_wrapping_add_nullable(bencher: Bencher) {
    bench_row_fn(bencher, RowWrappingAdd, nullable_inputs);
}
