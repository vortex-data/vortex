// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native wall-time probes. Fixture construction is outside the timed calls.

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

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
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Warm each case, then calibrate repeated calls to at least two milliseconds per sample.
fn measure<T>(name: &str, rows: usize, mut run: impl FnMut() -> T) {
    for _ in 0..200 {
        drop(black_box(run()));
    }
    let mut iterations = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            drop(black_box(run()));
        }
        if start.elapsed() >= Duration::from_millis(2) {
            break;
        }
        iterations *= 2;
    }
    for sample in 0..12 {
        let start = Instant::now();
        for _ in 0..iterations {
            drop(black_box(run()));
        }
        println!(
            "{name},{rows},{sample},{iterations},{:.3}",
            start.elapsed().as_nanos() as f64 / iterations as f64
        );
    }
}

/// Dense-safe deferred predicates reach dense retry on partial inputs. The null-only failure
/// forces the same predicate to run again in ExecuteValidRows.
#[derive(Clone)]
struct Predicate<const MULTIVERSIONED: bool>;
impl<const M: bool> RowFn for Predicate<M> {
    type Options = EmptyOptions;
    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = false;
    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("research.predicate");
        *ID
    }
    fn dispatch<V: RowVisitor>(
        &self,
        _: &EmptyOptions,
        _: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred_bool::<(i64, i64), bool, M>(
            |(a, b)| (a < b, a == i64::MAX || b == i64::MAX),
            |failed| {
                vortex_ensure!(!failed, "predicate rejected sentinel");
                Ok(())
            },
        )
    }
}

fn inputs(rows: usize, partial: bool, rejected: bool, constant: u8) -> VecExecutionArgs {
    let validity = if partial {
        Validity::Array(BoolArray::from_iter((0..rows).map(|i| i % 8 != 0)).into_array())
    } else {
        Validity::NonNullable
    };
    let column = PrimitiveArray::new(
        (0..rows)
            .map(|i| {
                if rejected && i % 8 == 0 {
                    i64::MAX
                } else {
                    (i % 1024) as i64
                }
            })
            .collect::<Vec<_>>(),
        validity,
    )
    .into_array();
    let other = PrimitiveArray::from_iter((0..rows).map(|i| (i % 512 + 1) as i64)).into_array();
    let (lhs, rhs) = match constant {
        1 => (ConstantArray::new(512_i64, rows).into_array(), column),
        2 => (column, ConstantArray::new(512_i64, rows).into_array()),
        _ => (column, other),
    };
    VecExecutionArgs::new(vec![lhs, rhs], rows)
}

fn main() -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    println!("case,rows,sample,iterations,ns");
    for rows in [0, 1, 64, 1024, 16384] {
        for (shape, partial, rejected) in [
            ("dense", false, false),
            ("partial", true, false),
            ("retry", true, true),
            ("failure", false, true),
        ] {
            for constant in 0..3 {
                let args = inputs(rows, partial, rejected, constant);
                let result = execute_rows(&Predicate::<false>, &EmptyOptions, &args, &mut ctx);
                assert_eq!(result.is_err(), rejected && !partial && rows > 0);
                measure(&format!("bool_{shape}_c{constant}_plain"), rows, || {
                    execute_rows(
                        &Predicate::<false>,
                        &EmptyOptions,
                        black_box(&args),
                        &mut ctx,
                    )
                });
                measure(&format!("bool_{shape}_c{constant}_multi"), rows, || {
                    execute_rows(
                        &Predicate::<true>,
                        &EmptyOptions,
                        black_box(&args),
                        &mut ctx,
                    )
                });
            }
        }
    }
    Ok(())
}
