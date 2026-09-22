// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native wall-time probes. Fixture construction is outside the timed calls.

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::VecExecutionArgs;
use vortex_array::scalar_fn::unstable::row::InputElement;
use vortex_array::scalar_fn::unstable::row::OutputElement;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_array::scalar_fn::unstable::row::Utf8Column;
use vortex_array::scalar_fn::unstable::row::execute_rows;
use vortex_array::scalar_fn::unstable::row::row_fn_return_dtype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::vector::Vector;

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

#[derive(Clone)]
struct AddOne;
impl RowFn for AddOne {
    type Options = EmptyOptions;
    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;
    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("research.add_one");
        *ID
    }
    fn dispatch<V: RowVisitor>(
        &self,
        _: &EmptyOptions,
        _: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), i64>(|(v,)| v.wrapping_add(1))
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

#[derive(Clone)]
struct CheckedAdd;
impl RowFn for CheckedAdd {
    type Options = EmptyOptions;
    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = false;
    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("research.checked_add");
        *ID
    }
    fn dispatch<V: RowVisitor>(
        &self,
        _: &EmptyOptions,
        _: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred::<(i64, i64), i64, bool>(
            |(a, b)| a.overflowing_add(b),
            |failed| {
                vortex_ensure!(!failed, "integer overflow in checked add");
                Ok(())
            },
        )
    }
}

/// Instrument only construction of the diagnostic that nullable retry subsequently discards.
#[derive(Clone)]
struct MeasuredAdd {
    diagnostic_ns: Arc<AtomicU64>,
    rejections: Arc<AtomicU64>,
}
impl RowFn for MeasuredAdd {
    type Options = EmptyOptions;
    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = false;
    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("research.measured_add");
        *ID
    }
    fn dispatch<V: RowVisitor>(
        &self,
        _: &EmptyOptions,
        _: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred::<(i64, i64), i64, bool>(
            |(a, b)| a.overflowing_add(b),
            |failed| {
                if failed {
                    let start = Instant::now();
                    let error = vortex_err!("integer overflow in checked add");
                    let elapsed = start.elapsed().as_nanos() as u64;
                    self.diagnostic_ns.fetch_add(elapsed, Ordering::Relaxed);
                    self.rejections.fetch_add(1, Ordering::Relaxed);
                    return Err(error);
                }
                Ok(())
            },
        )
    }
}

fn measure_discard(ctx: &mut ExecutionCtx) -> VortexResult<()> {
    let rows = 64;
    let args = inputs(rows, true, true, 0);
    let function = MeasuredAdd {
        diagnostic_ns: Arc::default(),
        rejections: Arc::default(),
    };
    for _ in 0..200 {
        black_box(execute_rows(&function, &EmptyOptions, &args, ctx)?);
    }
    for sample in 0..12 {
        function.diagnostic_ns.store(0, Ordering::Relaxed);
        function.rejections.store(0, Ordering::Relaxed);
        let start = Instant::now();
        for _ in 0..256 {
            // Success plus exactly one rejection proves that each timed diagnostic was discarded.
            black_box(execute_rows(&function, &EmptyOptions, &args, ctx)?);
        }
        let outer_ns = start.elapsed().as_nanos() as f64 / 256.0;
        assert_eq!(function.rejections.load(Ordering::Relaxed), 256);
        let diagnostic_ns = function.diagnostic_ns.load(Ordering::Relaxed) as f64 / 256.0;
        println!("discarded_error_construct,{rows},{sample},256,{diagnostic_ns:.3}");
        println!("discarded_error_outer_instrumented,{rows},{sample},256,{outer_ns:.3}");
    }
    Ok(())
}

/// A native input with conservative or null-tolerant selected decoding.
struct SelectedI64<const FILTER: bool>;
// SAFETY: all access delegates to the native i64 input, with identical views and lengths.
unsafe impl<const FILTER: bool> InputElement for SelectedI64<FILTER> {
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
    fn decode_constant(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<i64> {
        <i64 as InputElement>::decode_constant(array, ctx)
    }
    fn can_decode_null_tolerant(_: &ArrayRef) -> VortexResult<bool> {
        Ok(!FILTER)
    }
    fn get(column: &Self::Column, index: usize) -> i64 {
        column[index]
    }
    fn get_constant(value: &i64) -> i64 {
        *value
    }
    fn view(column: &Self::Column) -> &[i64] {
        column.as_slice()
    }
    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a,
    {
        view[index]
    }
    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a,
    {
        // SAFETY: forwarded from the caller's index bound.
        unsafe { *view.get_unchecked(index) }
    }
}

#[derive(Clone)]
struct SelectedPredicate<const FILTER: bool, const MULTI: bool>;
impl<const FILTER: bool, const MULTI: bool> RowFn for SelectedPredicate<FILTER, MULTI> {
    type Options = EmptyOptions;
    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = false;
    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("research.selected");
        *ID
    }
    fn dispatch<V: RowVisitor>(
        &self,
        _: &EmptyOptions,
        _: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred_bool::<(SelectedI64<FILTER>, SelectedI64<FILTER>), bool, MULTI>(
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

fn vectors(rows: usize, width: usize, seed: usize) -> ArrayRef {
    let elements: Buffer<f64> = (0..rows * width)
        .map(|i| ((i + seed) % 97) as f64 - 48.0)
        .collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        width as u32,
        Validity::NonNullable,
        rows,
    )
    .into_array();
    Vector::try_new_vector_array(storage).unwrap()
}

fn main() -> VortexResult<()> {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    println!("case,rows,sample,iterations,ns");
    if "discard".contains(&filter) {
        measure_discard(&mut ctx)?;
    }
    if "error".contains(&filter) {
        // Direct construction and destruction, with no cross-scenario subtraction.
        measure("error_construct_drop", 0, || {
            vortex_err!("integer overflow in checked add")
        });
    }
    for rows in [0, 1, 64, 1024, 16384] {
        if "utf8".contains(&filter) {
            for (kind, value) in [
                ("inline", "hello"),
                ("external", "a longer external UTF-8 string λ"),
            ] {
                let array =
                    VarBinViewArray::from_iter_str(std::iter::repeat_n(value, rows)).into_array();
                let decoded = Utf8Column::decode(array.clone(), &mut ctx)?;
                if rows > 0 {
                    assert_eq!(Utf8Column::get(&decoded, rows - 1).as_str(), value);
                }
                measure(&format!("utf8_decode_{kind}"), rows, || {
                    Utf8Column::decode(black_box(array.clone()), &mut ctx).unwrap()
                });
            }
        }
        if "invocation".contains(&filter) {
            let array = PrimitiveArray::from_iter((0..rows).map(|i| i as i64)).into_array();
            let args = VecExecutionArgs::new(vec![array.clone()], rows);
            measure("invocation_rowfn", rows, || {
                execute_rows(&AddOne, &EmptyOptions, black_box(&args), &mut ctx).unwrap()
            });
            measure("invocation_fresh_args", rows, || {
                let args = VecExecutionArgs::new(vec![array.clone()], rows);
                execute_rows(&AddOne, &EmptyOptions, black_box(&args), &mut ctx).unwrap()
            });
            measure("invocation_direct_collector", rows, || {
                direct(&array, &mut ctx, true).unwrap()
            });
            measure("invocation_direct_iterator", rows, || {
                direct(&array, &mut ctx, false).unwrap()
            });
            measure("invocation_clone_drop", rows, || black_box(&array).clone());
            measure("invocation_argument_storage", rows, || {
                Vec::<ArrayRef>::with_capacity(black_box(1))
            });
            measure("invocation_validate", rows, || {
                <i64 as InputElement>::validate(black_box(array.dtype())).unwrap()
            });
            let dtypes = [array.dtype().clone()];
            measure("invocation_plan", rows, || {
                row_fn_return_dtype(&AddOne, &EmptyOptions, black_box(&dtypes)).unwrap()
            });
        }
        if "selected".contains(&filter) {
            for constant in 0..3 {
                let args = inputs(rows, true, true, constant);
                measure(&format!("selected_valid_c{constant}_plain"), rows, || {
                    execute_rows(
                        &SelectedPredicate::<false, false>,
                        &EmptyOptions,
                        &args,
                        &mut ctx,
                    )
                });
                measure(&format!("selected_valid_c{constant}_multi"), rows, || {
                    execute_rows(
                        &SelectedPredicate::<false, true>,
                        &EmptyOptions,
                        &args,
                        &mut ctx,
                    )
                });
                measure(
                    &format!("selected_filtered_c{constant}_plain"),
                    rows,
                    || {
                        execute_rows(
                            &SelectedPredicate::<true, false>,
                            &EmptyOptions,
                            &args,
                            &mut ctx,
                        )
                    },
                );
                measure(
                    &format!("selected_filtered_c{constant}_multi"),
                    rows,
                    || {
                        execute_rows(
                            &SelectedPredicate::<true, true>,
                            &EmptyOptions,
                            &args,
                            &mut ctx,
                        )
                    },
                );
            }
        }
        if "bool".contains(&filter) || "retry".contains(&filter) {
            for (shape, partial, rejected) in [
                ("dense", false, false),
                ("partial", true, false),
                ("retry", true, true),
                ("failure", false, true),
            ] {
                for constant in 0..3 {
                    let args = inputs(rows, partial, rejected, constant);
                    if "bool".contains(&filter) {
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
                    if constant == 0 && "retry".contains(&filter) {
                        measure(&format!("retry_{shape}"), rows, || {
                            execute_rows(&CheckedAdd, &EmptyOptions, black_box(&args), &mut ctx)
                        });
                    }
                }
            }
        }
    }
    if "cosine".contains(&filter) {
        for width in [1, 2, 3, 4, 8, 16, 32, 256] {
            let rows = 1024;
            for constant in 0..3 {
                let column = vectors(rows, width, 0);
                let other = vectors(rows, width, 31);
                let query: Vec<f64> = (0..width).map(|i| i as f64 - 48.0).collect();
                let element_dtype =
                    DType::Primitive(vortex_array::dtype::PType::F64, Nullability::NonNullable);
                let storage = Scalar::fixed_size_list(
                    element_dtype,
                    query.into_iter().map(Scalar::from).collect(),
                    Nullability::NonNullable,
                );
                let query =
                    ConstantArray::new(Scalar::extension::<Vector>(EmptyMetadata, storage), rows)
                        .into_array();
                let (lhs, rhs) = match constant {
                    1 => (query, column),
                    2 => (column, query),
                    _ => (column, other),
                };
                let args = VecExecutionArgs::new(vec![lhs, rhs], rows);
                measure(&format!("cosine_w{width}_c{constant}"), rows, || {
                    execute_rows(&CosineSimilarity, &EmptyOptions, black_box(&args), &mut ctx)
                        .unwrap()
                });
            }
        }
    }
    Ok(())
}

#[inline(never)]
fn direct(array: &ArrayRef, ctx: &mut ExecutionCtx, collector: bool) -> VortexResult<ArrayRef> {
    let input = array
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<i64>();
    Ok(if collector {
        i64::build_from(input.as_slice(), |v| v.wrapping_add(1))
    } else {
        PrimitiveArray::new(
            input
                .as_slice()
                .iter()
                .map(|v| v.wrapping_add(1))
                .collect::<Vec<_>>(),
            Validity::NonNullable,
        )
        .into_array()
    })
}
