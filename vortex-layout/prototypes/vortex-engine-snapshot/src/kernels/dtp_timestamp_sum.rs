//! `(DateTimeParts, TimestampSum)` kernel.
//!
//! Computes `SUM(t) = SUM(days)·86_400·divisor + SUM(seconds)·divisor + SUM(subseconds)`
//! directly from the DateTimeParts component arrays. Skips the
//! materialisation that canonical Sum would pay (decoding DTP →
//! TemporalArray → primitive i64).
//!
//! Designed to compose with the DateTimeParts subtract pushdown in
//! vortex: a query like `SUM(EventTime - START_TS)` rewrites to
//! `SUM(shifted_dtp)` where the components are subtracted, then this
//! kernel sums the components in their (now-small) range. Per-batch
//! and per-shard totals stay within i64 when START_TS is chosen close
//! to the data's mean.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::Dict;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::kernels::TimestampSum;

/// Days in seconds: 24 * 60 * 60.
const SECONDS_PER_DAY: i64 = 86_400;

#[derive(Debug)]
pub struct DateTimePartsTimestampSumKernel;

impl DynAggregateKernel for DateTimePartsTimestampSumKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<TimestampSum>() {
            return Ok(None);
        }

        // We dispatch on `(DateTimeParts, TimestampSum)`. Use a
        // string match on the encoding id rather than a typed `as_opt`
        // so we don't have to take a compile-time dependency on the
        // vortex-datetime-parts crate from here. (The dispatch
        // registry already keys on `ArrayId`.)
        if batch.encoding_id().as_str() != "vortex.datetimeparts" {
            return Ok(None);
        }
        // Dict-wrapped DTP would route through the `Dict` kernel
        // first; if anything reached us via that path, bail out.
        if batch.is::<Dict>() {
            return Ok(None);
        }

        // Pull the three component arrays out via DateTimeParts'
        // slot accessor. We rely on the structural invariants of
        // DateTimeParts (3 children: days, seconds, subseconds in
        // that order) — same invariants the canonical Sum path
        // would walk after canonicalisation, just stopped one step
        // earlier.
        let slots = batch.slots();
        if slots.len() != 3 {
            return Err(vortex_err!(
                "DateTimePartsTimestampSumKernel: expected 3 slots, got {}",
                slots.len()
            ));
        }
        let days = slots[0]
            .as_ref()
            .ok_or_else(|| vortex_err!("DateTimeParts: missing days slot"))?;
        let seconds = slots[1]
            .as_ref()
            .ok_or_else(|| vortex_err!("DateTimeParts: missing seconds slot"))?;
        let subseconds = slots[2]
            .as_ref()
            .ok_or_else(|| vortex_err!("DateTimeParts: missing subseconds slot"))?;

        // Time unit from outer extension dtype.
        let DType::Extension(ext_dtype) = batch.dtype() else {
            return Ok(None);
        };
        let Some(options) = ext_dtype.metadata_opt::<Timestamp>() else {
            // Not a timestamp extension — bail.
            return Ok(None);
        };
        let divisor = match options.unit {
            TimeUnit::Nanoseconds => 1_000_000_000_i64,
            TimeUnit::Microseconds => 1_000_000_i64,
            TimeUnit::Milliseconds => 1_000_i64,
            TimeUnit::Seconds => 1_i64,
            TimeUnit::Days => {
                return Err(vortex_err!(
                    "DateTimePartsTimestampSumKernel: TimeUnit::Days not supported"
                ));
            }
        };
        let day_scale = SECONDS_PER_DAY.checked_mul(divisor).ok_or_else(|| {
            vortex_err!("DateTimePartsTimestampSumKernel: day scale overflow")
        })?;

        let sum_days = sum_to_i64(days, ctx)?;
        let sum_seconds = sum_to_i64(seconds, ctx)?;
        let sum_subseconds = sum_to_i64(subseconds, ctx)?;

        // Combine: SUM(t) = SUM(days)*day_scale + SUM(seconds)*divisor + SUM(subseconds).
        // Treat any None as "saturated" → return null partial.
        let result = (|| -> Option<i64> {
            let part_d = sum_days?.checked_mul(day_scale)?;
            let part_s = sum_seconds?.checked_mul(divisor)?;
            let part_ss = sum_subseconds?;
            part_d.checked_add(part_s)?.checked_add(part_ss)
        })();
        Ok(Some(match result {
            Some(v) => Scalar::primitive(v, Nullability::Nullable),
            None => Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
        }))
    }
}

/// Sum a component array, returning its partial as i64.
///
/// Tries an algebraic shortcut first: if `arr` is a lazy
/// `Binary(Sub, [inner, const])` (the shape produced by
/// `DTPSubtractPushDownRule`), compute
/// `SUM(inner) - len * const` directly instead of materialising
/// the elementwise subtract via Arrow. Falls back to
/// `vortex.sum(arr)` for any other shape.
///
/// The shortcut is what lets the kernel honour the spirit of the
/// subtract pushdown — without it, the pushdown produces a lazy
/// ScalarFn that the canonical Sum path immediately re-materialises
/// (see the samply trace: ~30% of CPU was in `arrow_arith` doing
/// exactly this subtract). With the shortcut, the elementwise
/// subtract is never performed.
fn sum_to_i64(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<i64>> {
    if let Some(result) = try_algebraic_sub_sum(arr, ctx)? {
        return Ok(Some(result));
    }
    sum_to_i64_direct(arr, ctx)
}

/// Plain "materialise and sum" path. Used both as the fallback when
/// the algebraic shortcut doesn't match, and as the inner step of
/// the shortcut itself (sum of the un-subtracted operand).
fn sum_to_i64_direct(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<i64>> {
    let scalar = vortex_array::aggregate_fn::fns::sum::sum(arr, ctx)?;
    if scalar.is_null() {
        return Ok(None);
    }
    let prim = scalar.as_primitive();
    Ok(match scalar.dtype() {
        DType::Primitive(PType::I64, _) => prim.typed_value::<i64>(),
        DType::Primitive(PType::U64, _) => prim.typed_value::<u64>().and_then(|v| i64::try_from(v).ok()),
        _ => None,
    })
}

/// Detect `arr = Binary(Sub, [inner, const])` and compute
/// `SUM(inner) - len * const` algebraically.
///
/// Returns `None` if the shape doesn't match, or if any intermediate
/// would overflow i64 even after going through an i128 intermediate
/// for `len * const`.
fn try_algebraic_sub_sum(
    arr: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<i64>> {
    use vortex_array::arrays::ScalarFn;
    use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
    use vortex_array::scalar_fn::fns::binary::Binary;
    use vortex_array::scalar_fn::fns::operators::Operator;

    let Some(sf_view) = arr.as_opt::<ScalarFn>() else {
        return Ok(None);
    };
    let scalar_fn = sf_view.scalar_fn();
    let Some(op) = scalar_fn.as_opt::<Binary>() else {
        return Ok(None);
    };
    if *op != Operator::Sub {
        return Ok(None);
    }
    if sf_view.nchildren() != 2 {
        return Ok(None);
    }
    let inner = sf_view.get_child(0);
    let rhs = sf_view.get_child(1);
    let Some(rhs_const) = rhs.as_constant() else {
        return Ok(None);
    };
    // Extract the constant as i64. Components are signed/unsigned
    // integers (i32 days, u32 secs, u32 subsecs typically); the
    // constant was cast to the same dtype at push-down time, so
    // either i64 or u64 should round-trip.
    let prim = rhs_const.as_primitive();
    let k_i64: i64 = match rhs_const.dtype() {
        DType::Primitive(p, _) => match p {
            PType::I8 => i64::from(prim.typed_value::<i8>().unwrap_or(0)),
            PType::I16 => i64::from(prim.typed_value::<i16>().unwrap_or(0)),
            PType::I32 => i64::from(prim.typed_value::<i32>().unwrap_or(0)),
            PType::I64 => prim.typed_value::<i64>().unwrap_or(0),
            PType::U8 => i64::from(prim.typed_value::<u8>().unwrap_or(0)),
            PType::U16 => i64::from(prim.typed_value::<u16>().unwrap_or(0)),
            PType::U32 => i64::from(prim.typed_value::<u32>().unwrap_or(0)),
            PType::U64 => i64::try_from(prim.typed_value::<u64>().unwrap_or(0))
                .ok()
                .unwrap_or(i64::MAX),
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };

    let Some(inner_sum) = sum_to_i64_direct(inner, ctx)? else {
        return Ok(None);
    };

    // Compute `len * k` in i128 to avoid overflow on large len*k, then
    // narrow back to i64 for the subtract. Vortex doesn't ship an i128
    // dtype so we can't keep wider intermediates around — but for
    // ClickBench-shaped inputs (k ≤ 22000 days × 1M rows = 2.2e10),
    // everything fits i64 comfortably; the i128 is an out-of-paranoia
    // guard for callers we don't know about.
    let len = arr.len() as i128;
    let len_k = len.saturating_mul(k_i64 as i128);
    let shifted_i128 = (inner_sum as i128).checked_sub(len_k);
    let Some(shifted) = shifted_i128 else {
        return Ok(None);
    };
    Ok(i64::try_from(shifted).ok())
}
