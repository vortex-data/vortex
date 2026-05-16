//! Engine-local aggregate kernels.
//!
//! Vortex's `DynAggregateKernel` registry is the right home for
//! encoding-specific aggregate compute. Vortex ships dict-aware
//! `MinMax`, `IsConstant`, `IsSorted` kernels; we add `Sum` here
//! and register it on the engine's [`VortexSession`].
//!
//! As of vortex develop (#7889 — "Reorder agg kernel dispatch,
//! and have Combined use inner accumulators"), `Combined<V>`'s
//! `try_accumulate` delegates each batch to its child accumulators,
//! and `Accumulator::accumulate` consults the kernel registry
//! per-child. A single `(Dict, Sum)` kernel therefore powers any
//! aggregate built on `Sum` — `Sum` itself, `Mean` (via the inner
//! `Sum` child of `Combined<Mean>`), and anything else that
//! composes `Sum` via `BinaryCombined`.
//!
//! [`TimestampSum`] is a hack — a local aggregate fn that accepts
//! `Timestamp[unit]` extension input and returns an i64 partial.
//! Avoids extending vortex's `Sum` to handle extension types. The
//! `(DateTimeParts, TimestampSum)` kernel computes the sum per
//! component (days/seconds/subseconds), skipping the materialisation
//! that canonical `Sum` would pay.

mod dict_sum;
mod dict_value_counts;
mod dtp_timestamp_sum;
mod timestamp_sum;
mod value_counts;

pub use dict_sum::DictSumKernel;
pub use dict_value_counts::DictValueCountsKernel;
pub use dtp_timestamp_sum::DateTimePartsTimestampSumKernel;
pub use timestamp_sum::TimestampSum;
pub use value_counts::ValueCounts;

use vortex_array::ArrayId;
use vortex_array::VTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::Dict;
use vortex_session::VortexSession;

use crate::kernels::TimestampSum as TimestampSumAggFn;
use crate::kernels::ValueCounts as ValueCountsAggFn;

/// Install all engine-local aggregate kernels on `session`. Idempotent
/// — re-registering replaces existing kernels for the same
/// `(encoding, aggregate_fn)` key.
pub fn install(session: &VortexSession) {
    static DICT_SUM: DictSumKernel = DictSumKernel;
    static DICT_VALUE_COUNTS: DictValueCountsKernel = DictValueCountsKernel;
    static DTP_TIMESTAMP_SUM: DateTimePartsTimestampSumKernel =
        DateTimePartsTimestampSumKernel;
    let aggregate_fns = session.aggregate_fns();
    aggregate_fns.register(ValueCounts);
    aggregate_fns.register(TimestampSum);
    aggregate_fns.register_aggregate_kernel(Dict.id(), Some(Sum.id()), &DICT_SUM);
    aggregate_fns.register_aggregate_kernel(
        Dict.id(),
        Some(ValueCountsAggFn.id()),
        &DICT_VALUE_COUNTS,
    );
    // (DateTimeParts, TimestampSum) — keyed by the encoding id
    // string since the DateTimeParts ArrayId isn't directly
    // accessible from this crate (we don't depend on the
    // vortex-datetime-parts crate).
    aggregate_fns.register_aggregate_kernel(
        ArrayId::new("vortex.datetimeparts"),
        Some(TimestampSumAggFn.id()),
        &DTP_TIMESTAMP_SUM,
    );
}
