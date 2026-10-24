// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The built-in aggregate functions, built on the `vortex_array::aggregate_fn` framework.
//!
//! This includes the aggregates that back array statistics (min/max, sum, null and NaN counts,
//! sortedness, constancy, uncompressed size and the `all_*` predicates), the general-purpose
//! [`count`], [`first`], [`last`] and [`mean`], the byte-bounded [`bounded_min`]/[`bounded_max`]
//! used by zone maps, and the [`combined`] adapter that composes two aggregates into one.
//!
//! Array statistics resolve their aggregate through the session's registry, so call
//! [`initialize`] on any session that computes statistics.

use vortex_array::VTable;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::ExtensionStoragePushdown;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::aggregate_fn::stat_aggregates::UncompressedSizeOpts;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Primitive;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::expr::stats::Stat;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::Time;
use vortex_array::extension::datetime::Timestamp;
use vortex_session::VortexSession;

pub mod all_nan;
pub mod all_non_nan;
pub mod all_non_null;
pub mod all_null;
pub mod bounded_max;
pub mod bounded_min;
pub mod combined;
pub mod count;
mod dict;
pub mod first;
mod fixed_size_bytes;
pub mod is_constant;
pub mod is_sorted;
pub mod last;
pub mod max;
pub mod mean;
pub mod min;
pub mod min_max;
pub mod nan_count;
pub mod null_count;
pub mod sum;
pub mod sum_v2;
pub mod uncompressed_size_in_bytes;

#[cfg(test)]
mod combined_tests;
#[cfg(test)]
mod framework_tests;

/// Register this crate's aggregate functions and kernels with the given session.
pub fn initialize(session: &VortexSession) {
    let aggregate_fns = session.aggregate_fns();

    aggregate_fns.register(all_non_nan::AllNonNan);
    aggregate_fns.register(all_non_null::AllNonNull);
    aggregate_fns.register(all_nan::AllNan);
    aggregate_fns.register(all_null::AllNull);
    aggregate_fns.register(bounded_max::BoundedMax);
    aggregate_fns.register(bounded_min::BoundedMin);
    aggregate_fns.register(first::First);
    aggregate_fns.register(is_constant::IsConstant);
    aggregate_fns.register(is_sorted::IsSorted);
    aggregate_fns.register(last::Last);
    aggregate_fns.register(max::Max);
    aggregate_fns.register(min::Min);
    aggregate_fns.register(min_max::MinMax);
    aggregate_fns.register(nan_count::NanCount);
    aggregate_fns.register(null_count::NullCount);
    aggregate_fns.register(sum::Sum);
    aggregate_fns.register(sum_v2::SumV2);
    aggregate_fns.register(uncompressed_size_in_bytes::UncompressedSizeInBytes);

    // Temporal extension types compare like their integer storage, so comparison-based
    // aggregates are pushed down into it.
    register_storage_ordered(session, Date);
    register_storage_ordered(session, Time);
    register_storage_ordered(session, Timestamp);

    aggregate_fns.register_aggregate_kernel(
        Dict.id(),
        Some(min_max::MinMax.id()),
        &dict::min_max::DictMinMaxKernel,
    );
    aggregate_fns.register_aggregate_kernel(
        Dict.id(),
        Some(is_constant::IsConstant.id()),
        &dict::is_constant::DictIsConstantKernel,
    );
    aggregate_fns.register_aggregate_kernel(
        Dict.id(),
        Some(is_sorted::IsSorted.id()),
        &dict::is_sorted::DictIsSortedKernel,
    );

    aggregate_fns.register_grouped_kernel(count::Count.id(), &count::CountGroupedKernel);
    aggregate_fns.register_grouped_encoding_kernel(
        Primitive.id(),
        sum::Sum.id(),
        &sum::PrimitiveGroupedSumEncodingKernel,
    );
    aggregate_fns.register_grouped_encoding_kernel(
        Primitive.id(),
        sum_v2::SumV2.id(),
        &sum_v2::PrimitiveGroupedSumV2EncodingKernel,
    );
}

/// Push the comparison-based aggregates [`IsSorted`](is_sorted::IsSorted) and
/// [`MinMax`](min_max::MinMax) down into the storage of extension type `ext`, through
/// [`ExtensionStoragePushdown`].
///
/// Only call this when comparing two values of `ext` is the same as comparing their storage, as
/// for the temporal types and for UUIDs, which are stored big-endian. See
/// [`ExtensionStoragePushdown`] for how to check.
///
/// # Example
///
/// Registering one pair by hand and binding it like any other aggregate:
///
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_session::VortexSession;
/// # use vortex_array::VortexSessionExecute;
/// # use vortex_array::aggregate_fn::Accumulator;
/// # use vortex_array::aggregate_fn::DynAccumulator;
/// # use vortex_array::aggregate_fn::ExtensionStoragePushdown;
/// # use vortex_aggregates::is_sorted::IsSorted;
/// # use vortex_aggregates::is_sorted::IsSortedOptions;
/// # use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
/// # use vortex_array::arrays::ExtensionArray;
/// # use vortex_array::arrays::PrimitiveArray;
/// # use vortex_array::dtype::DType;
/// # use vortex_array::dtype::Nullability;
/// # use vortex_array::extension::datetime::TimeUnit;
/// # use vortex_array::extension::datetime::Timestamp;
/// # fn main() -> vortex_error::VortexResult<()> {
/// let session = VortexSession::empty();
///
/// // Register `vortex.is_sorted[vortex.timestamp]`.
/// let is_sorted = ExtensionStoragePushdown::new(IsSorted, Timestamp);
/// session.aggregate_fns().register(is_sorted.clone());
///
/// // Aggregate a timestamp column through its storage.
/// let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
/// let column = ExtensionArray::new(
///     ext_dtype.clone(),
///     PrimitiveArray::from_iter([1_000i64, 2_000, 3_000]).into_array(),
/// )
/// .into_array();
///
/// let mut acc = Accumulator::try_new(
///     is_sorted,
///     IsSortedOptions { strict: true },
///     DType::Extension(ext_dtype),
/// )?;
/// acc.accumulate(&column, &mut session.create_execution_ctx())?;
/// assert!(bool::try_from(&acc.finish()?)?);
/// # Ok(())
/// # }
/// ```
pub fn register_storage_ordered<E: ExtVTable>(session: &VortexSession, ext: E) {
    let aggregate_fns = session.aggregate_fns();
    aggregate_fns.register(ExtensionStoragePushdown::new(
        is_sorted::IsSorted,
        ext.clone(),
    ));
    aggregate_fns.register(ExtensionStoragePushdown::new(min_max::MinMax, ext));
}

/// Return the aggregate function backing `stat`, if the statistic has a stat slot.
///
/// Statistics follow NaN-skipping semantics, and the uncompressed size statistic is exact.
/// Sortedness and constancy have no stat slot, so they return `None`.
pub fn stat_aggregate_fn(stat: Stat) -> Option<AggregateFnRef> {
    Some(match stat {
        Stat::Max => max::Max.bind(NumericalAggregateOpts::skip_nans()),
        Stat::Min => min::Min.bind(NumericalAggregateOpts::skip_nans()),
        Stat::Sum => sum::Sum.bind(NumericalAggregateOpts::skip_nans()),
        Stat::NullCount => null_count::NullCount.bind(EmptyOptions),
        Stat::NaNCount => nan_count::NanCount.bind(EmptyOptions),
        Stat::UncompressedSizeInBytes => {
            uncompressed_size_in_bytes::UncompressedSizeInBytes.bind(UncompressedSizeOpts::exact())
        }
        Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => return None,
    })
}
