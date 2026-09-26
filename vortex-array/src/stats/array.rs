// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stats as they are stored on arrays.
//!
//! Stats are a write-once, read-many cache of aggregate results over an immutable array. A
//! value is either seeded when the array is built, or computed on first request by
//! [`StatsSetRef::get`]. Stored values never change, so readers never lock.

use std::any::type_name;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::ExecutionCtx;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::StatsSet;
use super::lazy_arc::LazyArc;
use super::list::Entry;
use super::list::StatsList;
use crate::ArrayRef;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnSatisfaction;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_constant::constant_from_metadata;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::sorted_from_metadata;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A shared, write-once [`StatsSet`] stored in an array.
///
/// Storage is allocated lazily on first write, so arrays that never carry stats pay no
/// allocation. Arrays over the same logical values may share one storage.
// TODO(adamg): This is a very bad name.
#[derive(Clone, Default)]
pub struct ArrayStats {
    inner: LazyArc<StatsList>,
}

/// Reference to an array's stats. Reads stored values and computes missing ones on request.
///
/// Constructed by calling [`ArrayStats::to_ref`].
pub struct StatsSetRef<'a> {
    // We need to reference back to the array
    dyn_array_ref: &'a ArrayRef,
    array_stats: &'a ArrayStats,
}

impl ArrayStats {
    pub fn to_ref<'a>(&'a self, array: &'a ArrayRef) -> StatsSetRef<'a> {
        StatsSetRef {
            dyn_array_ref: array,
            array_stats: self,
        }
    }

    /// Stores `entry` unless a value is already known.
    pub(crate) fn insert(&self, entry: Entry) {
        self.list_or_init().insert(entry);
    }

    /// Iterates every stored entry.
    pub(crate) fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.list().into_iter().flat_map(StatsList::iter)
    }

    /// Builds storage seeded with `entries`, which must have distinct keys.
    pub(crate) fn from_entries(entries: Vec<Entry>) -> Self {
        if entries.is_empty() {
            return Self::default();
        }
        Self {
            inner: LazyArc::from_arc(Arc::new(StatsList::from_seed(entries))),
        }
    }

    fn list(&self) -> Option<&StatsList> {
        self.inner.get()
    }

    /// Returns the storage, allocating it on first use.
    fn list_or_init(&self) -> &StatsList {
        self.inner.get_or_init(StatsList::default)
    }

    /// Returns the value stored for `key`, preferring an exact value over a bound.
    fn get_value(&self, key: &AggregateFnRef) -> Option<&Precision<ScalarValue>> {
        let builtin = is_static_key(key);
        let mut bound = None;
        for entry in self.entries().filter(|entry| entry.is_for(key, builtin)) {
            if entry.value.is_exact() {
                return Some(&entry.value);
            }
            bound = Some(&entry.value);
        }
        bound
    }

    /// Iterates the best value of each stored key: an exact value hides a bound for the same key.
    pub(crate) fn best_entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries().filter(|entry| {
            entry.value.is_exact()
                || !self
                    .entries()
                    .any(|other| other.value.is_exact() && other.is_for(&entry.key, entry.builtin))
        })
    }

    /// Returns true if no storage was allocated yet.
    pub(crate) fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns true if both share the same allocated storage.
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        self.inner.ptr_eq(&other.inner)
    }
}

impl Debug for ArrayStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.entries()).finish()
    }
}

/// Normalises legacy stats, e.g. read from a file, to aggregate-keyed entries.
impl From<StatsSet> for ArrayStats {
    fn from(value: StatsSet) -> Self {
        // A plain loop: iterator adapters would each move the 350-byte set by value
        let mut entries = Vec::with_capacity(value.len());
        for (stat, value) in value {
            entries.push(Entry::builtin(stat.aggregate_fn(), value));
        }
        Self::from_entries(entries)
    }
}

impl StatsSetRef<'_> {
    /// Returns the stored value that best satisfies `agg`, without computing anything.
    ///
    /// Use this where computing is not acceptable, e.g. pruning over lazily loaded arrays.
    pub fn get_cached(&self, agg: &AggregateFnRef) -> Precision<Scalar> {
        let Some(list) = self.array_stats.list() else {
            return Precision::Absent;
        };
        let dtype = self.dyn_array_ref.dtype();
        let scalar = |key: &AggregateFnRef, value: &ScalarValue| {
            let scalar_dtype = key
                .return_dtype(dtype)
                .vortex_expect("stored aggregate must be defined for the array dtype");
            Scalar::try_new(scalar_dtype, Some(value.clone()))
                .vortex_expect("stored aggregate value must match its return dtype")
        };

        let builtin = is_static_key(agg);
        let mut bound = Precision::Absent;
        for entry in list.iter() {
            if entry.is_for(agg, builtin) {
                match &entry.value {
                    Precision::Exact(value) => return Precision::Exact(scalar(agg, value)),
                    Precision::Inexact(value) if bound.is_absent() => {
                        bound = Precision::Inexact(scalar(agg, value));
                    }
                    Precision::Inexact(_) | Precision::Absent => {}
                }
            } else if bound.is_absent()
                && !(builtin && entry.builtin)
                && let Precision::Exact(value) = &entry.value
                && entry.key.can_satisfy(agg) == AggregateFnSatisfaction::Approximate
            {
                // A different aggregate that bounds this one, e.g. a truncated maximum
                bound = Precision::Inexact(scalar(&entry.key, value));
            }
        }
        bound
    }

    /// Like [`Self::get_cached`], converting the value to `U`.
    ///
    /// # Panics
    ///
    /// Panics if the stored value cannot be converted to `U`.
    pub fn get_cached_as<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        agg: &AggregateFnRef,
    ) -> Precision<U> {
        self.get_cached(agg).map(|value| {
            U::try_from(&value).unwrap_or_else(|err| {
                vortex_panic!(err, "Failed to get {} as {}", agg, type_name::<U>())
            })
        })
    }

    /// Returns the exact value of `agg`, computing and storing it if it is not known yet.
    ///
    /// This is the only place that stores computed stats. One computation stores every result
    /// it implies: `Min` and `Max` are computed by `MinMax`, which stores both, and a strict
    /// `IsSorted` of `true` also stores `IsSorted`.
    ///
    /// Returns `None` if neither metadata nor an accumulator can compute `agg` for this array,
    /// or it has no value, e.g. the minimum of an all-null array. Null results (e.g. an
    /// overflowed sum) are returned but not stored.
    ///
    /// No lock is held while computing, so concurrent callers may compute the same value; the
    /// first one stored wins.
    pub fn get(
        &self,
        agg: &AggregateFnRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if let Precision::Exact(scalar) = self.get_cached(agg) {
            return Ok(Some(scalar));
        }

        let dtype = self.dyn_array_ref.dtype();
        let on_behalf = computed_for(agg);
        let computed = on_behalf.unwrap_or(agg);
        let result = if computed.can_compute(dtype) {
            let mut accumulator = computed.accumulator(dtype)?;
            accumulator.accumulate(self.dyn_array_ref, ctx)?;
            accumulator.finish()?
        } else {
            // Boolean stats can be known from metadata even for dtypes their accumulators
            // cannot process, e.g. a null array is constant and a singleton is sorted.
            let value = if computed.is::<IsConstant>() {
                constant_from_metadata(self.dyn_array_ref, ctx)?
            } else if let Some(options) = computed.as_opt::<IsSorted>() {
                sorted_from_metadata(self.dyn_array_ref, options.strict, ctx)?
            } else {
                None
            };
            let Some(value) = value else {
                return Ok(None);
            };
            Scalar::bool(value, Nullability::NonNullable)
        };

        if on_behalf.is_none() {
            for (key, value) in derive(computed, &result) {
                self.store(key, value.clone());
            }
            if let Some(value) = result.value() {
                self.store(agg, value.clone());
            }
            return Ok(Some(result));
        }

        // Computed for `Min` or `Max`: store the derived extrema, not the `MinMax` struct.
        // Both extrema share the nullable input dtype.
        let mut requested = None;
        for (key, value) in derive(computed, &result) {
            if requested.is_none() && (key.ptr_eq(agg) || key == agg) {
                requested = Some(value.clone());
            }
            self.store(key, value.clone());
        }
        requested
            .map(|value| Scalar::try_new(dtype.as_nullable(), Some(value)))
            .transpose()
    }

    /// Like [`Self::get`], converting the value to `U`.
    ///
    /// Returns `None` if the value is not defined, null, or cannot be computed.
    ///
    /// # Panics
    ///
    /// Panics if the value cannot be converted to `U`.
    pub fn get_as<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        agg: &AggregateFnRef,
        ctx: &mut ExecutionCtx,
    ) -> Option<U> {
        let value = self
            .get(agg, ctx)
            .inspect_err(|err| tracing::warn!("Failed to compute {agg}: {err}"))
            .ok()??;
        if value.is_null() {
            return None;
        }
        Some(U::try_from(&value).unwrap_or_else(|err| {
            vortex_panic!(err, "Failed to compute {} as {}", agg, type_name::<U>())
        }))
    }

    /// Iterates the stored stats as `(aggregate, value)` pairs, best value per aggregate.
    ///
    /// Writers convert to their own vocabulary from here, e.g. [`Stat`] for the file format.
    pub fn iter(&self) -> impl Iterator<Item = (&AggregateFnRef, &Precision<ScalarValue>)> {
        self.array_stats
            .best_entries()
            .map(|entry| (&*entry.key, &entry.value))
    }

    /// Returns true if no stats are stored.
    pub(crate) fn is_empty(&self) -> bool {
        self.array_stats.is_empty()
    }

    /// Returns a handle to the current storage without allocating any.
    ///
    /// The handle does not see storage allocated after this call.
    pub(crate) fn handle(&self) -> ArrayStats {
        self.array_stats.clone()
    }

    /// Returns a handle that shares the storage, allocating it if needed.
    ///
    /// Pass it to [`ArrayRef::with_shared_stats`] on an array with the same logical values, so
    /// results computed on either array serve both. Take it before an array is consumed, e.g.
    /// by execution, so stats computed on it meanwhile still reach the result.
    pub fn to_array_stats(&self) -> ArrayStats {
        self.array_stats.list_or_init();
        self.array_stats.clone()
    }

    /// Returns the stored value of `stat` without copying it.
    pub(crate) fn value(&self, stat: Stat) -> Option<&Precision<ScalarValue>> {
        self.array_stats.get_value(stat.aggregate_fn())
    }

    /// Stores the exact `value` of `key`.
    fn store(&self, key: &AggregateFnRef, value: ScalarValue) {
        self.array_stats
            .insert(Entry::new(key, Precision::exact(value)));
    }
}

/// Returns the aggregate [`StatsSetRef::get`] computes to answer `agg`, if it is not `agg`.
///
/// `Min` and `Max` are computed by `MinMax`, so asking for one stores both.
fn computed_for(agg: &AggregateFnRef) -> Option<&'static AggregateFnRef> {
    let options = agg.as_opt::<Min>().or_else(|| agg.as_opt::<Max>())?;
    Some(min_max_key(options.skip_nans))
}

/// Returns the static `MinMax` key for the given NaN handling.
pub(crate) fn min_max_key(skip_nans: bool) -> &'static AggregateFnRef {
    static SKIP_NANS: LazyLock<AggregateFnRef> =
        LazyLock::new(|| MinMax.bind(NumericalAggregateOpts::skip_nans()));
    static INCLUDE_NANS: LazyLock<AggregateFnRef> =
        LazyLock::new(|| MinMax.bind(NumericalAggregateOpts::include_nans()));

    if skip_nans { &SKIP_NANS } else { &INCLUDE_NANS }
}

/// Returns true if `agg` is one of the static built-in keys, comparing pointers only.
///
/// Lookups use this: a caller-bound equal aggregate still matches, through `==`.
fn is_static_key(agg: &AggregateFnRef) -> bool {
    Stat::static_from_aggregate_fn(agg).is_some()
        || [true, false]
            .into_iter()
            .map(min_max_key)
            .any(|key| key.ptr_eq(agg))
}

/// Returns the static instance of `agg` if it equals one of the built-in keys.
///
/// Canonicalises caller-bound aggregates, so every stored built-in is its static instance and
/// lookups of built-ins compare pointers only.
pub(crate) fn static_key(agg: &AggregateFnRef) -> Option<&'static AggregateFnRef> {
    if let Some(stat) = Stat::from_aggregate_fn(agg) {
        return Some(stat.aggregate_fn());
    }
    [true, false]
        .into_iter()
        .map(min_max_key)
        .find(|key| key.ptr_eq(agg) || **key == *agg)
}

/// Returns the other non-null results implied by `result` of `computed`.
///
/// Keys are static and values borrow from `result`, so nothing is cloned until stored.
fn derive<'a>(
    computed: &AggregateFnRef,
    result: &'a Scalar,
) -> impl Iterator<Item = (&'static AggregateFnRef, &'a ScalarValue)> {
    static TRUE: ScalarValue = ScalarValue::Bool(true);
    static FALSE: ScalarValue = ScalarValue::Bool(false);

    let mut derived: [Option<(&'static AggregateFnRef, &'a ScalarValue)>; 2] = [None, None];

    if let Some(options) = computed.as_opt::<MinMax>() {
        // A null struct means an empty or all-null array, which has no extrema
        if let Some(ScalarValue::Tuple(fields)) = result.value() {
            let (min, max) = extremum_keys(options.skip_nans);
            derived[0] = fields
                .first()
                .and_then(Option::as_ref)
                .map(|value| (min, value));
            derived[1] = fields
                .get(1)
                .and_then(Option::as_ref)
                .map(|value| (max, value));
        }
    } else if let Some(options) = computed.as_opt::<IsSorted>()
        && let Some(sorted) = result.as_bool().value()
    {
        // Strictly sorted implies sorted, and unsorted implies not strictly sorted
        if options.strict && sorted {
            derived[0] = Some((Stat::IsSorted.aggregate_fn(), &TRUE));
        } else if !options.strict && !sorted {
            derived[0] = Some((Stat::IsStrictSorted.aggregate_fn(), &FALSE));
        }
    }

    derived.into_iter().flatten()
}

/// Returns the static `Min` and `Max` keys for the given NaN handling.
fn extremum_keys(skip_nans: bool) -> (&'static AggregateFnRef, &'static AggregateFnRef) {
    static MIN_INCLUDE_NANS: LazyLock<AggregateFnRef> =
        LazyLock::new(|| Min.bind(NumericalAggregateOpts::include_nans()));
    static MAX_INCLUDE_NANS: LazyLock<AggregateFnRef> =
        LazyLock::new(|| Max.bind(NumericalAggregateOpts::include_nans()));

    // The NaN-skipping keys are the legacy stat slots
    if skip_nans {
        (Stat::Min.aggregate_fn(), Stat::Max.aggregate_fn())
    } else {
        (&MIN_INCLUDE_NANS, &MAX_INCLUDE_NANS)
    }
}

#[cfg(test)]
mod tests {
    use flatbuffers::FlatBufferBuilder;
    use flatbuffers::root;
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::count::Count;
    use crate::aggregate_fn::fns::is_constant::IsConstant;
    use crate::aggregate_fn::fns::is_constant::is_constant;
    use crate::aggregate_fn::fns::is_sorted::IsSorted;
    use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use crate::aggregate_fn::fns::is_sorted::is_sorted;
    use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::NullArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::flatbuffers::WriteFlatBuffer;
    use crate::flatbuffers::array as fba;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::stats::StatsSet;

    #[rstest]
    #[case::null_constant(NullArray::new(2).into_array(), Stat::IsConstant, true)]
    #[case::null_sorted(NullArray::new(2).into_array(), Stat::IsSorted, true)]
    #[case::null_strict_sorted(NullArray::new(2).into_array(), Stat::IsStrictSorted, false)]
    #[case::struct_sorted(StructArray::new_fieldless_with_len(2).into_array(), Stat::IsSorted, false)]
    fn legacy_boolean_stats_round_trip(
        #[case] array: ArrayRef,
        #[case] stat: Stat,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let session = array_session();
        let stats = StatsSet::of(stat, Precision::exact(expected));
        let mut fbb = FlatBufferBuilder::new();
        let offset = stats.write_flatbuffer(&mut fbb)?;
        fbb.finish(offset, None);
        let fb = root::<fba::ArrayStats>(fbb.finished_data())?;
        let stats = StatsSet::from_flatbuffer(&fb, array.dtype(), &session)?;
        let array = array.with_stats_set(stats);

        assert_eq!(
            array
                .statistics()
                .get_cached_as::<bool>(stat.aggregate_fn()),
            Precision::Exact(expected)
        );
        assert_eq!(
            array
                .statistics()
                .get(stat.aggregate_fn(), &mut session.create_execution_ctx())?,
            Some(Scalar::from(expected))
        );
        // Display also reads the stored flags without computing them.
        drop(array.tree_display().to_string());
        Ok(())
    }

    #[rstest]
    #[case::empty(0, false, true)]
    #[case::singleton(1, true, true)]
    #[case::multiple(2, true, false)]
    fn null_array_metadata_stats(
        #[case] len: usize,
        #[case] constant: bool,
        #[case] strict: bool,
    ) -> VortexResult<()> {
        let array = NullArray::new(len).into_array();
        let mut ctx = array_session().create_execution_ctx();

        assert_eq!(is_constant(&array, &mut ctx)?, constant);
        assert!(is_sorted(&array, &mut ctx)?);
        assert_eq!(is_strict_sorted(&array, &mut ctx)?, strict);
        for (stat, expected) in [
            (Stat::IsConstant, constant),
            (Stat::IsSorted, true),
            (Stat::IsStrictSorted, strict),
        ] {
            assert_eq!(
                array
                    .statistics()
                    .get_cached_as::<bool>(stat.aggregate_fn()),
                Precision::Exact(expected)
            );
        }
        Ok(())
    }

    #[rstest]
    fn struct_sortedness_from_metadata(
        #[values(0, 1, 2)] len: usize,
        #[values(false, true)] constant: bool,
    ) -> VortexResult<()> {
        let array = StructArray::new_fieldless_with_len(len).into_array();
        let array = if constant {
            ConstantArray::new(Scalar::struct_(array.dtype().clone(), vec![]), len).into_array()
        } else {
            array
        };
        let mut ctx = array_session().create_execution_ctx();

        assert_eq!(is_sorted(&array, &mut ctx)?, constant || len <= 1);
        assert_eq!(is_strict_sorted(&array, &mut ctx)?, len <= 1);
        assert!(
            Stat::IsSorted
                .aggregate_fn()
                .accumulator(array.dtype())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn get_stores_result() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4]).into_array();
        let count = Count.bind(NumericalAggregateOpts::default());
        let mut ctx = array_session().create_execution_ctx();

        assert!(array.statistics().get_cached(&count).is_absent());
        let computed = array.statistics().get(&count, &mut ctx)?;
        assert_eq!(array.statistics().get_cached(&count).as_exact(), computed);
        Ok(())
    }

    #[rstest]
    #[case::sum(Stat::Sum, Sum.bind(NumericalAggregateOpts::skip_nans()))]
    #[case::is_constant(Stat::IsConstant, IsConstant.bind(EmptyOptions))]
    #[case::is_sorted(Stat::IsSorted, IsSorted.bind(IsSortedOptions { strict: false }))]
    #[case::is_strict_sorted(Stat::IsStrictSorted, IsSorted.bind(IsSortedOptions { strict: true }))]
    fn aggregate_and_legacy_stat_share_a_slot(
        #[case] stat: Stat,
        #[case] agg: AggregateFnRef,
    ) -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let computed = array
            .statistics()
            .get(&agg, &mut array_session().create_execution_ctx())?;

        // Stored once, and found through either key
        assert_eq!(
            array
                .statistics()
                .iter()
                .filter(|(key, _)| key.ptr_eq(stat.aggregate_fn()))
                .count(),
            1
        );
        assert_eq!(
            array
                .statistics()
                .get_cached(stat.aggregate_fn())
                .as_exact(),
            computed
        );
        Ok(())
    }

    #[test]
    fn min_max_stores_both_extrema() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([3i32, 1, 2]).into_array();
        array.statistics().get(
            &MinMax.bind(NumericalAggregateOpts::skip_nans()),
            &mut array_session().create_execution_ctx(),
        )?;

        let nullable = array.dtype().as_nullable();
        assert_eq!(
            array.statistics().get_cached(Stat::Min.aggregate_fn()),
            Precision::Exact(Scalar::from(1i32).cast(&nullable)?)
        );
        assert_eq!(
            array.statistics().get_cached(Stat::Max.aggregate_fn()),
            Precision::Exact(Scalar::from(3i32).cast(&nullable)?)
        );
        Ok(())
    }

    #[test]
    fn min_stores_max_from_one_pass() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([3i32, 1, 2]).into_array();
        let min = array.statistics().get(
            Stat::Min.aggregate_fn(),
            &mut array_session().create_execution_ctx(),
        )?;

        assert_eq!(
            min,
            Some(Scalar::from(1i32).cast(&array.dtype().as_nullable())?)
        );
        assert!(
            array
                .statistics()
                .get_cached(Stat::Max.aggregate_fn())
                .is_exact()
        );
        Ok(())
    }
    #[rstest]
    // Strictly sorted implies sorted
    #[case::strict_true(vec![1i32, 2, 3], Stat::IsStrictSorted, Stat::IsSorted, true)]
    // Not sorted implies not strictly sorted
    #[case::sorted_false(vec![3i32, 1, 2], Stat::IsSorted, Stat::IsStrictSorted, false)]
    fn sortedness_stores_implied_key(
        #[case] values: Vec<i32>,
        #[case] requested: Stat,
        #[case] implied: Stat,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(values).into_array();
        array.statistics().get(
            requested.aggregate_fn(),
            &mut array_session().create_execution_ctx(),
        )?;

        assert_eq!(
            array
                .statistics()
                .get_cached_as::<bool>(implied.aggregate_fn()),
            Precision::Exact(expected)
        );
        Ok(())
    }

    #[test]
    fn metadata_shortcuts_store_their_result() -> VortexResult<()> {
        // A single value is constant without a scan; the answer is stored like any other
        let array = PrimitiveArray::from_iter([1i32]).into_array();
        assert!(is_constant(
            &array,
            &mut array_session().create_execution_ctx()
        )?);
        assert_eq!(
            array
                .statistics()
                .get_cached_as::<bool>(Stat::IsConstant.aggregate_fn()),
            Precision::Exact(true)
        );
        Ok(())
    }

    #[test]
    fn min_of_all_nulls_has_no_value() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None]).into_array();
        let min = array.statistics().get(
            Stat::Min.aggregate_fn(),
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(min, None);
        Ok(())
    }

    #[test]
    fn shared_handle_sees_later_results() -> VortexResult<()> {
        let source = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let handle = source.statistics().to_array_stats();
        source.statistics().get(
            Stat::Min.aggregate_fn(),
            &mut array_session().create_execution_ctx(),
        )?;

        let target = source.without_stats().with_shared_stats(&handle);
        assert!(
            target
                .statistics()
                .get_cached(Stat::Min.aggregate_fn())
                .is_exact()
        );
        Ok(())
    }

    #[test]
    fn shared_stats_carry_every_aggregate() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let count = Count.bind(NumericalAggregateOpts::default());
        let source = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        source.statistics().get(&count, &mut ctx)?;

        // The target already has stats, so entries are copied rather than shared
        let target = source
            .without_stats()
            .with_stats_set(StatsSet::of(Stat::NullCount, Precision::exact(0u64)))
            .with_shared_stats(&source.statistics().handle());
        assert!(target.statistics().get_cached(&count).is_exact());
        assert!(
            target
                .statistics()
                .get_cached(Stat::NullCount.aggregate_fn())
                .is_exact()
        );
        Ok(())
    }

    #[test]
    fn seeding_shared_array_leaves_original_unchanged() {
        let array = PrimitiveArray::from_iter([1i32, 2]).into_array();
        let seeded = array.clone().with_stats_set(StatsSet::of(
            Stat::Min,
            Precision::exact(ScalarValue::from(1i32)),
        ));

        assert!(
            array
                .statistics()
                .get_cached(Stat::Min.aggregate_fn())
                .is_absent()
        );
        assert_eq!(
            seeded.statistics().get_cached(Stat::Min.aggregate_fn()),
            Precision::Exact(Scalar::from(1i32))
        );
    }

    #[test]
    fn shared_stats_serve_both_arrays() -> VortexResult<()> {
        let source = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        source.statistics().get(
            Stat::Min.aggregate_fn(),
            &mut array_session().create_execution_ctx(),
        )?;
        let target = source.without_stats();
        assert!(
            target
                .statistics()
                .get_cached(Stat::Min.aggregate_fn())
                .is_absent()
        );

        let target = target.with_shared_stats(&source.statistics().handle());
        assert!(
            target
                .statistics()
                .get_cached(Stat::Min.aggregate_fn())
                .is_exact()
        );

        // A result computed on one array serves the other
        let count = Count.bind(NumericalAggregateOpts::default());
        let mut ctx = array_session().create_execution_ctx();
        target.statistics().get(&count, &mut ctx)?;
        assert!(source.statistics().get_cached(&count).is_exact());
        Ok(())
    }
}
