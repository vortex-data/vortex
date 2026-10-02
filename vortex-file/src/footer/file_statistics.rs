// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the file statistics component of the Vortex file footer.
//!
//! File statistics provide metadata about the data in the file, such as min/max values,
//! null counts, and other statistical information that can be used for query optimization
//! and data exploration.
use std::sync::Arc;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use itertools::Itertools;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::flatbuffers::FlatBufferRoot;
use vortex_array::flatbuffers::WriteFlatBuffer;
use vortex_array::flatbuffers::array::ArrayStats;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::stats::StatsSet;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_layout::layouts::file_stats::AggregateStat;
use vortex_layout::layouts::file_stats::AggregateStats;
use vortex_layout::layouts::file_stats::postorder_stats_layout;
use vortex_session::VortexSession;
use vortex_session::registry::Interner;
use vortex_session::registry::ReadContext;
use vortex_utils::aliases::hash_map::HashMap;

use crate::flatbuffers::footer as fb;

/// Contains statistical information about the data in a Vortex file.
///
/// Each entry holds the aggregates recorded for a field of the file. These statistics can be used
/// for query optimization and data exploration.
#[derive(Clone, Debug)]
pub struct FileStatistics {
    /// The aggregates recorded for each field or column in the file, following the post-order
    /// nested-struct layout.
    aggregates: Arc<[AggregateStats]>,
    /// An array of `DType`s, one for each field or column in the file.
    dtypes: Arc<[DType]>,
    /// An array of field paths, one for each field or column in the file. Parallel to `aggregates`
    /// and `dtypes`. For files written before nested field stats, every path has depth 1 (or is the
    /// root path, for a non-struct file dtype).
    paths: Arc<[FieldPath]>,
    /// Maps each entry in `paths` to its index, so [`Self::get_by_path`] doesn't need to scan
    /// `paths` linearly.
    path_index: Arc<HashMap<FieldPath, usize>>,
    /// Legacy top-level-fields-only aggregates, one per top-level struct field (or a single entry
    /// for a non-struct root dtype). Only populated by [`Self::new_with_dtype`], for serialization
    /// into `field_stats`; empty for instances built from [`Self::from_flatbuffer`], which are
    /// never re-serialized.
    legacy_aggregates: Arc<[AggregateStats]>,
}

impl FileStatistics {
    /// Builds `path_index` from `paths` and assembles the final struct. The single place that
    /// constructs a [`FileStatistics`], so `path_index` can't drift out of sync with `paths`.
    fn from_parts(
        aggregates: Arc<[AggregateStats]>,
        dtypes: Arc<[DType]>,
        paths: Arc<[FieldPath]>,
        legacy_aggregates: Arc<[AggregateStats]>,
    ) -> Self {
        let path_index = paths
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, p)| (p, i))
            .collect();
        Self {
            aggregates,
            dtypes,
            paths,
            path_index: Arc::new(path_index),
            legacy_aggregates,
        }
    }

    /// Creates a new [`FileStatistics`] from the given aggregates, data types, and field paths.
    ///
    /// # Panics
    ///
    /// Panics if `aggregates`, `dtypes`, and `paths` have different lengths.
    pub fn new(
        aggregates: Arc<[AggregateStats]>,
        dtypes: Arc<[DType]>,
        paths: Arc<[FieldPath]>,
    ) -> Self {
        assert_eq!(
            aggregates.len(),
            dtypes.len(),
            "aggregates and dtypes must have the same length"
        );
        assert_eq!(
            aggregates.len(),
            paths.len(),
            "aggregates and paths must have the same length"
        );
        Self::from_parts(aggregates, dtypes, paths, Arc::new([]))
    }

    /// Creates a new [`FileStatistics`] from the given nested aggregates, legacy top-level-only
    /// aggregates, and file dtype.
    ///
    /// `aggregates` must follow the post-order nested-struct layout produced by
    /// [`postorder_stats_layout`] for `file_dtype`. `legacy_aggregates` must have one entry per
    /// top-level struct field (or a single entry for a non-struct root dtype); they are written
    /// in the legacy `ArrayStats` format for readers that predate nested field statistics.
    ///
    /// # Panics
    ///
    /// Panics if the number of entries doesn't match the expected number based on the dtype.
    pub fn new_with_dtype(
        aggregates: Arc<[AggregateStats]>,
        legacy_aggregates: Arc<[AggregateStats]>,
        file_dtype: &DType,
    ) -> Self {
        let layout = postorder_stats_layout(file_dtype);
        assert_eq!(
            aggregates.len(),
            layout.len(),
            "aggregates length must match the post-order stats layout for the file dtype"
        );

        let (paths, dtypes): (Vec<FieldPath>, Vec<DType>) = layout.into_iter().unzip();
        Self::from_parts(aggregates, dtypes.into(), paths.into(), legacy_aggregates)
    }

    /// Creates [`FileStatistics`] from a flatbuffers [`fb::FileStatistics<'a>`].
    ///
    /// Aggregates the session doesn't know are skipped: file statistics are optional, so a reader
    /// without an aggregate loses only what that aggregate could tell it.
    pub fn from_flatbuffer<'a>(
        fb: &fb::FileStatistics<'a>,
        file_dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        if let Some(nested_field_aggregates) = fb.nested_field_aggregates() {
            #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
            let aggregate_ids: Arc<[_]> = fb
                .aggregate_specs()
                .iter()
                .flat_map(|specs| specs.iter())
                .map(|spec| AggregateFnId::new(spec.id()))
                .collect();
            let aggregate_ctx = ReadContext::new(aggregate_ids);
            let layout = postorder_stats_layout(file_dtype);
            vortex_ensure_eq!(nested_field_aggregates.len(), layout.len());

            let mut aggregates = Vec::with_capacity(layout.len());
            let mut dtypes = Vec::with_capacity(layout.len());
            let mut paths = Vec::with_capacity(layout.len());
            for (field_aggregates, (path, dtype)) in nested_field_aggregates.iter().zip(layout) {
                aggregates.push(aggregate_stats_from_flatbuffer(
                    &field_aggregates,
                    &aggregate_ctx,
                    &dtype,
                    session,
                )?);
                dtypes.push(dtype);
                paths.push(path);
            }

            return Ok(Self::from_parts(
                aggregates.into(),
                dtypes.into(),
                paths.into(),
                Arc::new([]),
            ));
        }

        // Legacy (pre-nested-stats) layout: top-level struct fields only, or a single entry for a
        // non-struct root dtype.
        let field_stats = fb.field_stats().unwrap_or_default();
        let mut array_stats: Vec<ArrayStats> = field_stats.iter().collect();

        if let DType::Struct(struct_fields, _) = file_dtype {
            vortex_ensure_eq!(array_stats.len(), struct_fields.nfields());

            let aggregates: Arc<[AggregateStats]> = array_stats
                .into_iter()
                .zip(struct_fields.fields())
                .map(|(array_stat, field_dtype)| {
                    aggregates_from_array_stats(&array_stat, &field_dtype, session)
                })
                .try_collect()?;

            let dtypes = struct_fields.fields().collect();
            let paths = struct_fields
                .names()
                .iter()
                .map(|name| FieldPath::from_name(name.clone()))
                .collect();

            Ok(Self::new(aggregates, dtypes, paths))
        } else {
            vortex_ensure_eq!(array_stats.len(), 1);

            let array_stat = array_stats
                .pop()
                .vortex_expect("we just checked that there was 1 field");
            let aggregates = aggregates_from_array_stats(&array_stat, file_dtype, session)?;

            Ok(Self::new(
                Arc::new([aggregates]),
                Arc::new([file_dtype.clone()]),
                Arc::new([FieldPath::root()]),
            ))
        }
    }

    /// Returns a reference to the aggregates recorded for each entry.
    pub fn aggregates(&self) -> &Arc<[AggregateStats]> {
        &self.aggregates
    }

    /// Returns `true` if there is no statistical information at all, in either the nested or the
    /// legacy layout.
    ///
    /// These can disagree: a non-nullable struct field whose entire subtree is unsupported dtypes
    /// (e.g. all-`Variant`) contributes no entries to the nested post-order layout (every leaf is
    /// skipped, and there's no nullable struct along the way to emit an own entry), but still gets
    /// a legacy top-level entry with a real `NullCount`, since that stat is dtype-agnostic.
    pub fn is_empty(&self) -> bool {
        self.aggregates.is_empty() && self.legacy_aggregates.is_empty()
    }

    /// Returns a reference to the data types.
    pub fn dtypes(&self) -> &Arc<[DType]> {
        &self.dtypes
    }

    /// Returns a reference to the field paths.
    pub fn paths(&self) -> &Arc<[FieldPath]> {
        &self.paths
    }

    /// Returns the aggregates and data type for a specific field.
    ///
    /// # Panics
    ///
    /// Panics if `field_idx` is out of bounds.
    pub fn get(&self, field_idx: usize) -> (&AggregateStats, &DType) {
        (&self.aggregates[field_idx], &self.dtypes[field_idx])
    }

    /// Returns the aggregates and data type for the field at the given path, if present.
    pub fn get_by_path(&self, path: &FieldPath) -> Option<(&AggregateStats, &DType)> {
        self.path_index
            .get(path)
            .map(|&idx| (&self.aggregates[idx], &self.dtypes[idx]))
    }
}

/// The aggregates a legacy [`ArrayStats`] over a field of `dtype` holds, for the stats that have an
/// aggregate function.
fn aggregates_from_array_stats(
    array_stats: &ArrayStats<'_>,
    dtype: &DType,
    session: &VortexSession,
) -> VortexResult<AggregateStats> {
    let stats_set = StatsSet::from_flatbuffer(array_stats, dtype, session)?;
    Ok(AggregateStats::new(
        stats_set
            .iter()
            .filter_map(|(stat, value)| {
                let aggregate_fn = stat.aggregate_fn()?;
                let stat_dtype = stat.dtype(dtype)?;
                let value = value
                    .clone()
                    .map(|value| Scalar::try_new(stat_dtype.clone(), Some(value)))
                    .transpose()
                    .ok()?;
                Some(AggregateStat::from_value(aggregate_fn, value))
            })
            .collect(),
    ))
}

/// Resolves an [`fb::AggregateState`]'s aggregate function, or `None` if the session doesn't
/// know it.
fn aggregate_fn_from_state(
    state: &fb::AggregateState<'_>,
    aggregate_ctx: &ReadContext,
    session: &VortexSession,
) -> VortexResult<Option<AggregateFnRef>> {
    let id = aggregate_ctx
        .resolve(state.aggregate_spec())
        .ok_or_else(|| {
            vortex_err!(
                "aggregate spec index {} out of range for {} specs",
                state.aggregate_spec(),
                aggregate_ctx.ids().len()
            )
        })?;
    let Some(plugin) = session.aggregate_fns().find_plugin(&id) else {
        return Ok(None);
    };
    let options = state
        .options()
        .map(|options| options.bytes())
        .unwrap_or_default();
    let aggregate_fn = plugin.deserialize(options, session)?;
    if aggregate_fn.id() != id {
        vortex_bail!(
            "Aggregate function ID mismatch: expected {}, got {}",
            id,
            aggregate_fn.id()
        );
    }
    Ok(Some(aggregate_fn))
}

fn aggregate_stats_from_flatbuffer(
    field_aggregates: &fb::FieldAggregates<'_>,
    aggregate_ctx: &ReadContext,
    dtype: &DType,
    session: &VortexSession,
) -> VortexResult<AggregateStats> {
    let mut aggregates = Vec::new();
    for state in field_aggregates.aggregates().unwrap_or_default() {
        let Some(aggregate_fn) = aggregate_fn_from_state(&state, aggregate_ctx, session)? else {
            continue;
        };
        let partial_dtype = aggregate_fn.state_dtype(dtype).ok_or_else(|| {
            vortex_err!("aggregate {aggregate_fn} does not support field dtype {dtype}")
        })?;
        let partial =
            ScalarValue::from_proto_bytes(state.partial().bytes(), &partial_dtype, session)?;
        let partial = Scalar::try_new(partial_dtype, partial)?;
        aggregates.push(AggregateStat::try_from_partial(
            aggregate_fn,
            partial,
            dtype,
        )?);
    }
    Ok(AggregateStats::new(aggregates))
}

impl<'a> IntoIterator for &'a FileStatistics {
    type Item = (&'a AggregateStats, &'a DType);
    type IntoIter =
        std::iter::Zip<std::slice::Iter<'a, AggregateStats>, std::slice::Iter<'a, DType>>;

    fn into_iter(self) -> Self::IntoIter {
        self.aggregates.iter().zip(self.dtypes.iter())
    }
}

impl FlatBufferRoot for FileStatistics {}

impl WriteFlatBuffer for FileStatistics {
    type Target<'a> = fb::FileStatistics<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        // The legacy `ArrayStats` format has a fixed slot per stat rather than open-ended
        // aggregates.
        let field_stats = self
            .legacy_aggregates
            .iter()
            .map(|aggregates| aggregates.to_stats_set().write_flatbuffer(fbb))
            .collect::<VortexResult<Vec<_>>>()?;
        let field_stats = fbb.create_vector(field_stats.as_slice());

        // Aggregate IDs are dictionary-encoded, like the footer's array and layout IDs.
        let interner = Interner::empty();
        let mut nested_field_aggregates = Vec::with_capacity(self.aggregates.len());
        for field_aggregates in self.aggregates.iter() {
            let mut states = Vec::new();
            // Values without a partial state (e.g. from legacy statistics) can't be written.
            for stat in field_aggregates.iter() {
                let Some(partial) = stat.partial() else {
                    continue;
                };
                let aggregate_fn = stat.aggregate_fn();
                let aggregate_spec = interner
                    .intern(&aggregate_fn.id())
                    .vortex_expect("an unrestricted interner interns every ID");
                let options = aggregate_fn.options().serialize()?.ok_or_else(|| {
                    vortex_err!(
                        "Aggregate function '{}' is not serializable",
                        aggregate_fn.id()
                    )
                })?;
                let options = fbb.create_vector(options.as_slice());
                let partial =
                    fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(partial.value()));
                states.push(fb::AggregateState::create(
                    fbb,
                    &fb::AggregateStateArgs {
                        aggregate_spec,
                        options: Some(options),
                        partial: Some(partial),
                    },
                ));
            }
            let aggregates = fbb.create_vector(states.as_slice());
            nested_field_aggregates.push(fb::FieldAggregates::create(
                fbb,
                &fb::FieldAggregatesArgs {
                    aggregates: Some(aggregates),
                },
            ));
        }
        let nested_field_aggregates = fbb.create_vector(nested_field_aggregates.as_slice());

        let aggregate_specs = interner
            .to_ids()
            .iter()
            .map(|id| {
                let id = fbb.create_string(id.as_ref());
                fb::AggregateSpec::create(fbb, &fb::AggregateSpecArgs { id: Some(id) })
            })
            .collect::<Vec<_>>();
        let aggregate_specs = fbb.create_vector(aggregate_specs.as_slice());

        Ok(fb::FileStatistics::create(
            fbb,
            &fb::FileStatisticsArgs {
                field_stats: Some(field_stats),
                aggregate_specs: Some(aggregate_specs),
                nested_field_aggregates: Some(nested_field_aggregates),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use flatbuffers::FlatBufferBuilder;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::null_count::NullCount;
    use vortex_array::array_session;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::stats::Precision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::flatbuffers::WriteFlatBufferExt;
    use vortex_array::scalar::ScalarValue;

    use super::*;

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    /// An aggregate over a field of `dtype` whose partial state is `value` itself, as for `Min`
    /// and `NullCount`.
    fn stat(aggregate_fn: AggregateFnRef, value: Scalar, dtype: &DType) -> AggregateStat {
        let partial_dtype = aggregate_fn
            .state_dtype(dtype)
            .vortex_expect("aggregate supports dtype");
        let partial = value.cast(&partial_dtype).vortex_expect("castable partial");
        AggregateStat::try_from_partial(aggregate_fn, partial, dtype).vortex_expect("valid partial")
    }

    fn min_fn() -> AggregateFnRef {
        Min.bind(NumericalAggregateOpts::skip_nans())
    }

    /// The value `aggregates` resolves for `aggregate_fn`.
    fn value(aggregates: &AggregateStats, aggregate_fn: &AggregateFnRef) -> Precision<ScalarValue> {
        aggregates.get(aggregate_fn).and_then(Scalar::into_value)
    }

    fn min(value: i32) -> AggregateStat {
        stat(min_fn(), Scalar::from(value), &i32_dtype())
    }

    fn null_count(value: u64, dtype: &DType) -> AggregateStat {
        stat(NullCount.bind(EmptyOptions), Scalar::from(value), dtype)
    }

    fn read_back(file_stats: &FileStatistics, file_dtype: &DType) -> VortexResult<FileStatistics> {
        let bytes = file_stats.write_flatbuffer_bytes()?;
        let fb = flatbuffers::root::<fb::FileStatistics>(bytes.as_ref())
            .vortex_expect("valid flatbuffer");
        FileStatistics::from_flatbuffer(&fb, file_dtype, &array_session())
    }

    #[test]
    fn nested_round_trip_resolves_by_path() -> VortexResult<()> {
        let inner = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let file_dtype = DType::struct_([("a", inner.clone())], Nullability::NonNullable);

        // Layout: [a.b, a] (a's own null-count entry trails its child).
        let file_stats = FileStatistics::new_with_dtype(
            Arc::from([
                AggregateStats::new(vec![min(1)]),
                AggregateStats::new(vec![null_count(1, &inner)]),
            ]),
            Arc::from([AggregateStats::default()]),
            &file_dtype,
        );

        let read_back = read_back(&file_stats, &file_dtype)?;

        let (b, _) = read_back
            .get_by_path(&FieldPath::from_name("a").push("b"))
            .expect("a.b stats");
        assert_eq!(
            value(b, &min_fn()),
            Precision::exact(ScalarValue::from(1i32))
        );

        let (a, _) = read_back
            .get_by_path(&FieldPath::from_name("a"))
            .expect("a's own null-count stats");
        assert_eq!(
            value(a, &NullCount.bind(EmptyOptions)),
            Precision::exact(ScalarValue::from(1u64))
        );

        assert!(read_back.get_by_path(&FieldPath::root()).is_none());

        Ok(())
    }

    #[test]
    fn aggregate_ids_are_written_once() -> VortexResult<()> {
        let file_dtype = DType::struct_(
            [("a", i32_dtype()), ("b", i32_dtype())],
            Nullability::NonNullable,
        );
        let file_stats = FileStatistics::new_with_dtype(
            Arc::from([
                AggregateStats::new(vec![min(1), null_count(0, &i32_dtype())]),
                AggregateStats::new(vec![min(2), null_count(0, &i32_dtype())]),
            ]),
            Arc::from([AggregateStats::default(), AggregateStats::default()]),
            &file_dtype,
        );

        let bytes = file_stats.write_flatbuffer_bytes()?;
        let fb = flatbuffers::root::<fb::FileStatistics>(bytes.as_ref())
            .vortex_expect("valid flatbuffer");
        let ids = fb
            .aggregate_specs()
            .vortex_expect("aggregate specs")
            .iter()
            .map(|spec| spec.id().to_string())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["vortex.min", "vortex.null_count"]);

        let read_back = read_back(&file_stats, &file_dtype)?;
        let (b, _) = read_back
            .get_by_path(&FieldPath::from_name("b"))
            .expect("b stats");
        assert_eq!(
            value(b, &min_fn()),
            Precision::exact(ScalarValue::from(2i32))
        );
        Ok(())
    }

    #[test]
    fn bounded_aggregate_round_trips_as_an_approximate_max() -> VortexResult<()> {
        let file_dtype = DType::struct_(
            [("s", DType::Utf8(Nullability::NonNullable))],
            Nullability::NonNullable,
        );
        let field_dtype = DType::Utf8(Nullability::NonNullable);
        let bounded_max = BoundedMax.bind(BoundedMaxOptions {
            max_bytes: NonZeroUsize::new(4).vortex_expect("non-zero"),
        });
        let mut accumulator = bounded_max.accumulator(&field_dtype)?;
        accumulator.accumulate(
            &VarBinViewArray::from_iter_str(["a value past the bound"]).into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        let partial = accumulator.partial_scalar()?;
        let file_stats = FileStatistics::new_with_dtype(
            Arc::from([AggregateStats::new(vec![AggregateStat::try_from_partial(
                bounded_max.clone(),
                partial,
                &field_dtype,
            )?])]),
            Arc::from([AggregateStats::default()]),
            &file_dtype,
        );

        let read_back = read_back(&file_stats, &file_dtype)?;
        let path = FieldPath::from_name("s");
        let (aggregates, _) = read_back.get_by_path(&path).expect("s aggregates");
        assert!(
            aggregates
                .iter()
                .all(|stat| stat.aggregate_fn() == &bounded_max)
        );
        assert_eq!(
            value(aggregates, &Max.bind(NumericalAggregateOpts::skip_nans())),
            Precision::inexact(ScalarValue::from("a vb"))
        );
        Ok(())
    }

    #[test]
    fn unknown_aggregates_are_skipped() -> VortexResult<()> {
        let file_dtype = DType::struct_([("col", i32_dtype())], Nullability::NonNullable);
        let known = min(7);
        let partial = ScalarValue::to_proto_bytes::<Vec<u8>>(
            known.partial().vortex_expect("has a partial").value(),
        );

        let options = known
            .aggregate_fn()
            .options()
            .serialize()?
            .vortex_expect("serializable options");

        // Spec 0 is an aggregate no session knows, spec 1 is the known `min`.
        let mut fbb = FlatBufferBuilder::new();
        let aggregate_specs = ["test.unknown", known.aggregate_fn().id().as_ref()].map(|id| {
            let id = fbb.create_string(id);
            fb::AggregateSpec::create(&mut fbb, &fb::AggregateSpecArgs { id: Some(id) })
        });
        let aggregate_specs = fbb.create_vector(&aggregate_specs);
        let states = [0u16, 1].map(|aggregate_spec| {
            let options = fbb.create_vector(options.as_slice());
            let partial = fbb.create_vector(partial.as_slice());
            fb::AggregateState::create(
                &mut fbb,
                &fb::AggregateStateArgs {
                    aggregate_spec,
                    options: Some(options),
                    partial: Some(partial),
                },
            )
        });
        let aggregates = fbb.create_vector(&states);
        let field_aggregates = fb::FieldAggregates::create(
            &mut fbb,
            &fb::FieldAggregatesArgs {
                aggregates: Some(aggregates),
            },
        );
        let nested_field_aggregates = fbb.create_vector(&[field_aggregates]);
        let root = fb::FileStatistics::create(
            &mut fbb,
            &fb::FileStatisticsArgs {
                field_stats: None,
                aggregate_specs: Some(aggregate_specs),
                nested_field_aggregates: Some(nested_field_aggregates),
            },
        );
        fbb.finish_minimal(root);
        let bytes = fbb.finished_data().to_vec();

        let fb = flatbuffers::root::<fb::FileStatistics>(&bytes).vortex_expect("valid flatbuffer");
        let read_back = FileStatistics::from_flatbuffer(&fb, &file_dtype, &array_session())?;
        let (aggregates, _) = read_back
            .get_by_path(&FieldPath::from_name("col"))
            .expect("col aggregates");
        assert_eq!(aggregates.iter().count(), 1);
        assert_eq!(
            value(aggregates, &min_fn()),
            Precision::exact(ScalarValue::from(7i32))
        );
        Ok(())
    }

    #[test]
    fn get_by_path_resolves_a_later_sibling_past_a_nullable_struct_field() {
        // Regression test: a nullable struct field ("s") contributes both an `s.b` entry and a
        // trailing entry for its own null count to the nested (post-order) layout, so that layout
        // has one more entry (3) than there are top-level fields (2: "s", "c"). Resolving the
        // later top-level field "c" by path must not be thrown off by "s"'s extra own-entry.
        let s_dtype = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let file_dtype = DType::struct_(
            [("s", s_dtype.clone()), ("c", i32_dtype())],
            Nullability::NonNullable,
        );

        // Post-order layout: [s.b, s, c].
        let file_stats = FileStatistics::new_with_dtype(
            Arc::from([
                AggregateStats::default(),
                AggregateStats::new(vec![null_count(0, &s_dtype)]),
                AggregateStats::new(vec![min(42)]),
            ]),
            Arc::from([AggregateStats::default(), AggregateStats::default()]),
            &file_dtype,
        );

        let (c, _) = file_stats
            .get_by_path(&FieldPath::from_name("c"))
            .expect("c stats");
        assert_eq!(
            value(c, &min_fn()),
            Precision::exact(ScalarValue::from(42i32))
        );
    }

    #[test]
    fn legacy_non_nested_footer_still_parses() -> VortexResult<()> {
        // Simulates a footer written before nested field stats existed:
        // `nested_field_aggregates` is absent, and `field_stats` holds one entry per top-level
        // struct field.
        let session = array_session();
        let file_dtype = DType::struct_([("col", i32_dtype())], Nullability::NonNullable);

        let mut stats = StatsSet::default();
        stats.set(Stat::Min, Precision::exact(ScalarValue::from(7i32)));

        let mut fbb = FlatBufferBuilder::new();
        let array_stats = stats.write_flatbuffer(&mut fbb)?;
        let field_stats = fbb.create_vector(&[array_stats]);
        let root = fb::FileStatistics::create(
            &mut fbb,
            &fb::FileStatisticsArgs {
                field_stats: Some(field_stats),
                aggregate_specs: None,
                nested_field_aggregates: None,
            },
        );
        fbb.finish_minimal(root);
        let bytes = fbb.finished_data().to_vec();

        let fb = flatbuffers::root::<fb::FileStatistics>(&bytes).vortex_expect("valid flatbuffer");
        assert!(fb.nested_field_aggregates().is_none());

        let read_back = FileStatistics::from_flatbuffer(&fb, &file_dtype, &session)?;
        let path = FieldPath::from_name("col");
        let (col, _) = read_back.get_by_path(&path).expect("col aggregates");
        assert_eq!(
            value(col, &min_fn()),
            Precision::exact(ScalarValue::from(7i32))
        );

        Ok(())
    }

    #[test]
    fn writer_still_emits_legacy_field_stats_matching_top_level_field_count() -> VortexResult<()> {
        // Regression test: an old reader (pre-nested-stats) only ever looks at `field_stats` and
        // requires its length to match the number of top-level struct fields. A nested/nullable
        // struct schema's post-order layout has a different length, so `field_stats` must keep
        // carrying the legacy top-level-only shape, not the nested one.
        let inner = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let file_dtype =
            DType::struct_([("a", inner), ("c", i32_dtype())], Nullability::NonNullable);
        let struct_fields = file_dtype
            .as_struct_fields_opt()
            .vortex_expect("file_dtype is a struct");

        // Nested (post-order) layout: [a.b, a, c] - 3 entries, differs from the 2 top-level fields.
        let nested: Arc<[AggregateStats]> = Arc::from([
            AggregateStats::default(),
            AggregateStats::default(),
            AggregateStats::default(),
        ]);
        // Legacy layout: one entry per top-level field ("a", "c").
        let legacy: Arc<[AggregateStats]> =
            Arc::from([AggregateStats::default(), AggregateStats::default()]);

        let file_stats = FileStatistics::new_with_dtype(nested, legacy, &file_dtype);
        let bytes = file_stats.write_flatbuffer_bytes()?;
        let fb = flatbuffers::root::<fb::FileStatistics>(bytes.as_ref())
            .vortex_expect("valid flatbuffer");

        let field_stats_len = fb.field_stats().map_or(0, |field_stats| field_stats.len());
        assert_eq!(field_stats_len, struct_fields.nfields());

        Ok(())
    }

    #[test]
    fn is_empty_considers_legacy_stats_too() {
        // Regression test: the nested (post-order) layout can be empty while the legacy layout
        // still carries real content, e.g. a non-nullable struct field whose entire subtree is
        // unsupported dtypes contributes no nested entries, but still gets a legacy NullCount
        // (dtype-agnostic). `is_empty` must not report "nothing to write" in that case, or the
        // caller (the footer serializer) would silently drop the legacy stats too.
        // A non-nullable struct field whose only child is an unsupported dtype (`Variant`)
        // contributes zero entries to the nested post-order layout: the leaf is skipped, and the
        // struct itself is non-nullable so it gets no own entry either.
        let file_dtype = DType::struct_(
            [("a", DType::Variant(Nullability::NonNullable))],
            Nullability::NonNullable,
        );
        assert_eq!(postorder_stats_layout(&file_dtype), Vec::new());
        let legacy = AggregateStats::new(vec![null_count(
            0,
            &DType::Variant(Nullability::NonNullable),
        )]);

        let file_stats =
            FileStatistics::new_with_dtype(Arc::from([]), Arc::from([legacy]), &file_dtype);

        assert!(!file_stats.is_empty());
    }
}
