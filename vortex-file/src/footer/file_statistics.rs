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
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::flatbuffers::FlatBufferRoot;
use vortex_array::flatbuffers::WriteFlatBuffer;
use vortex_array::flatbuffers::array::ArrayStats;
use vortex_array::stats::StatsSet;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_layout::layouts::file_stats::postorder_stats_layout;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::flatbuffers::footer as fb;

/// Contains statistical information about the data in a Vortex file.
///
/// This struct wraps an array of `StatsSet` objects, each containing statistics
/// for a field or column in the file. These statistics can be used for query
/// optimization and data exploration.
#[derive(Clone, Debug)]
pub struct FileStatistics {
    /// An array of statistics sets, one for each field or column in the file, following the
    /// post-order nested-struct layout.
    stats: Arc<[StatsSet]>,
    /// An array of `DType`s, one for each field or column in the file.
    dtypes: Arc<[DType]>,
    /// An array of field paths, one for each field or column in the file. Parallel to `stats` and
    /// `dtypes`. For files written before nested field stats, every path has depth 1 (or is the
    /// root path, for a non-struct file dtype).
    paths: Arc<[FieldPath]>,
    /// Maps each entry in `paths` to its index, so [`Self::get_by_path`] doesn't need to scan
    /// `paths` linearly.
    path_index: Arc<HashMap<FieldPath, usize>>,
    /// Legacy top-level-fields-only statistics sets, one per top-level struct field (or a single
    /// entry for a non-struct root dtype). Only populated by the writer-facing constructors, for
    /// serialization into `field_stats`; empty for instances built from [`Self::from_flatbuffer`],
    /// which are never re-serialized.
    legacy_stats: Arc<[StatsSet]>,
}

impl FileStatistics {
    /// Builds `path_index` from `paths` and assembles the final struct. The single place that
    /// constructs a [`FileStatistics`], so `path_index` can't drift out of sync with `paths`.
    fn from_parts(
        stats: Arc<[StatsSet]>,
        dtypes: Arc<[DType]>,
        paths: Arc<[FieldPath]>,
        legacy_stats: Arc<[StatsSet]>,
    ) -> Self {
        let path_index = paths
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, p)| (p, i))
            .collect();
        Self {
            stats,
            dtypes,
            paths,
            path_index: Arc::new(path_index),
            legacy_stats,
        }
    }

    /// Creates a new [`FileStatistics`] from the given statistics, data types, and field paths.
    ///
    /// # Panics
    ///
    /// Panics if `stats`, `dtypes`, and `paths` have different lengths.
    pub fn new(stats: Arc<[StatsSet]>, dtypes: Arc<[DType]>, paths: Arc<[FieldPath]>) -> Self {
        assert_eq!(
            stats.len(),
            dtypes.len(),
            "stats and dtypes must have the same length"
        );
        assert_eq!(
            stats.len(),
            paths.len(),
            "stats and paths must have the same length"
        );

        Self::from_parts(stats, dtypes, paths, Arc::new([]))
    }

    /// Creates a new [`FileStatistics`] from the given nested statistics, legacy top-level-only
    /// statistics, and file dtype.
    ///
    /// `stats` must follow the post-order nested-struct layout produced by
    /// [`postorder_stats_layout`] for `file_dtype`. `legacy_stats` must have one entry per
    /// top-level struct field (or a single entry for a non-struct root dtype).
    ///
    /// # Panics
    ///
    /// Panics if the number of stats doesn't match the expected number based on the dtype.
    pub fn new_with_dtype(
        stats: Arc<[StatsSet]>,
        legacy_stats: Arc<[StatsSet]>,
        file_dtype: &DType,
    ) -> Self {
        let layout = postorder_stats_layout(file_dtype);
        assert_eq!(
            stats.len(),
            layout.len(),
            "stats length must match the post-order stats layout for the file dtype"
        );

        let (paths, dtypes): (Vec<FieldPath>, Vec<DType>) = layout.into_iter().unzip();

        Self::from_parts(stats, dtypes.into(), paths.into(), legacy_stats)
    }

    /// Creates [`FileStatistics`] from a flatbuffers [`fb::FileStatistics<'a>`].
    pub fn from_flatbuffer<'a>(
        fb: &fb::FileStatistics<'a>,
        file_dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        if let Some(nested_field_stats) = fb.nested_field_stats() {
            let array_stats: Vec<ArrayStats> = nested_field_stats.iter().collect();
            let layout = postorder_stats_layout(file_dtype);
            vortex_ensure_eq!(array_stats.len(), layout.len());

            let mut stats_sets = Vec::with_capacity(array_stats.len());
            let mut dtypes = Vec::with_capacity(layout.len());
            let mut paths = Vec::with_capacity(layout.len());
            for (array_stat, (path, dtype)) in array_stats.into_iter().zip(layout) {
                stats_sets.push(StatsSet::from_flatbuffer(&array_stat, &dtype, session)?);
                dtypes.push(dtype);
                paths.push(path);
            }

            return Ok(Self::from_parts(
                stats_sets.into(),
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

            let stats_sets: Arc<[StatsSet]> = array_stats
                .into_iter()
                .zip(struct_fields.fields())
                .map(|(array_stat, field_dtype)| {
                    StatsSet::from_flatbuffer(&array_stat, &field_dtype, session)
                })
                .try_collect()?;

            let dtypes = struct_fields.fields().collect();
            let paths = struct_fields
                .names()
                .iter()
                .map(|name| FieldPath::from_name(name.clone()))
                .collect();

            Ok(Self::from_parts(stats_sets, dtypes, paths, Arc::new([])))
        } else {
            vortex_ensure_eq!(array_stats.len(), 1);

            let array_stat = array_stats
                .pop()
                .vortex_expect("we just checked that there was 1 field");
            let stats_set = StatsSet::from_flatbuffer(&array_stat, file_dtype, session)?;

            Ok(Self::from_parts(
                Arc::new([stats_set]),
                Arc::new([file_dtype.clone()]),
                Arc::new([FieldPath::root()]),
                Arc::new([]),
            ))
        }
    }

    /// Returns a reference to the statistics sets.
    pub fn stats_sets(&self) -> &Arc<[StatsSet]> {
        &self.stats
    }

    /// Returns `true` if there is no statistical information at all, in either the nested or the
    /// legacy layout.
    ///
    /// These can disagree: a non-nullable struct field whose entire subtree is unsupported dtypes
    /// (e.g. all-`Variant`) contributes no entries to the nested post-order layout (every leaf is
    /// skipped, and there's no nullable struct along the way to emit an own entry), but still gets
    /// a legacy top-level entry with a real `NullCount`, since that stat is dtype-agnostic.
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty() && self.legacy_stats.is_empty()
    }

    /// Returns a reference to the data types.
    pub fn dtypes(&self) -> &Arc<[DType]> {
        &self.dtypes
    }

    /// Returns a reference to the field paths.
    pub fn paths(&self) -> &Arc<[FieldPath]> {
        &self.paths
    }

    /// Returns the statistics and data type for a specific field.
    ///
    /// # Panics
    ///
    /// Panics if `field_idx` is out of bounds.
    pub fn get(&self, field_idx: usize) -> (&StatsSet, &DType) {
        (&self.stats[field_idx], &self.dtypes[field_idx])
    }

    /// Returns the statistics and data type for the field at the given path, if present.
    pub fn get_by_path(&self, path: &FieldPath) -> Option<(&StatsSet, &DType)> {
        self.path_index
            .get(path)
            .map(|&idx| (&self.stats[idx], &self.dtypes[idx]))
    }
}

impl<'a> IntoIterator for &'a FileStatistics {
    type Item = (&'a StatsSet, &'a DType);
    type IntoIter = std::iter::Zip<std::slice::Iter<'a, StatsSet>, std::slice::Iter<'a, DType>>;

    fn into_iter(self) -> Self::IntoIter {
        self.stats.iter().zip(self.dtypes.iter())
    }
}

impl FlatBufferRoot for FileStatistics {}

impl WriteFlatBuffer for FileStatistics {
    type Target<'a> = fb::FileStatistics<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        let field_stats = self
            .legacy_stats
            .iter()
            .map(|s| s.write_flatbuffer(fbb))
            .collect::<VortexResult<Vec<_>>>()?;
        let field_stats = fbb.create_vector(field_stats.as_slice());

        let nested_field_stats = self
            .stats_sets()
            .iter()
            .map(|s| s.write_flatbuffer(fbb))
            .collect::<VortexResult<Vec<_>>>()?;
        let nested_field_stats = fbb.create_vector(nested_field_stats.as_slice());

        Ok(fb::FileStatistics::create(
            fbb,
            &fb::FileStatisticsArgs {
                field_stats: Some(field_stats),
                nested_field_stats: Some(nested_field_stats),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use flatbuffers::FlatBufferBuilder;
    use vortex_array::array_session;
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

    #[test]
    fn nested_round_trip_resolves_by_path() -> VortexResult<()> {
        let session = array_session();
        let inner = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let file_dtype = DType::struct_([("a", inner)], Nullability::NonNullable);

        // Layout: [a.b, a] (a's own null-count entry trails its child).
        let mut b_stats = StatsSet::default();
        b_stats.set(Stat::Min, Precision::exact(ScalarValue::from(1i32)));
        let mut a_stats = StatsSet::default();
        a_stats.set(Stat::NullCount, Precision::exact(ScalarValue::from(1u64)));

        // Legacy layout: one entry for top-level field "a" (a struct, so only NullCount survives).
        let mut legacy_a_stats = StatsSet::default();
        legacy_a_stats.set(Stat::NullCount, Precision::exact(ScalarValue::from(1u64)));

        let file_stats = FileStatistics::new_with_dtype(
            Arc::from([b_stats, a_stats]),
            Arc::from([legacy_a_stats]),
            &file_dtype,
        );

        let bytes = file_stats.write_flatbuffer_bytes()?;
        let fb = flatbuffers::root::<fb::FileStatistics>(bytes.as_ref())
            .vortex_expect("valid flatbuffer");
        assert!(fb.nested_field_stats().is_some());

        let read_back = FileStatistics::from_flatbuffer(&fb, &file_dtype, &session)?;

        let (b, _) = read_back
            .get_by_path(&FieldPath::from_name("a").push("b"))
            .expect("a.b stats");
        assert_eq!(b.get(Stat::Min).as_exact(), Some(ScalarValue::from(1i32)));

        let (a, _) = read_back
            .get_by_path(&FieldPath::from_name("a"))
            .expect("a's own null-count stats");
        assert_eq!(
            a.get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(1u64))
        );

        assert!(read_back.get_by_path(&FieldPath::root()).is_none());

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
            [("s", s_dtype), ("c", i32_dtype())],
            Nullability::NonNullable,
        );

        // Post-order layout: [s.b, s, c].
        let mut c_stats = StatsSet::default();
        c_stats.set(Stat::Min, Precision::exact(ScalarValue::from(42i32)));
        let mut s_own_stats = StatsSet::default();
        s_own_stats.set(Stat::NullCount, Precision::exact(ScalarValue::from(0u64)));

        let file_stats = FileStatistics::new_with_dtype(
            Arc::from([StatsSet::default(), s_own_stats, c_stats]),
            Arc::from([StatsSet::default(), StatsSet::default()]),
            &file_dtype,
        );

        let (c, _) = file_stats
            .get_by_path(&FieldPath::from_name("c"))
            .expect("c stats");
        assert_eq!(c.get(Stat::Min).as_exact(), Some(ScalarValue::from(42i32)));
    }

    #[test]
    fn legacy_non_nested_footer_still_parses() -> VortexResult<()> {
        // Simulates a footer written before nested field stats existed: `nested_field_stats` is
        // absent, and `field_stats` holds one entry per top-level struct field.
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
                nested_field_stats: None,
            },
        );
        fbb.finish_minimal(root);
        let bytes = fbb.finished_data().to_vec();

        let fb = flatbuffers::root::<fb::FileStatistics>(&bytes).vortex_expect("valid flatbuffer");
        assert!(fb.nested_field_stats().is_none());

        let read_back = FileStatistics::from_flatbuffer(&fb, &file_dtype, &session)?;
        let (col, _) = read_back
            .get_by_path(&FieldPath::from_name("col"))
            .expect("col stats");
        assert_eq!(col.get(Stat::Min).as_exact(), Some(ScalarValue::from(7i32)));

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
        let nested_stats: Arc<[StatsSet]> = Arc::from([
            StatsSet::default(),
            StatsSet::default(),
            StatsSet::default(),
        ]);
        // Legacy layout: one entry per top-level field ("a", "c").
        let legacy_stats: Arc<[StatsSet]> = Arc::from([StatsSet::default(), StatsSet::default()]);

        let file_stats = FileStatistics::new_with_dtype(nested_stats, legacy_stats, &file_dtype);
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
        let mut legacy = StatsSet::default();
        legacy.set(Stat::NullCount, Precision::exact(ScalarValue::from(0u64)));

        // A non-nullable struct field whose only child is an unsupported dtype (`Variant`)
        // contributes zero entries to the nested post-order layout: the leaf is skipped, and the
        // struct itself is non-nullable so it gets no own entry either.
        let file_dtype = DType::struct_(
            [("a", DType::Variant(Nullability::NonNullable))],
            Nullability::NonNullable,
        );
        assert_eq!(postorder_stats_layout(&file_dtype), Vec::new());

        let file_stats =
            FileStatistics::new_with_dtype(Arc::from([]), Arc::from([legacy]), &file_dtype);

        assert!(!file_stats.is_empty());
    }
}
