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
use vortex_array::stats::StatsSet;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::array::ArrayStats;
use vortex_flatbuffers::footer as fb;
use vortex_layout::layouts::file_stats::postorder_stats_layout;
use vortex_session::VortexSession;

/// Contains statistical information about the data in a Vortex file.
///
/// This struct wraps an array of `StatsSet` objects, each containing statistics
/// for a field or column in the file. These statistics can be used for query
/// optimization and data exploration.
#[derive(Clone, Debug)]
pub struct FileStatistics {
    /// An array of statistics sets, one for each field or column in the file.
    stats: Arc<[StatsSet]>,
    /// An array of `DType`s, one for each field or column in the file.
    dtypes: Arc<[DType]>,
    /// An array of field paths, one for each field or column in the file. Parallel to `stats` and
    /// `dtypes`. For files written before nested field stats, every path has depth 1 (or is the
    /// root path, for a non-struct file dtype).
    paths: Arc<[FieldPath]>,
}

impl FileStatistics {
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

        Self {
            stats,
            dtypes,
            paths,
        }
    }

    /// Creates a new [`FileStatistics`] from the given statistics and file dtype.
    ///
    /// `stats` must follow the post-order nested-struct layout produced by
    /// [`postorder_stats_layout`] for `file_dtype`.
    ///
    /// # Panics
    ///
    /// Panics if the number of stats doesn't match the expected number based on the dtype.
    pub fn new_with_dtype(stats: Arc<[StatsSet]>, file_dtype: &DType) -> Self {
        let layout = postorder_stats_layout(file_dtype);
        assert_eq!(
            stats.len(),
            layout.len(),
            "stats length must match the post-order stats layout for the file dtype"
        );

        let (paths, dtypes): (Vec<FieldPath>, Vec<DType>) = layout.into_iter().unzip();

        Self {
            stats,
            dtypes: dtypes.into(),
            paths: paths.into(),
        }
    }

    /// Creates [`FileStatistics`] from a flatbuffers [`fb::FileStatistics<'a>`].
    pub fn from_flatbuffer<'a>(
        fb: &fb::FileStatistics<'a>,
        file_dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let field_stats = fb.field_stats().unwrap_or_default();
        let mut array_stats: Vec<ArrayStats> = field_stats.iter().collect();

        if fb.is_nested() {
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

            return Ok(Self {
                stats: stats_sets.into(),
                dtypes: dtypes.into(),
                paths: paths.into(),
            });
        }

        // Legacy (pre-nested-stats) layout: top-level struct fields only, or a single entry for a
        // non-struct root dtype.
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

            Ok(Self {
                stats: stats_sets,
                dtypes,
                paths,
            })
        } else {
            vortex_ensure_eq!(array_stats.len(), 1);

            let array_stat = array_stats
                .pop()
                .vortex_expect("we just checked that there was 1 field");
            let stats_set = StatsSet::from_flatbuffer(&array_stat, file_dtype, session)?;

            Ok(Self {
                stats: Arc::new([stats_set]),
                dtypes: Arc::new([file_dtype.clone()]),
                paths: Arc::new([FieldPath::root()]),
            })
        }
    }

    /// Returns a reference to the statistics sets.
    pub fn stats_sets(&self) -> &Arc<[StatsSet]> {
        &self.stats
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
        self.paths
            .iter()
            .position(|p| p == path)
            .map(|idx| (&self.stats[idx], &self.dtypes[idx]))
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
            .stats_sets()
            .iter()
            .map(|s| s.write_flatbuffer(fbb))
            .collect::<VortexResult<Vec<_>>>()?;
        let field_stats = fbb.create_vector(field_stats.as_slice());

        Ok(fb::FileStatistics::create(
            fbb,
            &fb::FileStatisticsArgs {
                field_stats: Some(field_stats),
                is_nested: true,
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
    use vortex_array::scalar::ScalarValue;
    use vortex_flatbuffers::WriteFlatBuffer;
    use vortex_flatbuffers::WriteFlatBufferExt;

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

        let file_stats = FileStatistics::new_with_dtype(Arc::from([b_stats, a_stats]), &file_dtype);

        let bytes = file_stats.write_flatbuffer_bytes()?;
        let fb = flatbuffers::root::<fb::FileStatistics>(bytes.as_ref())
            .vortex_expect("valid flatbuffer");
        assert!(fb.is_nested());

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
    fn legacy_non_nested_footer_still_parses() -> VortexResult<()> {
        // Simulates a footer written before nested field stats existed: `is_nested` is absent
        // (defaults to false), and `field_stats` holds one entry per top-level struct field.
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
                is_nested: false,
            },
        );
        fbb.finish_minimal(root);
        let bytes = fbb.finished_data().to_vec();

        let fb = flatbuffers::root::<fb::FileStatistics>(&bytes).vortex_expect("valid flatbuffer");
        assert!(!fb.is_nested());

        let read_back = FileStatistics::from_flatbuffer(&fb, &file_dtype, &session)?;
        let (col, _) = read_back
            .get_by_path(&FieldPath::from_name("col"))
            .expect("col stats");
        assert_eq!(col.get(Stat::Min).as_exact(), Some(ScalarValue::from(7i32)));

        Ok(())
    }
}
