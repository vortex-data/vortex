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
use vortex_array::stats::StatsSet;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::array::ArrayStats;
use vortex_flatbuffers::footer as fb;
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
}

impl FileStatistics {
    /// Creates a new [`FileStatistics`] from the given statistics and data types.
    ///
    /// # Panics
    ///
    /// Panics if `stats` and `dtypes` have different lengths.
    pub fn new(stats: Arc<[StatsSet]>, dtypes: Arc<[DType]>) -> Self {
        assert_eq!(
            stats.len(),
            dtypes.len(),
            "stats and dtypes must have the same length"
        );

        Self { stats, dtypes }
    }

    /// Creates a new [`FileStatistics`] from the given statistics and file dtype.
    ///
    /// If the [`DType`] of the file is a [`DType::Struct`], then there must be the same number of
    /// stats as struct fields. Otherwise, there must be only 1 statistic.
    ///
    /// # Panics
    ///
    /// Panics if the number of stats doesn't match the expected number based on the dtype.
    pub fn new_with_dtype(stats: Arc<[StatsSet]>, file_dtype: &DType) -> Self {
        if let DType::Struct(struct_fields, _) = file_dtype {
            assert_eq!(
                stats.len(),
                struct_fields.nfields(),
                "stats length must match number of struct fields"
            );

            let dtypes = struct_fields.fields().collect();

            Self { stats, dtypes }
        } else {
            assert_eq!(
                stats.len(),
                1,
                "non-struct dtype must have exactly 1 statistic"
            );

            Self {
                stats,
                dtypes: Arc::new([file_dtype.clone()]),
            }
        }
    }

    /// Creates [`FileStatistics`] from a flatbuffers [`fb::FileStatistics<'a>`].
    ///
    /// If the [`DType`] of the file is a [`DType::Struct`], then there must be the same number of
    /// file stats in the flatbuffer. Otherwise, there must be only 1 statistic.
    pub fn from_flatbuffer<'a>(
        fb: &fb::FileStatistics<'a>,
        file_dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
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

            Ok(Self {
                stats: stats_sets,
                dtypes,
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

    /// Returns the statistics and data type for a specific field.
    ///
    /// # Panics
    ///
    /// Panics if `field_idx` is out of bounds.
    pub fn get(&self, field_idx: usize) -> (&StatsSet, &DType) {
        (&self.stats[field_idx], &self.dtypes[field_idx])
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
            },
        ))
    }
}
