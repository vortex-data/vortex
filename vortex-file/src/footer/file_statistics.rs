// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Finalized aggregate summaries stored in the file footer.

use std::sync::Arc;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use itertools::Itertools;
use vortex_array::dtype::DType;
use vortex_array::flatbuffers::FlatBufferRoot;
use vortex_array::flatbuffers::WriteFlatBuffer;
use vortex_array::stats::AggregateResults;
use vortex_array::stats::compat::read_summary;
use vortex_array::stats::compat::write_summary;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::flatbuffers::footer as fb;

/// Finalized aggregate results for each top-level field, or for a single non-struct column.
///
/// Results preserve exact values, bounds, and missing metadata. They are not mergeable partial
/// states. Each field's lookup keys include the aggregate function and its options.
#[derive(Clone, Debug)]
pub struct FileStatistics {
    /// Finalized results, one collection per field or column.
    stats: Arc<[AggregateResults]>,
    /// An array of `DType`s, one for each field or column in the file.
    dtypes: Arc<[DType]>,
}

impl FileStatistics {
    /// Creates a new [`FileStatistics`] from the given statistics and data types.
    ///
    /// # Panics
    ///
    /// Panics if `stats` and `dtypes` have different lengths.
    pub fn new(stats: Arc<[AggregateResults]>, dtypes: Arc<[DType]>) -> Self {
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
    /// result collections as struct fields. Otherwise, there must be one collection.
    ///
    /// # Panics
    ///
    /// Panics if the number of stats doesn't match the expected number based on the dtype.
    pub fn new_with_dtype(stats: Arc<[AggregateResults]>, file_dtype: &DType) -> Self {
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
    /// field summaries in the flatbuffer. Otherwise, there must be one summary.
    pub fn from_flatbuffer<'a>(
        fb: &fb::FileStatistics<'a>,
        file_dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let field_stats = fb.field_stats().unwrap_or_default();

        if let DType::Struct(struct_fields, _) = file_dtype {
            vortex_ensure_eq!(field_stats.len(), struct_fields.nfields());

            let fields: Arc<[AggregateResults]> = field_stats
                .into_iter()
                .zip(struct_fields.fields())
                .map(|(array_stat, field_dtype)| read_summary(&array_stat, &field_dtype, session))
                .try_collect()?;

            let dtypes = struct_fields.fields().collect();

            Ok(Self {
                stats: fields,
                dtypes,
            })
        } else {
            vortex_ensure_eq!(field_stats.len(), 1);

            let array_stat = field_stats.get(0);
            let results = read_summary(&array_stat, file_dtype, session)?;

            Ok(Self {
                stats: Arc::new([results]),
                dtypes: Arc::new([file_dtype.clone()]),
            })
        }
    }

    /// Returns the finalized aggregate results for each field.
    pub fn fields(&self) -> &Arc<[AggregateResults]> {
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
    pub fn get(&self, field_idx: usize) -> (&AggregateResults, &DType) {
        (&self.stats[field_idx], &self.dtypes[field_idx])
    }
}

impl<'a> IntoIterator for &'a FileStatistics {
    type Item = (&'a AggregateResults, &'a DType);
    type IntoIter = std::iter::Zip<std::slice::Iter<'a, AggregateResults>, std::slice::Iter<'a, DType>>;

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
            .into_iter()
            .map(|(results, dtype)| write_summary(results, dtype, fbb))
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
