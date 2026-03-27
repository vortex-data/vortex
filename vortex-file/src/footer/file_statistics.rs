// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the file statistics component of the Vortex file footer.
//!
//! File statistics provide metadata about the data in the file, such as min/max values,
//! null counts, and other statistical information that can be used for query optimization
//! and data exploration.
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::ScalarValue;
use vortex_array::stats::StatsSet;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_flatbuffers::FlatBufferBuilder;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WIPOffset;
use vortex_flatbuffers::WriteFlatBuffer;
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
    pub fn from_flatbuffer(
        fb: &fb::FileStatisticsRef<'_>,
        file_dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let mut array_stats: Vec<fb::ArrayStatsRef<'_>> = fb
            .field_stats()?
            .map(|stats| stats.iter().collect::<Result<Vec<_>, _>>())
            .transpose()?
            .unwrap_or_default();

        if let DType::Struct(struct_fields, _) = file_dtype {
            vortex_ensure_eq!(array_stats.len(), struct_fields.nfields());

            let stats_sets: Arc<[StatsSet]> = array_stats
                .into_iter()
                .zip(struct_fields.fields())
                .map(|(array_stat, field_dtype)| {
                    parse_field_stats(&array_stat, &field_dtype, session)
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
            let stats_set = parse_field_stats(&array_stat, file_dtype, session)?;

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
    type Target = fb::FileStatistics;

    fn write_flatbuffer(
        &self,
        fbb: &mut FlatBufferBuilder,
    ) -> VortexResult<WIPOffset<Self::Target>> {
        let field_stats = self
            .stats_sets()
            .iter()
            .map(write_field_stats)
            .collect::<VortexResult<Vec<_>>>()?;
        let field_stats = fbb.create_vector(field_stats.as_slice());

        Ok(fb::FileStatistics::create(fbb, Some(field_stats)))
    }
}

fn write_field_stats(stats: &StatsSet) -> VortexResult<fb::ArrayStats> {
    let (min_precision, min) = stats
        .get(Stat::Min)
        .map(|min| {
            (
                if min.is_exact() {
                    fb::Precision::Exact
                } else {
                    fb::Precision::Inexact
                },
                Some(ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
                    &min.into_inner(),
                ))),
            )
        })
        .unwrap_or((fb::Precision::Inexact, None));

    let (max_precision, max) = stats
        .get(Stat::Max)
        .map(|max| {
            (
                if max.is_exact() {
                    fb::Precision::Exact
                } else {
                    fb::Precision::Inexact
                },
                Some(ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
                    &max.into_inner(),
                ))),
            )
        })
        .unwrap_or((fb::Precision::Inexact, None));

    let sum = stats
        .get(Stat::Sum)
        .and_then(Precision::as_exact)
        .map(|sum| ScalarValue::to_proto_bytes::<Vec<u8>>(Some(&sum)));

    Ok(fb::ArrayStats {
        min,
        min_precision,
        max,
        max_precision,
        sum,
        is_sorted: stats
            .get_as::<bool>(Stat::IsSorted, &DType::Bool(Nullability::NonNullable))
            .and_then(Precision::as_exact),
        is_strict_sorted: stats
            .get_as::<bool>(Stat::IsStrictSorted, &DType::Bool(Nullability::NonNullable))
            .and_then(Precision::as_exact),
        is_constant: stats
            .get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable))
            .and_then(Precision::as_exact),
        null_count: stats
            .get_as::<u64>(Stat::NullCount, &PType::U64.into())
            .and_then(Precision::as_exact),
        uncompressed_size_in_bytes: stats
            .get_as::<u64>(Stat::UncompressedSizeInBytes, &PType::U64.into())
            .and_then(Precision::as_exact),
        nan_count: stats
            .get_as::<u64>(Stat::NaNCount, &PType::U64.into())
            .and_then(Precision::as_exact),
    })
}

fn parse_field_stats(
    fb: &fb::ArrayStatsRef<'_>,
    array_dtype: &DType,
    session: &VortexSession,
) -> VortexResult<StatsSet> {
    let mut stats_set = StatsSet::default();

    for stat in Stat::all() {
        let stat_dtype = stat.dtype(array_dtype);

        match stat {
            Stat::IsConstant => {
                if let Some(is_constant) = fb.is_constant()? {
                    stats_set.set(Stat::IsConstant, Precision::Exact(is_constant.into()));
                }
            }
            Stat::IsSorted => {
                if let Some(is_sorted) = fb.is_sorted()? {
                    stats_set.set(Stat::IsSorted, Precision::Exact(is_sorted.into()));
                }
            }
            Stat::IsStrictSorted => {
                if let Some(is_strict_sorted) = fb.is_strict_sorted()? {
                    stats_set.set(
                        Stat::IsStrictSorted,
                        Precision::Exact(is_strict_sorted.into()),
                    );
                }
            }
            Stat::Max => {
                if let Some(max) = fb.max()?
                    && let Some(stat_dtype) = stat_dtype
                {
                    let value = ScalarValue::from_proto_bytes(max, &stat_dtype, session)?;
                    let Some(value) = value else {
                        continue;
                    };

                    stats_set.set(
                        Stat::Max,
                        match fb.max_precision()? {
                            fb::Precision::Exact => Precision::Exact(value),
                            fb::Precision::Inexact => Precision::Inexact(value),
                        },
                    );
                }
            }
            Stat::Min => {
                if let Some(min) = fb.min()?
                    && let Some(stat_dtype) = stat_dtype
                {
                    let value = ScalarValue::from_proto_bytes(min, &stat_dtype, session)?;
                    let Some(value) = value else {
                        continue;
                    };

                    stats_set.set(
                        Stat::Min,
                        match fb.min_precision()? {
                            fb::Precision::Exact => Precision::Exact(value),
                            fb::Precision::Inexact => Precision::Inexact(value),
                        },
                    );
                }
            }
            Stat::NullCount => {
                if let Some(null_count) = fb.null_count()? {
                    stats_set.set(Stat::NullCount, Precision::Exact(null_count.into()));
                }
            }
            Stat::UncompressedSizeInBytes => {
                if let Some(uncompressed_size_in_bytes) = fb.uncompressed_size_in_bytes()? {
                    stats_set.set(
                        Stat::UncompressedSizeInBytes,
                        Precision::Exact(uncompressed_size_in_bytes.into()),
                    );
                }
            }
            Stat::Sum => {
                if let Some(sum) = fb.sum()?
                    && let Some(stat_dtype) = stat_dtype
                {
                    let value = ScalarValue::from_proto_bytes(sum, &stat_dtype, session)?;
                    let Some(value) = value else {
                        continue;
                    };

                    stats_set.set(Stat::Sum, Precision::Exact(value));
                }
            }
            Stat::NaNCount => {
                if let Some(nan_count) = fb.nan_count()? {
                    stats_set.set(
                        Stat::NaNCount,
                        Precision::Exact(ScalarValue::from(nan_count)),
                    );
                }
            }
        }
    }

    Ok(stats_set)
}
