// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::array as fba;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::scalar::ScalarValue;
use crate::stats::StatsSet;
use crate::stats::StatsSetRef;

impl WriteFlatBuffer for StatsSetRef<'_> {
    type Target<'t> = fba::ArrayStats<'t>;

    /// All statistics written must be exact
    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        self.with_typed_stats_set(|stats_set| stats_set.values.write_flatbuffer(fbb))
    }
}

impl WriteFlatBuffer for StatsSet {
    type Target<'t> = fba::ArrayStats<'t>;

    /// All statistics written must be exact
    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        let (min_precision, min) = self
            .get(Stat::Min)
            .map(|min| {
                (
                    if min.is_exact() {
                        fba::Precision::Exact
                    } else {
                        fba::Precision::Inexact
                    },
                    Some(
                        fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
                            &min.into_inner(),
                        ))),
                    ),
                )
            })
            .unwrap_or_else(|| (fba::Precision::Inexact, None));

        let (max_precision, max) = self
            .get(Stat::Max)
            .map(|max| {
                (
                    if max.is_exact() {
                        fba::Precision::Exact
                    } else {
                        fba::Precision::Inexact
                    },
                    Some(
                        fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(
                            &max.into_inner(),
                        ))),
                    ),
                )
            })
            .unwrap_or_else(|| (fba::Precision::Inexact, None));

        let sum = self
            .get(Stat::Sum)
            .and_then(Precision::as_exact)
            .map(|sum| fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(&sum))));

        let stat_args = &fba::ArrayStatsArgs {
            min,
            min_precision,
            max,
            max_precision,
            sum,
            is_sorted: self
                .get_as::<bool>(Stat::IsSorted, &DType::Bool(Nullability::NonNullable))
                .and_then(Precision::as_exact),
            is_strict_sorted: self
                .get_as::<bool>(Stat::IsStrictSorted, &DType::Bool(Nullability::NonNullable))
                .and_then(Precision::as_exact),
            is_constant: self
                .get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable))
                .and_then(Precision::as_exact),
            null_count: self
                .get_as::<u64>(Stat::NullCount, &PType::U64.into())
                .and_then(Precision::as_exact),
            uncompressed_size_in_bytes: self
                .get_as::<u64>(Stat::UncompressedSizeInBytes, &PType::U64.into())
                .and_then(Precision::as_exact),
            nan_count: self
                .get_as::<u64>(Stat::NaNCount, &PType::U64.into())
                .and_then(Precision::as_exact),
        };

        Ok(fba::ArrayStats::create(fbb, stat_args))
    }
}

impl StatsSet {
    /// Creates a [`StatsSet`] from a flatbuffers array [`fba::ArrayStats<'a>`].
    pub fn from_flatbuffer<'a>(
        fb: &fba::ArrayStats<'a>,
        array_dtype: &DType,
    ) -> VortexResult<Self> {
        let mut stats_set = StatsSet::default();

        for stat in Stat::all() {
            let stat_dtype = stat.dtype(array_dtype);

            match stat {
                Stat::IsConstant => {
                    if let Some(is_constant) = fb.is_constant() {
                        stats_set.set(Stat::IsConstant, Precision::Exact(is_constant.into()));
                    }
                }
                Stat::IsSorted => {
                    if let Some(is_sorted) = fb.is_sorted() {
                        stats_set.set(Stat::IsSorted, Precision::Exact(is_sorted.into()));
                    }
                }
                Stat::IsStrictSorted => {
                    if let Some(is_strict_sorted) = fb.is_strict_sorted() {
                        stats_set.set(
                            Stat::IsStrictSorted,
                            Precision::Exact(is_strict_sorted.into()),
                        );
                    }
                }
                Stat::Max => {
                    if let Some(max) = fb.max()
                        && let Some(stat_dtype) = stat_dtype
                    {
                        let value = ScalarValue::from_proto_bytes(max.bytes(), &stat_dtype)?;
                        let Some(value) = value else {
                            continue;
                        };

                        stats_set.set(
                            Stat::Max,
                            match fb.max_precision() {
                                fba::Precision::Exact => Precision::Exact(value),
                                fba::Precision::Inexact => Precision::Inexact(value),
                                other => vortex_bail!("Corrupted max_precision field: {other:?}"),
                            },
                        );
                    }
                }
                Stat::Min => {
                    if let Some(min) = fb.min()
                        && let Some(stat_dtype) = stat_dtype
                    {
                        let value = ScalarValue::from_proto_bytes(min.bytes(), &stat_dtype)?;
                        let Some(value) = value else {
                            continue;
                        };

                        stats_set.set(
                            Stat::Min,
                            match fb.min_precision() {
                                fba::Precision::Exact => Precision::Exact(value),
                                fba::Precision::Inexact => Precision::Inexact(value),
                                other => vortex_bail!("Corrupted min_precision field: {other:?}"),
                            },
                        );
                    }
                }
                Stat::NullCount => {
                    if let Some(null_count) = fb.null_count() {
                        stats_set.set(Stat::NullCount, Precision::Exact(null_count.into()));
                    }
                }
                Stat::UncompressedSizeInBytes => {
                    if let Some(uncompressed_size_in_bytes) = fb.uncompressed_size_in_bytes() {
                        stats_set.set(
                            Stat::UncompressedSizeInBytes,
                            Precision::Exact(uncompressed_size_in_bytes.into()),
                        );
                    }
                }
                Stat::Sum => {
                    if let Some(sum) = fb.sum()
                        && let Some(stat_dtype) = stat_dtype
                    {
                        let value = ScalarValue::from_proto_bytes(sum.bytes(), &stat_dtype)?;
                        let Some(value) = value else {
                            continue;
                        };

                        stats_set.set(Stat::Sum, Precision::Exact(value));
                    }
                }
                Stat::NaNCount => {
                    if let Some(nan_count) = fb.nan_count() {
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
}
