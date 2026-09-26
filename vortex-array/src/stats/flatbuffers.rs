// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::flatbuffers::WriteFlatBuffer;
use crate::flatbuffers::array as fba;
use crate::scalar::Scalar;
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
        // Read stored values in place instead of copying them out first
        write_stats(fbb, |stat| self.value(stat))
    }
}

impl WriteFlatBuffer for StatsSet {
    type Target<'t> = fba::ArrayStats<'t>;

    /// All statistics written must be exact
    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        write_stats(fbb, |stat| {
            self.iter()
                .find(|(stored, _)| *stored == stat)
                .map(|(_, value)| value)
        })
    }
}

/// Writes the stats that `get` returns for each [`Stat`].
fn write_stats<'fb, 'a>(
    fbb: &mut FlatBufferBuilder<'fb>,
    get: impl Fn(Stat) -> Option<&'a Precision<ScalarValue>>,
) -> VortexResult<WIPOffset<fba::ArrayStats<'fb>>> {
    let mut bound = |stat| match get(stat) {
        Some(Precision::Exact(value)) => (
            fba::Precision::Exact,
            Some(fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(value)))),
        ),
        Some(Precision::Inexact(value)) => (
            fba::Precision::Inexact,
            Some(fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(value)))),
        ),
        None | Some(Precision::Absent) => (fba::Precision::Inexact, None),
    };
    let (min_precision, min) = bound(Stat::Min);
    let (max_precision, max) = bound(Stat::Max);

    let sum = get(Stat::Sum)
        .and_then(|sum| sum.as_ref().as_exact())
        .map(|sum| fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(sum))));

    // Flags and counts are small, so converting them through a typed scalar costs nothing
    let exact_as = |stat, dtype: DType| {
        get(stat)
            .and_then(|value| value.as_ref().as_exact())
            .and_then(|value| Scalar::try_new(dtype, Some(value.clone())).ok())
    };
    let flag = |stat| {
        exact_as(stat, DType::Bool(Nullability::NonNullable))
            .and_then(|value| bool::try_from(&value).ok())
    };
    let count =
        |stat| exact_as(stat, PType::U64.into()).and_then(|value| u64::try_from(&value).ok());

    let stat_args = &fba::ArrayStatsArgs {
        min,
        min_precision,
        max,
        max_precision,
        sum,
        is_sorted: flag(Stat::IsSorted),
        is_strict_sorted: flag(Stat::IsStrictSorted),
        is_constant: flag(Stat::IsConstant),
        null_count: count(Stat::NullCount),
        uncompressed_size_in_bytes: count(Stat::UncompressedSizeInBytes),
        nan_count: count(Stat::NaNCount),
    };

    Ok(fba::ArrayStats::create(fbb, stat_args))
}

impl StatsSet {
    /// Creates a [`StatsSet`] from a flatbuffers array [`fba::ArrayStats<'a>`].
    pub fn from_flatbuffer<'a>(
        fb: &fba::ArrayStats<'a>,
        array_dtype: &DType,
        session: &VortexSession,
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
                        let value =
                            ScalarValue::from_proto_bytes(max.bytes(), &stat_dtype, session)?;
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
                        let value =
                            ScalarValue::from_proto_bytes(min.bytes(), &stat_dtype, session)?;
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
                        let value =
                            ScalarValue::from_proto_bytes(sum.bytes(), &stat_dtype, session)?;
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
