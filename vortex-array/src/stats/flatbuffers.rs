// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use half::f16;
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
use crate::scalar::PValue;
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
        let (min_precision, min) = match self.get(Stat::Min) {
            Precision::Exact(min) => (
                fba::Precision::Exact,
                Some(fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(&min)))),
            ),
            Precision::Inexact(min) => (
                fba::Precision::Inexact,
                Some(fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(&min)))),
            ),
            Precision::Absent => (fba::Precision::Inexact, None),
        };

        let (max_precision, max) = match self.get(Stat::Max) {
            Precision::Exact(max) => (
                fba::Precision::Exact,
                Some(fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(&max)))),
            ),
            Precision::Inexact(max) => (
                fba::Precision::Inexact,
                Some(fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(&max)))),
            ),
            Precision::Absent => (fba::Precision::Inexact, None),
        };

        let sum = self.get(Stat::Sum).as_exact().map(|sum| {
            let sum = canonicalize_nan_sum(sum);
            fbb.create_vector(&ScalarValue::to_proto_bytes::<Vec<u8>>(Some(&sum)))
        });

        let stat_args = &fba::ArrayStatsArgs {
            min,
            min_precision,
            max,
            max_precision,
            sum,
            is_sorted: self
                .get_as::<bool>(Stat::IsSorted, &DType::Bool(Nullability::NonNullable))
                .as_exact(),
            is_strict_sorted: self
                .get_as::<bool>(Stat::IsStrictSorted, &DType::Bool(Nullability::NonNullable))
                .as_exact(),
            is_constant: self
                .get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable))
                .as_exact(),
            null_count: self
                .get_as::<u64>(Stat::NullCount, &PType::U64.into())
                .as_exact(),
            uncompressed_size_in_bytes: self
                .get_as::<u64>(Stat::UncompressedSizeInBytes, &PType::U64.into())
                .as_exact(),
            nan_count: self
                .get_as::<u64>(Stat::NaNCount, &PType::U64.into())
                .as_exact(),
        };

        Ok(fba::ArrayStats::create(fbb, stat_args))
    }
}

/// Replace a NaN float sum with the canonical quiet NaN of its width.
///
/// A float sum is NaN when the data holds both `+inf` and `-inf`, because `inf + -inf` is an IEEE 754
/// invalid operation. IEEE 754 does not specify the bits of that result, and targets disagree:
/// x86_64 sets the sign bit and aarch64 does not. Without this step, the bytes of a written file
/// depend on the architecture that wrote it. A NaN sum has no payload information, so nothing is
/// lost. Stats that are already written are read back unchanged.
fn canonicalize_nan_sum(sum: ScalarValue) -> ScalarValue {
    match sum {
        ScalarValue::Primitive(PValue::F16(v)) if v.is_nan() => {
            ScalarValue::Primitive(PValue::F16(f16::from_bits(0x7e00)))
        }
        ScalarValue::Primitive(PValue::F32(v)) if v.is_nan() => {
            ScalarValue::Primitive(PValue::F32(f32::from_bits(0x7fc0_0000)))
        }
        ScalarValue::Primitive(PValue::F64(v)) if v.is_nan() => {
            ScalarValue::Primitive(PValue::F64(f64::from_bits(0x7ff8_0000_0000_0000)))
        }
        sum => sum,
    }
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

#[cfg(test)]
mod tests {
    use flatbuffers::FlatBufferBuilder;
    use flatbuffers::root;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::array_session;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::flatbuffers::WriteFlatBuffer;
    use crate::flatbuffers::array as fba;
    use crate::scalar::PValue;
    use crate::scalar::ScalarValue;
    use crate::stats::StatsSet;

    fn roundtrip_sum(sum: f64) -> VortexResult<u64> {
        let stats = StatsSet::of(
            Stat::Sum,
            Precision::exact(ScalarValue::Primitive(PValue::F64(sum))),
        );

        let mut fbb = FlatBufferBuilder::new();
        let offset = stats.write_flatbuffer(&mut fbb)?;
        fbb.finish_minimal(offset);
        let fb = root::<fba::ArrayStats>(fbb.finished_data())?;

        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let read = StatsSet::from_flatbuffer(&fb, &dtype, &array_session())?;
        Ok(read
            .get_as::<f64>(
                Stat::Sum,
                &DType::Primitive(PType::F64, Nullability::Nullable),
            )
            .as_exact()
            .ok_or_else(|| vortex_err!("sum must be exact after a round trip"))?
            .to_bits())
    }

    #[test]
    fn nan_sum_is_written_canonical() -> VortexResult<()> {
        // x86_64 gives this NaN for `inf + -inf`.
        assert_eq!(
            roundtrip_sum(f64::from_bits(0xfff8_0000_0000_0000))?,
            0x7ff8_0000_0000_0000
        );
        assert_eq!(
            roundtrip_sum(f64::from_bits(0x7ff8_0000_dead_beef))?,
            0x7ff8_0000_0000_0000
        );

        assert_eq!(roundtrip_sum(3.0)?, 3.0f64.to_bits());
        assert_eq!(
            roundtrip_sum(f64::NEG_INFINITY)?,
            f64::NEG_INFINITY.to_bits()
        );

        Ok(())
    }
}
