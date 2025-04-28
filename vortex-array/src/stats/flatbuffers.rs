use enum_iterator::all;
use flatbuffers::{FlatBufferBuilder, Follow, WIPOffset};
use vortex_error::VortexError;
use vortex_flatbuffers::{ReadFlatBuffer, WriteFlatBuffer};
use vortex_scalar::ScalarValue;

use super::traits::{StatsProvider, StatsProviderExt};
use crate::stats::{Precision, Stat, StatsSet};

impl WriteFlatBuffer for StatsSet {
    type Target<'t> = crate::flatbuffers::ArrayStats<'t>;

    /// All statistics written must be exact
    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let min = self
            .get(Stat::Min)
            .and_then(Precision::as_exact)
            .map(|min| fbb.create_vector(&min.to_protobytes::<Vec<u8>>()));

        let max = self
            .get(Stat::Max)
            .and_then(Precision::as_exact)
            .map(|max| fbb.create_vector(&max.to_protobytes::<Vec<u8>>()));

        let sum = self
            .get(Stat::Sum)
            .and_then(Precision::as_exact)
            .map(|sum| fbb.create_vector(&sum.to_protobytes::<Vec<u8>>()));

        let stat_args = &crate::flatbuffers::ArrayStatsArgs {
            min,
            max,
            sum,
            is_sorted: self
                .get_as::<bool>(Stat::IsSorted)
                .and_then(Precision::as_exact),
            is_strict_sorted: self
                .get_as::<bool>(Stat::IsStrictSorted)
                .and_then(Precision::as_exact),
            is_constant: self
                .get_as::<bool>(Stat::IsConstant)
                .and_then(Precision::as_exact),
            null_count: self
                .get_as::<u64>(Stat::NullCount)
                .and_then(Precision::as_exact),
            uncompressed_size_in_bytes: self
                .get_as::<u64>(Stat::UncompressedSizeInBytes)
                .and_then(Precision::as_exact),
            nan_count: self
                .get_as::<u64>(Stat::NaNCount)
                .and_then(Precision::as_exact),
        };

        crate::flatbuffers::ArrayStats::create(fbb, stat_args)
    }
}

impl ReadFlatBuffer for StatsSet {
    type Source<'a> = crate::flatbuffers::ArrayStats<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error> {
        let mut stats_set = StatsSet::default();

        for stat in all::<Stat>() {
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
                    if let Some(max) = fb.max() {
                        stats_set.set(
                            Stat::Max,
                            Precision::Exact(ScalarValue::from_protobytes(max.bytes())?),
                        );
                    }
                }
                Stat::Min => {
                    if let Some(min) = fb.min() {
                        stats_set.set(
                            Stat::Min,
                            Precision::Exact(ScalarValue::from_protobytes(min.bytes())?),
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
                    if let Some(sum) = fb.sum() {
                        stats_set.set(
                            Stat::Sum,
                            Precision::Exact(ScalarValue::from_protobytes(sum.bytes())?),
                        );
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
