use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_flatbuffers::WriteFlatBuffer;

use crate::stats::{DirectionalBound, Precision, Stat, Statistics};

impl WriteFlatBuffer for &dyn Statistics {
    type Target<'t> = crate::flatbuffers::ArrayStats<'t>;

    /// All statistics written must be exact
    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let trailing_zero_freq = self
            .get_as::<Vec<u64>>(Stat::TrailingZeroFreq)
            .map(|v| {
                v.into_value()
                    .ok_exact()
                    .iter()
                    .flatten()
                    .copied()
                    .collect_vec()
            })
            .map(|v| fbb.create_vector(v.as_slice()));

        let bit_width_freq = self
            .get_as::<Vec<u64>>(Stat::BitWidthFreq)
            .map(|v| {
                v.into_value()
                    .ok_exact()
                    .iter()
                    .flatten()
                    .copied()
                    .collect_vec()
            })
            .map(|v| fbb.create_vector(v.as_slice()));

        let min = self
            .get(Stat::Min)
            .map(DirectionalBound::into_value)
            .and_then(Precision::ok_exact)
            .map(|min| min.write_flatbuffer(fbb));

        let max = self
            .get(Stat::Max)
            .map(DirectionalBound::into_value)
            .and_then(Precision::ok_exact)
            .map(|max| max.write_flatbuffer(fbb));

        let stat_args = &crate::flatbuffers::ArrayStatsArgs {
            min,
            max,
            is_sorted: self
                .get_as::<bool>(Stat::IsSorted)
                .map(DirectionalBound::into_value)
                .and_then(Precision::ok_exact),
            is_strict_sorted: self
                .get_as::<bool>(Stat::IsStrictSorted)
                .map(DirectionalBound::into_value)
                .and_then(Precision::ok_exact),
            is_constant: self
                .get_as::<bool>(Stat::IsConstant)
                .map(DirectionalBound::into_value)
                .and_then(Precision::ok_exact),
            run_count: self
                .get_as::<u64>(Stat::RunCount)
                .map(DirectionalBound::into_value)
                .and_then(Precision::ok_exact),
            true_count: self
                .get_as::<u64>(Stat::TrueCount)
                .map(DirectionalBound::into_value)
                .and_then(Precision::ok_exact),
            null_count: self
                .get_as::<u64>(Stat::NullCount)
                .map(DirectionalBound::into_value)
                .and_then(Precision::ok_exact),
            bit_width_freq,
            trailing_zero_freq,
            uncompressed_size_in_bytes: self
                .get_as::<u64>(Stat::UncompressedSizeInBytes)
                .map(DirectionalBound::into_value)
                .and_then(Precision::ok_exact),
        };

        crate::flatbuffers::ArrayStats::create(fbb, stat_args)
    }
}
