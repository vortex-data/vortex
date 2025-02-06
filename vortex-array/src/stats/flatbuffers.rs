use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_flatbuffers::WriteFlatBuffer;

use crate::stats::{Precision, Stat, Statistics};

impl WriteFlatBuffer for &dyn Statistics {
    type Target<'t> = crate::flatbuffers::ArrayStats<'t>;

    /// All statistics written must be exact
    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let trailing_zero_freq = self
            .get_as::<Vec<u64>>(Stat::TrailingZeroFreq)
            .map(|v| v.some_exact().iter().flatten().copied().collect_vec())
            .map(|v| fbb.create_vector(v.as_slice()));

        let bit_width_freq = self
            .get_as::<Vec<u64>>(Stat::BitWidthFreq)
            .map(|v| v.some_exact().iter().flatten().copied().collect_vec())
            .map(|v| fbb.create_vector(v.as_slice()));

        let min = self
            .get_stat(Stat::Min)
            .and_then(Precision::some_exact)
            .map(|min| min.write_flatbuffer(fbb));

        let max = self
            .get_stat(Stat::Max)
            .and_then(Precision::some_exact)
            .map(|max| max.write_flatbuffer(fbb));

        let stat_args = &crate::flatbuffers::ArrayStatsArgs {
            min,
            max,
            is_sorted: self
                .get_as::<bool>(Stat::IsSorted)
                .and_then(Precision::some_exact),
            is_strict_sorted: self
                .get_as::<bool>(Stat::IsStrictSorted)
                .and_then(Precision::some_exact),
            is_constant: self
                .get_as::<bool>(Stat::IsConstant)
                .and_then(Precision::some_exact),
            run_count: self
                .get_as::<u64>(Stat::RunCount)
                .and_then(Precision::some_exact),
            true_count: self
                .get_as::<u64>(Stat::TrueCount)
                .and_then(Precision::some_exact),
            null_count: self
                .get_as::<u64>(Stat::NullCount)
                .and_then(Precision::some_exact),
            bit_width_freq,
            trailing_zero_freq,
            uncompressed_size_in_bytes: self
                .get_as::<u64>(Stat::UncompressedSizeInBytes)
                .and_then(Precision::some_exact),
        };

        crate::flatbuffers::ArrayStats::create(fbb, stat_args)
    }
}
