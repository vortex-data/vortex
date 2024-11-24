use flatbuffers::{FlatBufferBuilder, WIPOffset};
use itertools::Itertools;
use vortex_flatbuffers::WriteFlatBuffer;

use crate::stats::{Stat, Statistics};

impl WriteFlatBuffer for &dyn Statistics {
    type Target<'t> = crate::flatbuffers::ArrayStats<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let trailing_zero_freq = self
            .get_as::<Vec<u64>>(Stat::TrailingZeroFreq)
            .map(|v| v.iter().copied().collect_vec())
            .map(|v| fbb.create_vector(v.as_slice()));

        let bit_width_freq = self
            .get_as::<Vec<u64>>(Stat::BitWidthFreq)
            .map(|v| v.iter().copied().collect_vec())
            .map(|v| fbb.create_vector(v.as_slice()));

        let min = self
            .get(Stat::Min)
            .map(|min| min.into_value().write_flatbuffer(fbb));

        let max = self
            .get(Stat::Max)
            .map(|max| max.into_value().write_flatbuffer(fbb));

        let stat_args = &crate::flatbuffers::ArrayStatsArgs {
            min,
            max,
            is_sorted: self.get_as::<bool>(Stat::IsSorted),
            is_strict_sorted: self.get_as::<bool>(Stat::IsStrictSorted),
            is_constant: self.get_as::<bool>(Stat::IsConstant),
            run_count: self.get_as_cast::<u64>(Stat::RunCount),
            true_count: self.get_as_cast::<u64>(Stat::TrueCount),
            null_count: self.get_as_cast::<u64>(Stat::NullCount),
            bit_width_freq,
            trailing_zero_freq,
            uncompressed_size_in_bytes: self.get_as_cast::<u64>(Stat::UncompressedSizeInBytes),
        };

        crate::flatbuffers::ArrayStats::create(fbb, stat_args)
    }
}
