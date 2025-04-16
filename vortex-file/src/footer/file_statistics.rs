//! This module defines the file statistics component of the Vortex file footer.
//!
//! File statistics provide metadata about the data in the file, such as min/max values,
//! null counts, and other statistical information that can be used for query optimization
//! and data exploration.
use std::sync::Arc;

use flatbuffers::{FlatBufferBuilder, Follow, WIPOffset};
use itertools::Itertools;
use vortex_array::stats::StatsSet;
use vortex_error::VortexError;
use vortex_flatbuffers::{FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer, footer as fb};

/// Contains statistical information about the data in a Vortex file.
///
/// This struct wraps an array of `StatsSet` objects, each containing statistics
/// for a field or column in the file. These statistics can be used for query
/// optimization and data exploration.
#[derive(Clone, Debug)]
pub(crate) struct FileStatistics(
    /// An array of statistics sets, one for each field or column in the file.
    pub(crate) Arc<[StatsSet]>,
);

impl FlatBufferRoot for FileStatistics {}

impl ReadFlatBuffer for FileStatistics {
    type Source<'a> = fb::FileStatistics<'a>;
    type Error = VortexError;

    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error> {
        let field_stats = fb.field_stats().unwrap_or_default();
        let field_stats: Vec<StatsSet> = field_stats
            .iter()
            .map(|s| StatsSet::read_flatbuffer(&s))
            .try_collect()?;
        Ok(Self(Arc::from(field_stats)))
    }
}

impl WriteFlatBuffer for FileStatistics {
    type Target<'a> = fb::FileStatistics<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let field_stats = self.0.iter().map(|s| s.write_flatbuffer(fbb)).collect_vec();
        let field_stats = fbb.create_vector(field_stats.as_slice());

        fb::FileStatistics::create(
            fbb,
            &fb::FileStatisticsArgs {
                field_stats: Some(field_stats),
            },
        )
    }
}
