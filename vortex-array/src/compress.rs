// FIXME(ngates): move this file into the compressor
use vortex_error::VortexResult;

use crate::aliases::hash_set::HashSet;
use crate::stats::PRUNING_STATS;
use crate::{Array, ArrayRef, EncodingId};

/// Extendable compression interface, allowing implementations to explore different choices.
pub trait CompressionStrategy {
    /// Compress input array.
    fn compress(&self, array: &dyn Array) -> VortexResult<ArrayRef>;

    /// A set of the IDs of the encodings the compressor can choose from.
    fn used_encodings(&self) -> HashSet<EncodingId>;
}

/// Check that compression did not alter the length of the validity array.
pub fn check_validity_unchanged(arr: &dyn Array, compressed: &dyn Array) {
    let _ = arr;
    let _ = compressed;
    #[cfg(debug_assertions)]
    {
        use vortex_error::VortexExpect;

        let old_validity = arr
            .validity_mask()
            .vortex_expect("failed to compute validity")
            .len();
        let new_validity = compressed
            .validity_mask()
            .vortex_expect("failed to compute validity ")
            .len();

        debug_assert!(
            old_validity == new_validity,
            "validity length changed after compression: {old_validity} -> {new_validity}\n From tree {} To tree {}\n",
            arr.tree_display(),
            compressed.tree_display()
        );
    }
}

/// Check that compression did not alter the dtype
pub fn check_dtype_unchanged(arr: &dyn Array, compressed: &dyn Array) {
    let _ = arr;
    let _ = compressed;
    #[cfg(debug_assertions)]
    {
        debug_assert!(
            arr.dtype() == compressed.dtype(),
            "Compression changed dtype: {} -> {}\nFrom array: {}Into array {}",
            arr.dtype(),
            compressed.dtype(),
            arr.tree_display(),
            compressed.tree_display(),
        );
    }
}

// Check that compression preserved the statistics.
pub fn check_statistics_unchanged(arr: &dyn Array, compressed: &dyn Array) {
    let _ = arr;
    let _ = compressed;
    #[cfg(debug_assertions)]
    {
        use crate::stats::Stat;

        // Run count merge_ordered assumes that the run is "broken" on each chunk, which is a useful estimate but not guaranteed to be correct.
        for (stat, value) in arr
            .statistics()
            .stats_set()
            .into_iter()
            .filter(|(stat, _)| *stat != Stat::RunCount)
        {
            let compressed_scalar = compressed
                .statistics()
                .get_stat(stat)
                .map(|sv| sv.into_scalar(stat.dtype(compressed.dtype())));
            debug_assert_eq!(
                compressed_scalar.clone(),
                Some(value.clone().into_scalar(stat.dtype(arr.dtype()))),
                "Compression changed {stat} from {value} to {:?}",
                compressed_scalar.as_ref(),
            );
        }
    }
}

/// Eagerly compute certain statistics (i.e., pruning stats plus UncompressedSizeInBytes) for an array.
/// This function is intended to be called in compressors, immediately before compression occurs.
pub fn compute_precompression_stats(arr: &dyn Array) -> VortexResult<()> {
    arr.statistics().compute_uncompressed_size_in_bytes();
    arr.statistics().compute_all(PRUNING_STATS).map(|_| ())
}
