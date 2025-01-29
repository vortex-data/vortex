use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::aliases::hash_set::HashSet;
use crate::encoding::EncodingRef;
use crate::stats::PRUNING_STATS;
use crate::ArrayData;

pub trait CompressionStrategy {
    fn compress(&self, array: &ArrayData) -> VortexResult<ArrayData>;

    fn used_encodings(&self) -> HashSet<EncodingRef>;
}

/// Check that compression did not alter the length of the validity array.
pub fn check_validity_unchanged(arr: &ArrayData, compressed: &ArrayData) {
    let _ = arr;
    let _ = compressed;
    #[cfg(debug_assertions)]
    {
        let old_validity = arr
            .logical_validity()
            .vortex_expect("failed to compute validity")
            .len();
        let new_validity = compressed
            .logical_validity()
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
pub fn check_dtype_unchanged(arr: &ArrayData, compressed: &ArrayData) {
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
pub fn check_statistics_unchanged(arr: &ArrayData, compressed: &ArrayData) {
    let _ = arr;
    let _ = compressed;
    #[cfg(debug_assertions)]
    {
        use crate::stats::Stat;

        // Run count merge_ordered assumes that the run is "broken" on each chunk, which is a useful estimate but not guaranteed to be correct.
        for (stat, value) in arr
            .statistics()
            .to_set()
            .into_iter()
            .filter(|(stat, _)| *stat != Stat::RunCount)
        {
            let compressed_scalar = compressed
                .statistics()
                .get(stat)
                .map(|sv| Scalar::new(stat.dtype(compressed.dtype()), sv));
            debug_assert_eq!(
                compressed_scalar,
                Some(Scalar::new(stat.dtype(arr.dtype()), value.clone())),
                "Compression changed {stat} from {value} to {}",
                compressed_scalar
                    .as_ref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "null".to_string()),
            );
        }
    }
}

/// Eagerly compute certain statistics (i.e., pruning stats plus UncompressedSizeInBytes) for an array.
/// This function is intended to be called in compressors, immediately before compression occurs.
pub fn compute_precompression_stats(arr: &ArrayData) -> VortexResult<()> {
    arr.statistics().compute_uncompressed_size_in_bytes();
    arr.statistics().compute_all(PRUNING_STATS).map(|_| ())
}
