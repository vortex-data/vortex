// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-flavor `.vortex` file paths derived from the cached parquet layout.
//!
//! ```text
//! <vortex-bench/data>/vector-search/<dataset>/<layout>/
//!     train/                         (parquet shards, owned by the catalog)
//!     vortex-uncompressed/           (one .vortex file per train shard)
//!     vortex-turboquant/             (one .vortex file per train shard)
//! ```
//!
//! Path construction is mechanical and lives here so the prepare and scan modules can stay
//! focused on the streaming pipeline.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use vortex_bench::vector_dataset::VectorDataset;
use vortex_bench::vector_dataset::layout::TrainLayout;
use vortex_bench::vector_dataset::paths as catalog_paths;

use crate::compression::VortexCompression;

/// `<dataset_dir>/<flavor>/`.
pub fn flavor_dir(ds: VectorDataset, layout: TrainLayout, flavor: VortexCompression) -> PathBuf {
    catalog_paths::dataset_dir(ds, layout).join(flavor.dir_name())
}

/// Translate a parquet shard path to its `.vortex` companion under the flavor directory.
///
/// Just swaps the file extension and rebases the file name into [`flavor_dir`]. The shard
/// stem is preserved so a directory listing pairs `00-of-10.parquet` with `00-of-10.vortex`.
pub fn vortex_path_for_parquet(
    parquet: &Path,
    ds: VectorDataset,
    layout: TrainLayout,
    flavor: VortexCompression,
) -> Result<PathBuf> {
    let stem = parquet
        .file_stem()
        .with_context(|| format!("parquet path {} has no file stem", parquet.display()))?
        .to_owned();
    let mut name = stem;
    name.push(".vortex");
    Ok(flavor_dir(ds, layout, flavor).join(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vortex_path_swaps_extension_and_dir() -> Result<()> {
        let parquet =
            Path::new("/data/vector-search/cohere-large-10m/partitioned/train/03-of-10.parquet");
        let path = vortex_path_for_parquet(
            parquet,
            VectorDataset::CohereLarge10m,
            TrainLayout::Partitioned,
            VortexCompression::TurboQuant,
        )?;
        assert!(
            path.to_string_lossy().ends_with(
                "vector-search/cohere-large-10m/partitioned/vortex-turboquant/03-of-10.vortex"
            ),
            "{}",
            path.display(),
        );
        Ok(())
    }

    #[test]
    fn single_layout_uses_train_stem() -> Result<()> {
        let parquet = Path::new("/data/vector-search/cohere-small-100k/single/train/train.parquet");
        let path = vortex_path_for_parquet(
            parquet,
            VectorDataset::CohereSmall100k,
            TrainLayout::Single,
            VortexCompression::Uncompressed,
        )?;
        assert!(
            path.to_string_lossy()
                .ends_with("vortex-uncompressed/train.vortex")
        );
        Ok(())
    }
}
