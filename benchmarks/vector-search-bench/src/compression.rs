// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex write-side compression flavors exercised by the benchmark.
//!
//! Each [`VortexCompression`] variant maps to a [`vortex::file::WriteStrategyBuilder`]
//! configuration applied to the same input data. The benchmark writes one `.vortex` file
//! per flavor per dataset, then scans them all with the same query so the comparison is
//! apples-to-apples.
//!
//! The handrolled `&[f32]` parquet baseline is **not** a flavor here — it sits on a
//! different axis (storage format, not compressor) and lives in
//! [`crate::handrolled_baseline`].

use clap::ValueEnum;
use vortex::array::ArrayId;
use vortex::array::scalar_fn::ScalarFnVTable;
use vortex::file::ALLOWED_ENCODINGS;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_set::HashSet;
use vortex_bench::Format;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_tensor::scalar_fns::l2_denorm::L2Denorm;
use vortex_tensor::scalar_fns::sorf_transform::SorfTransform;

/// One write-side compression configuration we measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum VortexCompression {
    /// `BtrBlocksCompressorBuilder::empty()` — Vortex framing with no compression schemes
    /// enabled. The `emb` column lands as canonical `FixedSizeList<f32>` on disk, so this
    /// is the lossless ceiling on the size axis.
    #[clap(name = "vortex-uncompressed")]
    Uncompressed,
    /// `BtrBlocksCompressorBuilder::empty().with_turboquant()` — only the TurboQuant
    /// scheme is registered, so the `emb` column ends up wrapped as
    /// `L2Denorm(SorfTransform(FixedSizeList(Dict)))`. Lossy; significant size win.
    #[clap(name = "vortex-turboquant")]
    TurboQuant,
}

impl VortexCompression {
    /// Stable kebab-cased label used in CLI args and metric names.
    pub fn label(&self) -> &'static str {
        match self {
            VortexCompression::Uncompressed => "vortex-uncompressed",
            VortexCompression::TurboQuant => "vortex-turboquant",
        }
    }

    /// The `target.format` value emitted on measurements for this flavor. Both flavors
    /// produce `.vortex` files; TurboQuant routes through [`Format::VortexLossy`] so
    /// dashboards can split out the lossy run from the lossless one.
    pub fn as_format(&self) -> Format {
        match self {
            VortexCompression::Uncompressed => Format::OnDiskVortex,
            VortexCompression::TurboQuant => Format::VortexLossy,
        }
    }

    /// Subdirectory name under the per-dataset cache root used to store this flavor's
    /// `.vortex` files.
    pub fn dir_name(&self) -> &'static str {
        match self {
            VortexCompression::Uncompressed => "vortex-uncompressed",
            VortexCompression::TurboQuant => "vortex-turboquant",
        }
    }

    /// Build the [`vortex::file::WriteStrategyBuilder`]-backed write options for this flavor.
    ///
    /// TurboQuant produces `L2Denorm(SorfTransform(...))` which the default file
    /// `ALLOWED_ENCODINGS` set rejects on normalization — we extend the allow-list with
    /// the two scalar-fn array IDs the scheme actually emits.
    pub fn write_options(&self, session: &VortexSession) -> VortexWriteOptions {
        let strategy = match self {
            VortexCompression::Uncompressed => {
                let compressor = BtrBlocksCompressorBuilder::empty().build();
                WriteStrategyBuilder::default()
                    .with_compressor(compressor)
                    .build()
            }
            VortexCompression::TurboQuant => {
                let compressor = BtrBlocksCompressorBuilder::empty()
                    .with_turboquant()
                    .build();
                let mut allowed: HashSet<ArrayId> = ALLOWED_ENCODINGS.clone();
                allowed.insert(L2Denorm.id());
                allowed.insert(SorfTransform.id());
                WriteStrategyBuilder::default()
                    .with_compressor(compressor)
                    .with_allow_encodings(allowed)
                    .build()
            }
        };
        session.write_options().with_strategy(strategy)
    }
}

/// Every [`VortexCompression`] variant in CLI-help order.
pub const ALL_VORTEX_COMPRESSIONS: &[VortexCompression] = &[
    VortexCompression::Uncompressed,
    VortexCompression::TurboQuant,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_round_trips_through_value_enum() {
        for &flavor in ALL_VORTEX_COMPRESSIONS {
            let parsed = VortexCompression::from_str(flavor.label(), true).unwrap();
            assert_eq!(parsed, flavor);
        }
    }

    #[test]
    fn dir_name_matches_label() {
        for &flavor in ALL_VORTEX_COMPRESSIONS {
            assert_eq!(flavor.dir_name(), flavor.label());
        }
    }
}
