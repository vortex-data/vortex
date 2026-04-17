// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector compression flavors exercised by the benchmark.
//!
//! Each [`VectorFlavor`] variant maps to a [`vortex::file::WriteStrategyBuilder`] configuration
//! applied to the same input data.
//!
//! The benchmark writes one `.vortex` file per flavor per data file, then scans them all with the
//! same query so the comparison is apples-to-apples with the Parquet files.
//!
//! Note that the handrolled `&[f32]` parquet baseline is **not** a flavor here.

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
use vortex_tensor::encodings::l2_denorm::L2DenormScheme;
use vortex_tensor::scalar_fns::l2_denorm::L2Denorm;
use vortex_tensor::scalar_fns::sorf_transform::SorfTransform;

/// Every [`VectorFlavor`] variant in CLI-help order.
pub const ALL_VECTOR_FLAVORS: &[VectorFlavor] =
    &[VectorFlavor::Uncompressed, VectorFlavor::TurboQuant];

/// One write-side compression configuration we measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum VectorFlavor {
    /// `BtrBlocksCompressorBuilder::empty()`
    #[clap(name = "vortex-uncompressed")]
    Uncompressed,
    /// `BtrBlocksCompressorBuilder::default().with_turboquant()`.
    #[clap(name = "vortex-turboquant")]
    TurboQuant,
    // TODO(connor): We will want to add `Default` here which is just the default compressor.
}

impl VectorFlavor {
    /// Stable kebab-cased label used in CLI args and metric names.
    pub fn label(&self) -> &'static str {
        match self {
            VectorFlavor::Uncompressed => "vortex-uncompressed",
            VectorFlavor::TurboQuant => "vortex-turboquant",
        }
    }

    /// The `target.format` value emitted on measurements for this flavor. Both flavors produce
    /// `.vortex` files, so the compression label carries the flavor split.
    pub fn as_format(&self) -> Format {
        match self {
            VectorFlavor::Uncompressed => Format::OnDiskVortex,
            VectorFlavor::TurboQuant => Format::OnDiskVortex,
        }
    }

    /// Subdirectory name under the per-dataset cache root used to store this flavor's `.vortex`
    /// files.
    pub fn dir_name(&self) -> &'static str {
        match self {
            VectorFlavor::Uncompressed => "vortex-uncompressed",
            VectorFlavor::TurboQuant => "vortex-turboquant",
        }
    }

    /// Build the [`vortex::file::WriteStrategyBuilder`]-backed write options for this flavor.
    ///
    /// TurboQuant produces `L2Denorm(SorfTransform(...))` which the default file
    /// `ALLOWED_ENCODINGS` set rejects on normalization — we extend the allow-list with the two
    /// scalar-fn array IDs the scheme actually emits.
    pub fn create_write_options(&self, session: &VortexSession) -> VortexWriteOptions {
        let strategy = match self {
            VectorFlavor::Uncompressed => {
                // Even though this is uncompressed, we still want to denormalize the data first so
                // that the results are fair.
                let compressor = BtrBlocksCompressorBuilder::empty()
                    .with_new_scheme(&L2DenormScheme)
                    .build();

                let mut allowed: HashSet<ArrayId> = ALLOWED_ENCODINGS.clone();
                allowed.insert(L2Denorm.id());

                WriteStrategyBuilder::default()
                    .with_compressor(compressor)
                    .with_allow_encodings(allowed)
                    .build()
            }
            VectorFlavor::TurboQuant => {
                let compressor = BtrBlocksCompressorBuilder::default()
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
