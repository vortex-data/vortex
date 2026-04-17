// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector compression flavors exercised by the benchmark.
//!
//! Each [`VectorFlavor`] variant maps to a [`vortex::file::WriteStrategyBuilder`] configuration
//! applied to the same input data.
//!
//! The benchmark writes one `.vortex` file per flavor per data file, then scans them all with the
//! same query so the comparison is apples-to-apples with the Parquet files.

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
pub const ALL_VECTOR_FLAVORS: &[VectorFlavor] = &[
    VectorFlavor::Uncompressed,
    VectorFlavor::TurboQuant,
    VectorFlavor::SynchronousHandrolled,
];

/// One write-side compression configuration we measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum VectorFlavor {
    /// `BtrBlocksCompressorBuilder::empty()`
    #[clap(name = "vortex-uncompressed")]
    Uncompressed,
    /// `BtrBlocksCompressorBuilder::default().with_turboquant()`.
    #[clap(name = "vortex-turboquant")]
    TurboQuant,
    /// Hand-rolled non-Vortex baseline: pre-L2-normalized little-endian f32 vectors in a flat
    /// `.f32` file, scanned with a straight-line dot-product loop from synchronous `std::fs` I/O.
    ///
    /// The intent is a "theoretical minimum" ceiling for uncompressed f32 cosine scans. The
    /// `Synchronous` in the name reserves space for a future `AsynchronousHandrolled` variant
    /// that uses `io_uring` on NVMe or async object-store fetches on S3.
    #[clap(name = "baseline-sync-handrolled")]
    SynchronousHandrolled,
    // TODO(connor): We will want to add `Default` here which is just the default compressor.
}

impl VectorFlavor {
    /// Stable kebab-cased label used in CLI args and metric names.
    pub fn label(&self) -> &'static str {
        match self {
            VectorFlavor::Uncompressed => "vortex-uncompressed",
            VectorFlavor::TurboQuant => "vortex-turboquant",
            VectorFlavor::SynchronousHandrolled => "baseline-sync-handrolled",
        }
    }

    /// The `target.format` value emitted on measurements for this flavor. Both Vortex flavors
    /// produce `.vortex` files, so the compression label carries the flavor split; the handrolled
    /// flavor produces a non-Vortex flat binary file.
    pub fn as_format(&self) -> Format {
        match self {
            VectorFlavor::Uncompressed => Format::OnDiskVortex,
            VectorFlavor::TurboQuant => Format::OnDiskVortex,
            VectorFlavor::SynchronousHandrolled => Format::OnDiskVortex,
        }
    }

    /// Subdirectory name under the per-dataset cache root used to store this flavor's shard
    /// files.
    pub fn dir_name(&self) -> &'static str {
        self.label()
    }

    /// Extension (without the leading dot) of the per-shard output file this flavor produces.
    pub fn output_extension(&self) -> &'static str {
        match self {
            VectorFlavor::Uncompressed | VectorFlavor::TurboQuant => "vortex",
            VectorFlavor::SynchronousHandrolled => "f32",
        }
    }

    /// Whether this flavor produces `.vortex` files via the Vortex writer. The handrolled
    /// baselines are non-Vortex and skip [`Self::create_write_options`] entirely.
    pub fn is_vortex(&self) -> bool {
        match self {
            VectorFlavor::Uncompressed | VectorFlavor::TurboQuant => true,
            VectorFlavor::SynchronousHandrolled => false,
        }
    }

    /// Build the [`vortex::file::WriteStrategyBuilder`]-backed write options for this flavor.
    ///
    /// TurboQuant produces `L2Denorm(SorfTransform(...))` which the default file
    /// `ALLOWED_ENCODINGS` set rejects on normalization — we extend the allow-list with the two
    /// scalar-fn array IDs the scheme actually emits.
    ///
    /// # Panics
    ///
    /// Panics when called on a non-Vortex flavor. Callers must guard on [`Self::is_vortex`].
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
            VectorFlavor::SynchronousHandrolled => {
                unreachable!(
                    "create_write_options called on non-Vortex flavor {}; guard on `is_vortex()`",
                    self.label(),
                );
            }
        };

        session.write_options().with_strategy(strategy)
    }
}
