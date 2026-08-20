// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The GPU decompression mode behind `--gpu-decompress`.
//!
//! This module is the only place in the crate that mentions the `cuda` feature. The two device
//! backends need it, so they are gated here and reached through [`compressor`]; the rest of the
//! crate calls that one function and stays feature-agnostic.
//!
//! [`GpuOptions`] and [`writer`] deliberately sit outside the gate. Neither touches CUDA, and
//! keeping them unconditional is what lets `main` parse the `--gpu-*` flags — and reject them
//! with a clear message — in a build without the feature.

use vortex_bench::Format;
use vortex_bench::compress::Compressor;

pub mod writer;

#[cfg(feature = "cuda")]
mod parquet;
#[cfg(feature = "cuda")]
mod vortex;

pub use crate::gpu::writer::GpuCodec;

/// Settings for the GPU decompression mode.
#[derive(Clone, Copy, Debug)]
pub struct GpuOptions {
    /// Parquet page codec to write the GPU file with.
    pub codec: GpuCodec,
    /// Cross-check decompressed output against the CPU decoders.
    pub verify: bool,
    /// Read the Vortex file with direct IO instead of through the page cache.
    pub direct_io: bool,
}

/// The GPU backend that measures `format`.
#[cfg(feature = "cuda")]
pub fn compressor(format: Format, options: GpuOptions) -> Box<dyn Compressor> {
    match format {
        Format::OnDiskVortex => Box::new(vortex::GpuVortexCompressor::new(
            options.verify,
            options.direct_io,
        )) as Box<dyn Compressor>,
        Format::Parquet => Box::new(parquet::GpuParquetCompressor::new(
            options.codec,
            options.verify,
        )),
        _ => unimplemented!("GPU compress bench not implemented for {format}"),
    }
}

/// Stands in for [`compressor`] in a build without the `cuda` feature.
///
/// `main` rejects `--gpu-decompress` before any compressor is selected, so reaching this is a
/// bug. Destructuring the options is what marks their fields as read: they are only otherwise
/// used by the gated backends, and without this they are dead code in a non-CUDA build.
#[cfg(not(feature = "cuda"))]
pub fn compressor(format: Format, options: GpuOptions) -> Box<dyn Compressor> {
    let GpuOptions {
        codec: _,
        verify: _,
        direct_io: _,
    } = options;
    unreachable!("GPU mode requires the cuda feature, checked before selecting a {format} backend")
}
