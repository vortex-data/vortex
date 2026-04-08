// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant vector quantization encoding for Vortex.
//!
//! Implements a Stage 1 TurboQuant encoding ([arXiv:2504.19874], [RFC 0033]) for lossy
//! compression of high-dimensional vector data. The encoding operates on [`Vector`] extension
//! arrays, compressing their `FixedSizeList` storage into quantized codes after a structured
//! orthogonal surrogate rotation.
//!
//! [arXiv:2504.19874]: https://arxiv.org/abs/2504.19874
//! [RFC 0033]: https://vortex-data.github.io/rfcs/rfc/0033.html
//! [`Vector`]: crate::vector::Vector
//!
//! # Overview
//!
//! TurboQuant minimizes mean-squared reconstruction error (1-8 bits per coordinate)
//! using MSE-optimal scalar quantization on coordinates of a rotated unit vector.
//!
//! The TurboQuant paper analyzes a full random orthogonal rotation. The current Vortex
//! implementation instead uses a fixed 3-round Walsh-Hadamard-based structured transform with
//! random sign diagonals. This is a practical approximation chosen for encode/decode efficiency,
//! and should be understood as an implementation choice rather than the exact construction used in
//! the paper's proofs.
//!
//! The current encoding is also intentionally MSE-only. It does not yet implement the paper's QJL
//! residual correction for unbiased inner-product estimation, and it still uses internal
//! power-of-2 padding rather than the block decomposition proposed in RFC 0033.
//!
//! # Theoretical error bounds
//!
//! For unit-norm vectors quantized at `b` bits per coordinate, the paper's Theorem 1
//! guarantees normalized MSE distortion:
//!
//! > `E[||x - x_hat||^2 / ||x||^2] <= (sqrt(3) * pi / 2) / 4^b`
//!
//! | Bits | MSE bound  | Quality           |
//! |------|------------|-------------------|
//! | 1    | 6.80e-01   | Poor              |
//! | 2    | 1.70e-01   | Usable for ANN    |
//! | 3    | 4.25e-02   | Good              |
//! | 4    | 1.06e-02   | Very good         |
//! | 5    | 2.66e-03   | Excellent         |
//! | 6    | 6.64e-04   | Near-lossless     |
//! | 7    | 1.66e-04   | Near-lossless     |
//! | 8    | 4.15e-05   | Near-lossless     |
//!
//! # Compression ratios
//!
//! Each vector is stored as `padded_dim * bit_width / 8` bytes of quantized codes plus one stored
//! norm. In the current implementation, that norm uses the vector's element float type, not a
//! separate fixed storage precision. Non-power-of-2 dimensions are padded to the next power of 2
//! for the structured rotation, which reduces the effective ratio for those dimensions.
//!
//! The table below assumes f32 input, so the stored norm is 4 bytes.
//!
//! | dim  | padded | bits | f32 bytes | TQ bytes | ratio  |
//! |------|--------|------|-----------|----------|--------|
//! |  768 |   1024 |    2 |      3072 |      260 | 11.8x  |
//! | 1024 |   1024 |    2 |      4096 |      260 | 15.8x  |
//! |  768 |   1024 |    4 |      3072 |      516 |  6.0x  |
//! | 1024 |   1024 |    4 |      4096 |      516 |  7.9x  |
//! |  768 |   1024 |    8 |      3072 |     1028 |  3.0x  |
//! | 1024 |   1024 |    8 |      4096 |     1028 |  4.0x  |
//!
//! # Example
//!
//! ```
//! use vortex_array::IntoArray;
//! use vortex_array::VortexSessionExecute;
//! use vortex_array::arrays::ExtensionArray;
//! use vortex_array::arrays::FixedSizeListArray;
//! use vortex_array::arrays::PrimitiveArray;
//! use vortex_array::dtype::extension::ExtDType;
//! use vortex_array::extension::EmptyMetadata;
//! use vortex_array::validity::Validity;
//! use vortex_buffer::BufferMut;
//! use vortex_array::session::ArraySession;
//! use vortex_session::VortexSession;
//! use vortex_tensor::encodings::turboquant::{TurboQuantConfig, turboquant_encode};
//! use vortex_tensor::vector::Vector;
//!
//! // Create a Vector extension array of 100 random 128-d vectors.
//! let num_rows = 100;
//! let dim = 128u32;
//! let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim as usize);
//! for i in 0..(num_rows * dim as usize) {
//!     buf.push((i as f32 * 0.001).sin());
//! }
//! let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
//! let fsl = FixedSizeListArray::try_new(
//!     elements.into_array(), dim, Validity::NonNullable, num_rows,
//! ).unwrap();
//! let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())
//!     .unwrap().erased();
//! let ext = ExtensionArray::new(ext_dtype, fsl.into_array());
//!
//! // Quantize at 2 bits per coordinate.
//! let config = TurboQuantConfig { bit_width: 2, seed: Some(42), num_rounds: 3 };
//! let session = VortexSession::empty().with::<ArraySession>();
//! let mut ctx = session.create_execution_ctx();
//! let encoded = turboquant_encode(ext.as_view(), &config, &mut ctx).unwrap();
//!
//! // Verify compression: 100 vectors x 128 dims x 4 bytes = 51200 bytes input.
//! assert!(encoded.nbytes() < 51200);
//! ```

mod array;
pub use array::data::TurboQuantArrayExt;
pub use array::data::TurboQuantData;

pub(crate) mod compute;

mod metadata;

mod vtable;

pub use vtable::TurboQuant;
pub use vtable::TurboQuantArray;

mod scheme;
pub use scheme::TurboQuantScheme;
pub use scheme::compress::TurboQuantConfig;
pub use scheme::compress::turboquant_encode;

#[cfg(test)]
mod tests;
