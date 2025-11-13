// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This crate contains support for GPU and CUDA accelerated compute for Vortex.
//!
//! This crate is currently considered unstable, and much of its code is behind a the `gpu_unstable` config.
//! If you wish to use it, you should build your code with:
//! ```shell
//! RUSTFLAGS="--cfg gpu_unstable" cargo build -p ...
//! ```

pub mod bit_unpack;
pub mod for_;
mod for_bp;
mod gpu_array;
mod indent;
mod jit;
mod rle_decompress;
mod take;
mod task;

pub use bit_unpack::{cuda_bit_unpack, cuda_bit_unpack_timed};
pub use for_::{cuda_for_unpack, cuda_for_unpack_timed};
pub use for_bp::{cuda_for_bp_unpack, cuda_for_bp_unpack_timed};
pub use gpu_array::*;
pub use jit::create_run_jit_kernel;
pub use take::cuda_take;
