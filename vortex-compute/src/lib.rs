// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of compute functions primarily for operating over Vortex vectors.

#![deny(missing_docs)]
#![deny(clippy::missing_panics_doc)]
#![deny(clippy::missing_safety_doc)]

pub mod arithmetic;
#[cfg(feature = "arrow")]
pub mod arrow;
pub mod comparison;
pub mod expand;
pub mod filter;
pub mod logical;
pub mod mask;

/// Functions exported for benchmarking purposes.
#[cfg(feature = "bench")]
pub mod bench {
    #[cfg(target_arch = "aarch64")]
    pub use crate::filter::slice::neon::bench_filter_neon;
    pub use crate::filter::slice::scalar::bench_filter_scalar;
}
