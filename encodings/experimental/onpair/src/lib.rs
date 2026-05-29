// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex string array backed by the [OnPair][onpair] short-string
//! compression library, with `cast` and `filter` pushdown.
//!
//! The default training preset is `dict-12` (12 bits per token, dictionary
//! capped at 4 096 entries). See [`onpair_compress`] for the entry point and
//! [`OnPairArray`] for the resulting array type.
//!
//! [onpair]: https://arxiv.org/abs/2508.02280

mod array;
mod canonical;
mod compress;
mod compute;
mod decode;
mod kernel;
mod ops;
mod rules;
#[cfg(test)]
mod tests;

pub use array::*;
pub use compress::*;
pub use onpair::Bits;
pub use onpair::Config;
pub use onpair::Error as OnPairError;
pub use onpair::Threshold;
