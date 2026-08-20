// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex string array backed by the [OnPair][onpair] short-string
//! compression library, with pushdown for common string operations.
//!
//! See [`onpair_compress`] for the compression entry point and [`OnPairArray`]
//! for the encoded representation of non-null inputs.
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
pub use onpair::CompactDictionaryView;
pub use onpair::Config;
pub use onpair::DEFAULT_CONFIG;
pub use onpair::DictionaryView;
pub use onpair::Error as OnPairError;
pub use onpair::MAX_TOKEN_SIZE;
pub use onpair::MaxDictBits;
pub use onpair::Threshold;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize OnPair encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(OnPair);
    kernel::initialize(session);
}
