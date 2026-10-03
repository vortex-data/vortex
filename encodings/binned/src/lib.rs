// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A binned tANS numeric codec with block-level random access.
//!
//! Like Pco, numbers become unsigned latents that are split into bins: each bin index is tANS
//! coded and each offset within its bin is stored as raw bits. Unlike Pco, values live in small,
//! independently decodable blocks behind a compact index, so reading one value decodes at most
//! one block, and the decode tables are built once per array rather than once per read.
//!
//! [`modes`] holds the elementwise transforms (Pco's modes) that split numbers into one or two
//! latent streams; [`stream`] codes each latent stream.

use vortex_array::session::ArraySessionExt;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

mod ans;
mod array;
mod bins;
mod bits;
pub mod editions;
mod latent;
mod lookback;
pub mod modes;
mod rules;
mod packed;
pub mod serde;
pub mod stream;
#[cfg(test)]
mod tests;

pub use array::Binned;
pub use array::BinnedArray;
pub use array::BinnedArrayExt;
pub use array::BinnedData;
pub use latent::Latent;
pub use latent::Number;
pub use modes::Decoder;
pub use modes::Encoded;
pub use modes::Mode;
pub use modes::Numeric;
pub use modes::choose_lookback;
pub use modes::choose_mode;
pub use modes::compress;
pub use modes::compress_with_mode;
pub use packed::Packed;
pub use stream::BLOCK_SIZE;
pub use stream::Config;

/// Registers the `vortex.binned` encoding and declares its opt-in edition, without enabling it.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(Binned);
    if session.editions().find(&editions::BINNED_2026_10).is_none() {
        session
            .editions()
            .declare_family(&editions::FAMILY)
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("binned edition family is valid");
        session
            .register_edition(&editions::DECLARATION)
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("binned edition declaration is valid");
    }
}
