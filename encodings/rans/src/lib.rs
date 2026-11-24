// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! 32-lane block-reset `Pow2-Bin + Raw-Bits` rANS integer encoding (`vortex.rans`).
//!
//! [`RansArray`] compresses arbitrary unsigned and signed primitive integers (`u8..=u64`,
//! `i8..=i64`) into independent `4,096`-element logical blocks (`128` stripes × `32` rANS lanes)
//! sharing a compact per-chunk `2,048`-slot probability table (`PROB_BITS = 11`).
//!
//! Each non-negative (or ZigZag-mapped) value `u: u64` is bijectively decomposed into a bounded
//! symbol `sym in 0..150` (`0..32` exact, `32..150` half-power-of-two bins) plus `ebits` verbatim
//! trailing raw bits. Every `4,096`-row block resets its 32 `u32` rANS states, enabling `O(1)`
//! block skipping in `slice`, `scalar_at`, `take`, and `filter`.

mod array;
pub mod codec;
mod compute;
mod plugin;

pub use array::Rans;
pub use array::RansArray;
pub use array::RansArrayExt;
pub use array::RansArraySlotsExt;
pub use array::RansData;
pub use array::RansSlots;
pub use codec::BLOCK_VALUES;
pub use codec::EBITS_TABLE;
pub use codec::MAX_SYMS;
pub use codec::NUM_LANES;
pub use codec::PROB_BITS;
pub use codec::PROB_SCALE;
pub use codec::STRIPES_PER_BLOCK;
pub use codec::pow2_sym_and_ebits;
pub use codec::zigzag_decode;
pub use codec::zigzag_encode;
pub use plugin::RansPlugin;
pub use plugin::rans_id;
use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Register the `vortex.rans` encoding plugin, parent-reduce rules, and parent-execute kernels.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(RansPlugin);
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Slice, Rans, const { &SliceReduceAdaptor(Rans) });
    kernels.register_execute_parent_kernel(Filter.id(), Rans, FilterExecuteAdaptor(Rans));
    kernels.register_execute_parent_kernel(Dict.id(), Rans, TakeExecuteAdaptor(Rans));
}

/// Serialized protobuf metadata for a [`RansArray`].
#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct RansMetadata {
    /// Element offset within the first `4,096`-value block (`0..4096`, or `0` when empty).
    #[prost(uint32, tag = "1")]
    pub offset: u32,
    /// Whether signed integer values are ZigZag-encoded (`((i << 1) ^ (i >> 63)) as u64`).
    #[prost(bool, tag = "2")]
    pub signed_zigzag: bool,
    /// Whether any active symbol in the chunk has `ebits > 0` (`sym >= 32`).
    #[prost(bool, tag = "3")]
    pub has_extra_bits: bool,
    /// Packed `(sym: u8, freq_le: [u8; 2])` triples (`3` bytes per active symbol, summing to 2048).
    #[prost(bytes, tag = "4")]
    pub sym_freqs: Vec<u8>,
    /// Base byte offset subtracted from `block_headers` offsets when `payload` was sliced.
    #[prost(uint32, tag = "5")]
    pub payload_base_offset: u32,
}

#[cfg(test)]
mod tests;
