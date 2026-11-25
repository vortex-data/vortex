// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! 32-lane block-reset `Pow2-Bin + Raw-Bits` rANS integer encoding (`vortex.rans`).
//!
//! [`RansArray`] compresses non-negative primitive integers (`u8..=u64`, and `i8..=i64` whose valid
//! values are `>= 0`) into independent `4,096`-element logical blocks (`128` stripes × `32` rANS
//! lanes) sharing a compact per-chunk `2,048`-slot probability table (`PROB_BITS = 11`). Signed
//! arrays with negative values compose via `ZigZag(Rans)` and decode in a single pass through a
//! fused [`ExecuteParentKernel`](vortex_array::kernel::ExecuteParentKernel).
//!
//! Each non-negative value `u: u64` is bijectively decomposed into a bounded symbol `sym in 0..150`
//! (`0..32` exact, `32..150` half-power-of-two bins) plus `ebits` verbatim trailing raw bits.
//! Every `4,096`-row block resets its 32 `u32` rANS states, enabling `O(1)` block skipping in
//! `slice`, `scalar_at`, `take`, and `filter`.

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
pub use codec::BLOCK_HEADER_BYTES;
pub use codec::BLOCK_STATES_BYTES;
pub use codec::BLOCK_VALUES;
pub use codec::EBITS_TABLE;
pub use codec::MAX_SYMS;
pub use codec::NUM_LANES;
pub use codec::PROB_BITS;
pub use codec::PROB_SCALE;
pub use codec::STRIPES_PER_BLOCK;
pub use codec::pow2_sym_and_ebits;
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
use vortex_zigzag::ZigZag;

use crate::compute::RansZigZagKernel;

/// Register the `vortex.rans` encoding plugin, parent-reduce rules, and parent-execute kernels.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(RansPlugin);
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Slice, Rans, const { &SliceReduceAdaptor(Rans) });
    kernels.register_execute_parent_kernel(Filter.id(), Rans, FilterExecuteAdaptor(Rans));
    kernels.register_execute_parent_kernel(Dict.id(), Rans, TakeExecuteAdaptor(Rans));
    kernels.register_execute_parent_kernel(ZigZag.id(), Rans, RansZigZagKernel);
}

/// Serialized protobuf metadata for a [`RansArray`].
#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct RansMetadata {
    /// Element offset within the first `4,096`-value block (`0..4096`, or `0` when empty).
    #[prost(uint32, tag = "1")]
    pub offset: u32,
    /// Whether any active symbol in the chunk has `ebits > 0` (`sym >= 32`).
    #[prost(bool, tag = "2")]
    pub has_extra_bits: bool,
    /// Packed `(sym: u8, freq_le: [u8; 2])` triples (`3` bytes per active symbol, summing to 2048).
    #[prost(bytes, tag = "3")]
    pub sym_freqs: Vec<u8>,
    /// Base byte offset subtracted from `block_headers` offsets when `payload` was sliced.
    #[prost(uint32, tag = "4")]
    pub payload_base_offset: u32,
}

#[cfg(test)]
mod tests;
