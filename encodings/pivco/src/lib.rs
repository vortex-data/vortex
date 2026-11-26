// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-reset `Pow2-Bin + PIVCO` wavelet-tree integer encoding (`vortex.pivco`).
//!
//! Combines `Pow2-Bin + Raw-Bits` outer value decomposition (`150` bounded symbols + parallel `u64`
//! `raw_bits` stream) with Marcin Żukowski's bottom-up SIMD wavelet-tree symbol codec (`PH` raw
//! bitmaps + `PHA` skewed-bitmap compression + equal-depth flat-subtree SIMD unpack) and optional
//! `OrderPreserving` alphabet trees for direct `<` / `<=` / `>` / `>=` bitmap predicate pushdown.
//!
//! # Attribution
//!
//! The bottom-up SIMD wavelet-tree codec, `SABD` two-table merge, and equal-depth flat-subtree
//! design are based on Marcin Żukowski's Apache-2.0 licensed [`pivco-huffman`] implementation
//! (commit `16b48fcfcbd5cd71d945eb48e2702eec398bb36b`, Copyright Marcin Żukowski) and accompanying
//! paper, Marcin Żukowski, *"PivCo-Huffman"* (`arXiv:2606.05765`, v1.0, 2026-06-04).
//!
//! Vortex extends that symbol codec with:
//! - `Pow2-Bin + Raw-Bits` outer decomposition for full-width 8/16/32/64-bit integers (`150` bins
//!   plus a parallel `u64` verbatim tail-bits stream),
//! - Self-contained `PHA` node bitmap coding: canonical 32-lane (`1` CUDA warp) static Bernoulli
//!   byte-rANS, sparse `u16` minority-index lists (`<= 1.56%`), and 1-byte `AllZero` / `AllOne`
//!   markers,
//! - Optional `OrderPreserving` trees (`rank_to_sym` strictly sorted) enabling direct
//!   `<` / `<=` / `>` / `>=` spine-walking bitmap predicate pushdown without decoding values,
//! - Zero-copy block-level slicing (`offset`, `tree_base_offset`, `raw_base_word_offset`) across
//!   3 split buffers (`[block_headers, tree_payload, raw_bits]`) and fused `ZigZag(Pivco)` signed
//!   integer decoding.
//!
//! [`pivco-huffman`]: https://github.com/MarcinZukowski/pivco-huffman

mod array;
mod bitmap_rans;
mod codec;
mod compute;
mod plugin;
mod pow2_bin;
mod simd;
mod tree;

pub use array::Pivco;
pub use array::PivcoArray;
pub use array::PivcoArrayExt;
pub use array::PivcoArraySlotsExt;
pub use array::PivcoData;
pub use array::PivcoMetadata;
pub use array::PivcoSlots;
pub use array::pivco_id;
pub use codec::BLOCK_HEADER_BYTES_NO_EXTRA;
pub use codec::BLOCK_HEADER_BYTES_WITH_EXTRA;
pub use codec::DEFAULT_BLOCK_SHIFT;
pub use codec::MAX_BLOCK_SHIFT;
pub use codec::MAX_BLOCK_VALUES;
pub use codec::MIN_BLOCK_SHIFT;
pub use codec::PivcoBlockOffset;
pub use codec::PivcoOptions;
pub use codec::block_header_stride;
pub use codec::estimate_compressed_bits;
pub use plugin::PivcoPlugin;
pub use pow2_bin::EBITS_TABLE;
pub use pow2_bin::MAX_SYMS;
pub use pow2_bin::pow2_sym_and_ebits;
pub use tree::PivcoTree;
pub use tree::PivcoTreeMode;
use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable as _;
use vortex_array::scalar_fn::builtin::compare::Compare;
use vortex_array::scalar_fn::builtin::compare::CompareExecuteAdaptor;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;
use vortex_zigzag::ZigZag;

use crate::compute::PivcoZigZagKernel;

/// Register the `vortex.pivco` encoding plugin, parent-reduce rules, and parent-execute kernels.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(PivcoPlugin);
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Slice, Pivco, const { &SliceReduceAdaptor(Pivco) });
    kernels.register_execute_parent_kernel(Filter.id(), Pivco, FilterExecuteAdaptor(Pivco));
    kernels.register_execute_parent_kernel(Dict.id(), Pivco, TakeExecuteAdaptor(Pivco));
    kernels.register_execute_parent_kernel(ZigZag.id(), Pivco, PivcoZigZagKernel);
    kernels.register_execute_parent_kernel(Compare.id(), Pivco, CompareExecuteAdaptor(Pivco));
}

#[cfg(test)]
mod conformance_tests;
#[cfg(test)]
mod tests;
