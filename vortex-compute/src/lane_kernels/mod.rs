// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Elementwise lane kernels over indexed sources.
//!
//! Replaces `&[T]` with an [`IndexedSource`] trait: each lane read is
//! `unsafe fn get_unchecked(i) -> Item`, independent across iterations. For `&[T]`
//! this inlines to the same indexed load as the slice kernel; for [`LaneZip`]`(&[A], &[B])`
//! it gives two independent indexed reads per lane — both shapes the auto-vectorizer
//! handles.
//!
//! The module is split into:
//!
//! - [`mask_words`] — walks a mask one `u64` word at a time, for kernels over 64-lane blocks.
//! - [`source`] — the [`IndexedSource`] trait, [`LaneZip`], and read-only adapters.
//! - [`sink`] — the [`IndexedSink`] trait and [`ReinterpretSink`].
//! - [`map_into`] — out-of-place kernels via [`IndexedSourceExt`] (writes into a
//!   caller-provided `&mut [MaybeUninit<R>]`).
//! - [`map_in_place`] — in-place kernels via [`IndexedSinkExt`] (writes back through
//!   the sink itself).
//!
//! The kernels never allocate. Both kernel families handle a mask with a non-byte-aligned
//! offset and with a logical `len` shorter than the underlying byte buffer, via
//! [`for_each_mask_word`].

pub mod map_in_place;
pub mod map_into;
pub mod mask_words;
pub mod sink;
pub mod source;

pub use map_in_place::IndexedSinkExt;
pub use map_into::IndexedSourceExt;
pub use mask_words::MaskWords;
pub use mask_words::for_each_mask_word;
pub use mask_words::for_each_masked_value;
pub use mask_words::low_bits_mask;
pub use mask_words::try_for_each_mask_word;
pub use sink::IndexedSink;
pub use sink::ReinterpretSink;
pub use source::IndexedSource;
pub use source::LaneZip;

/// Loop-tiling chunk length for the **no-mask** kernels ([`IndexedSourceExt::map_into`],
/// [`IndexedSourceExt::try_map_into`], [`IndexedSinkExt::map_into_in_place`]).
///
/// This is a pure tuning knob: those kernels split the lane range into
/// `len / CHUNK_LEN` full chunks plus a remainder, so any value yields correct
/// results and only the codegen/tiling changes. Vary it to tune performance.
///
/// It does **not** apply to the masked or bit-packed kernels: those consume one
/// u64 validity word per chunk from [`for_each_mask_word`] and pack per-lane fail bits
/// with `<< bit_idx` into a `u64`, so their chunk length is locked to 64.
pub(crate) const CHUNK_LEN: usize = 64;
