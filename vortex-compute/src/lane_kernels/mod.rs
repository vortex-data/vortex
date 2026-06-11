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
//! - [`source`] — the [`IndexedSource`] trait, [`LaneZip`], and read-only adapters.
//! - [`sink`] — the [`IndexedSink`] trait and [`ReinterpretSink`].
//! - [`map_into`] — out-of-place kernels via [`IndexedSourceExt`] (writes into a
//!   caller-provided `&mut [MaybeUninit<R>]`).
//! - [`map_in_place`] — in-place kernels via [`IndexedSinkExt`] (writes back through
//!   the sink itself).
//!
//! The kernels never allocate. Both kernel families handle a mask with a non-byte-aligned
//! offset and with a logical `len` shorter than the underlying byte buffer, via
//! `BitBuffer::chunks`.

pub mod map_in_place;
pub mod map_into;
pub mod sink;
pub mod source;

pub use map_in_place::IndexedSinkExt;
pub use map_into::IndexedSourceExt;
pub use sink::IndexedSink;
pub use sink::ReinterpretSink;
pub use source::IndexedSource;
pub use source::LaneZip;
