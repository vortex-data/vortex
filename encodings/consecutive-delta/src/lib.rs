// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array implementing pco's first-order consecutive-delta layer for
//! `i64` inputs. The decoded stream is reconstructed from a stored `seed`
//! (the first element) and a child `primary` array of consecutive
//! differences:
//!
//! ```text
//! encode: delta[i-1] = x[i].wrapping_sub(x[i-1])   for i in 1..N
//!         seed       = x[0]
//!
//! decode: x[0] = seed
//!         x[i] = x[i-1].wrapping_add(delta[i-1])    for i in 1..N
//! ```
//!
//! This phase implements `order = 1` only. The round-trip is bit-exact for
//! any `i64` stream because every arithmetic step uses wrapping arithmetic
//! on the latent type.
//!
//! Higher orders, lookback deltas, and `Conv1` are deferred per the design
//! doc.
//!
//! # Random-access cliff
//!
//! `scalar_at(i)` on a [`ConsecutiveDeltaArray`] must replay the prefix sum
//! from element zero and is therefore **O(i)** without checkpoints. This is
//! intentional: it is the first layer in the layered-pco stack that breaks
//! element-level random access, and quantifying that break is one of the
//! goals of the design.
//!
//! # Nullability
//!
//! Computing a delta to a null neighbour is undefined under wrapping
//! arithmetic. For this phase the encoder rejects any input that is not
//! [`Validity::NonNullable`][nn] with a clear error. Plumbing nulls through
//! consecutive deltas is tracked as a follow-up in
//! `encodings/pco/DESIGN.md`'s open questions.
//!
//! [nn]: vortex_array::validity::Validity::NonNullable
//!
//! See [`ConsecutiveDeltaArray`] for the public type.

pub use array::*;

mod array;

/// Prost-encoded metadata for a [`ConsecutiveDeltaArray`].
///
/// `seed` is the first absolute value of the encoded stream, used as the
/// running accumulator when decode replays the prefix sum.
#[derive(Clone, prost::Message)]
pub struct ConsecutiveDeltaMetadata {
    /// First absolute value (`x[0]`). For empty input this field is `0`
    /// and the `primary` child is also empty.
    #[prost(int64, tag = "1")]
    pub seed: i64,
}
