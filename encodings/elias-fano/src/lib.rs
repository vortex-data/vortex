// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Elias-Fano encoding for monotonically non-decreasing integer sequences.
//!
//! Stores about `log2(u / n) + 2` bits per value for `n` values over a universe of `u`, while still
//! answering random access in expected constant time (see [`ef`] for the worst case). Against
//! bit-packing at `ceil(log2(u))` bits the saving is about `log2(n) - 2`, so it widens with row
//! count.
//!
//! Inputs must be non-decreasing; duplicates are fine.
//!
//! [`ef`] holds the codec, which depends on nothing beyond `std`.
//!
//! The sampled select index follows Vigna's [broadword][] construction, with the two-table sampling
//! scheme and its parameters after [`rise-rs`][] (MIT); [`vers`][] was consulted as a further
//! reference. No code is taken from either.
//!
//! [broadword]: https://vigna.di.unimi.it/ftp/papers/Broadword.pdf
//! [`rise-rs`]: https://github.com/AngeloSav/rise-rs
//! [`vers`]: https://github.com/Cydhra/vers

pub mod ef;
