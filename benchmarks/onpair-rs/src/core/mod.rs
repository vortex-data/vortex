// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Foundational, dependency-free data types and views.
//!
//! Nothing in `core` performs compression or decompression; it defines the
//! buffers a column is made of (the [`dictionary`] in its compact and wide
//! representations, both behind the `Dictionary` / `DictionaryView` traits), the
//! integers that address them ([`types`], [`offset`]), and their borrowed views.
//! The encoding, decoding, and column layers build on these.

pub(crate) mod dictionary;
pub(crate) mod offset;
pub(crate) mod types;
pub(crate) mod validate;
