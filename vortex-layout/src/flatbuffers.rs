// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bindings generated from this crate's `flatbuffers` schema.

/// Structures describing the physical layout of Vortex arrays in random access storage.
///
/// `layout.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-layout/layout.fbs")]
/// ```
#[allow(clippy::all)]
#[allow(clippy::absolute_paths)]
#[allow(clippy::borrow_as_ptr)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::unwrap_used)]
#[allow(dead_code)]
#[allow(mismatched_lifetime_syntaxes)]
#[allow(missing_docs)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused_imports)]
#[allow(unused_lifetimes)]
#[allow(unused_qualifications)]
pub mod layout {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/layout.rs"));
}
