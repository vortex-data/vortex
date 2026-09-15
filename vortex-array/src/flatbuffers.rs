// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FlatBuffers read/write traits, plus bindings generated from this crate's `flatbuffers`
//! schemas. Schemas owned by other crates generate into those crates instead.

mod traits;

pub use traits::*;

/// A serialized array without its buffer (i.e. data).
///
/// `array.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-array/array.fbs")]
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
pub mod array {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/array.rs"));
}

/// A serialized data type.
///
/// `dtype.fbs`:
/// ```flatbuffers
#[doc = include_str!("../flatbuffers/vortex-dtype/dtype.fbs")]
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
pub mod dtype {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/dtype.rs"));
}
