// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dictionary encoding schemes for integer, float, and string arrays.

/// Dictionary encoding for low-cardinality float values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatDictScheme;

/// Dictionary encoding for low-cardinality integer values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntDictScheme;

/// Dictionary encoding for low-cardinality string values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StringDictScheme;

mod float;
mod integer;
mod string;

pub use float::dictionary_encode as float_dictionary_encode;
pub use integer::dictionary_encode as integer_dictionary_encode;
