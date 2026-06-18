// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dictionary encoding schemes for binary, integer, float, and string arrays.

mod binary;
mod float;
mod integer;
mod string;

pub use binary::BinaryDictScheme;
pub use float::FloatDictScheme;
pub use float::dictionary_encode as float_dictionary_encode;
pub use integer::IntDictScheme;
pub use integer::dictionary_encode as integer_dictionary_encode;
pub use string::StringDictScheme;
