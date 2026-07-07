// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in compression schemes that use only `vortex-array` encodings.
//!
//! These schemes produce arrays using types already in `vortex-array` ([`DictArray`], etc.) and
//! have no external encoding crate dependencies.
//!
//! [`DictArray`]: vortex_array::arrays::DictArray

mod dict;

pub use dict::BinaryDictScheme;
pub use dict::FloatDictScheme;
pub use dict::IntDictScheme;
pub use dict::StringDictScheme;
pub use dict::float_dictionary_encode;
pub use dict::integer_dictionary_encode;
