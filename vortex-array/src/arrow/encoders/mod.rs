// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in encoding-keyed [`ArrowEncoder`](super::ArrowEncoder) plugins.
//!
//! These plugins ride alongside the [`CanonicalArrowEncoder`](super::canonical::CanonicalArrowEncoder)
//! to short-circuit forward conversion for encodings that have a cheaper Arrow representation
//! than canonicalizing the array first (for example, [`crate::arrays::VarBin`] preferring
//! offset-based [`arrow_schema::DataType::Utf8`] over [`arrow_schema::DataType::Utf8View`]).

pub mod list;
pub mod temporal;
pub mod varbin;
