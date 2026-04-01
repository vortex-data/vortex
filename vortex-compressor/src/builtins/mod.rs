// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in compression schemes that use only `vortex-array` encodings.
//!
//! These schemes produce arrays using types already in `vortex-array` ([`ConstantArray`],
//! [`DictArray`], [`MaskedArray`], etc.) and have no external encoding crate dependencies.
//!
//! [`ConstantArray`]: vortex_array::arrays::ConstantArray
//! [`DictArray`]: vortex_array::arrays::DictArray
//! [`MaskedArray`]: vortex_array::arrays::MaskedArray

pub use constant::FloatConstantScheme;
pub use constant::IntConstantScheme;
pub use constant::StringConstantScheme;
pub use dict::FloatDictScheme;
pub use dict::IntDictScheme;
pub use dict::StringDictScheme;
pub use dict::float::dictionary_encode as float_dictionary_encode;
pub use dict::integer::dictionary_encode as integer_dictionary_encode;

mod constant;
mod dict;

use vortex_array::Canonical;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;

/// Returns `true` if the canonical array is a primitive with an integer ptype.
pub fn is_integer_primitive(canonical: &Canonical) -> bool {
    matches!(canonical, Canonical::Primitive(p) if p.ptype().is_int())
}

/// Returns `true` if the canonical form represents a floating-point primitive.
pub fn is_float_primitive(canonical: &Canonical) -> bool {
    matches!(canonical, Canonical::Primitive(p) if !p.ptype().is_int())
}

/// Returns `true` if the canonical array is a UTF-8 string type.
pub fn is_utf8_string(canonical: &Canonical) -> bool {
    matches!(canonical,
        Canonical::VarBinView(v) if
            v.dtype().eq_ignore_nullability(&DType::Utf8(Nullability::NonNullable))
    )
}
