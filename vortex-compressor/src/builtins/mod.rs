// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in compression schemes that use only `vortex-array` encodings.
//!
//! These schemes produce arrays using types already in `vortex-array` ([`DictArray`], etc.) and
//! have no external encoding crate dependencies.
//!
//! [`DictArray`]: vortex_array::arrays::DictArray

use crate::session::CompressionSessionExt;
mod dict;

pub use dict::BinaryDictScheme;
pub use dict::FloatDictScheme;
pub use dict::IntDictScheme;
pub use dict::StringDictScheme;
pub use dict::float_dictionary_encode;
pub use dict::integer_dictionary_encode;

mod varbin;
pub use varbin::VarBinScheme;

mod constant;
pub use constant::ConstantScheme;
pub use constant::MaskedConstantScheme;

/// Register compression strategies for canonical array encodings.
pub fn initialize(session: &vortex_session::VortexSession) {
    session.register_scheme(&ConstantScheme);
    session.register_scheme(&MaskedConstantScheme);
    session.register_scheme(&IntDictScheme);
    session.register_scheme(&FloatDictScheme);
    session.register_scheme(&StringDictScheme);
    session.register_scheme(&BinaryDictScheme);
    session.register_scheme(&VarBinScheme);
}
