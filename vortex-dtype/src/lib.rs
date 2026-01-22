// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(target_endian = "little")]
#![deny(missing_docs)]

//! A type system for Vortex
//!
//! This crate contains the core logical type system for Vortex, including the definition of data types,
//! and (optionally) logic for their serialization and deserialization.

#[cfg(feature = "arbitrary")]
mod arbitrary;
pub mod arrow;
mod bigint;
pub mod datetime;
mod decimal;
mod dtype;
pub mod extension;
mod f16;
mod field;
mod field_mask;
mod field_names;
mod native_dtype;
mod nullability;
mod ptype;
mod serde;
pub mod session;
mod struct_;

pub use bigint::*;
pub use decimal::*;
pub use dtype::DType;
pub use dtype::NativeDType;
pub use extension::ExtDType;
pub use extension::ExtID;
pub use f16::*;
pub use field::*;
pub use field_mask::*;
pub use field_names::*;
pub use half;
pub use nullability::*;
pub use ptype::*;
pub use struct_::*;

pub mod proto {
    //! Protocol buffer representations for DTypes
    //!
    //! This module contains the code to serialize and deserialize DTypes to and from protocol buffers.

    pub use vortex_proto::dtype;
}

pub mod flatbuffers {
    //! Flatbuffer representations for DTypes
    //!
    //! This module contains the code to serialize and deserialize DTypes to and from flatbuffers.

    pub use vortex_flatbuffers::dtype::*;

    pub use super::serde::flatbuffers::*;
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_session::VortexSession;

    use crate::session::DTypeSession;

    pub(crate) static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<DTypeSession>());
}
