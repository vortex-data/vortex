#![cfg(target_endian = "little")]
#![deny(missing_docs)]

//! A type system for Vortex
//!
//! This crate contains the core logical type system for Vortex, including the definition of data types,
//! and (optionally) logic for their serialization and deserialization.

pub use decimal::*;
pub use dtype::*;
pub use extension::*;
pub use field::*;
pub use field_mask::*;
pub use half;
pub use nullability::*;
pub use ptype::*;
pub use struct_::*;

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arrow")]
pub mod arrow;
pub mod datetime;
mod decimal;
mod dtype;
mod extension;
mod field;
mod field_mask;
mod nullability;
mod ptype;
mod serde;
mod struct_;

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
