// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
mod dtype_impl;
pub mod extension;
mod f16;
mod field;
mod field_mask;
mod field_names;
mod native_dtype;
mod nullability;
mod ptype;
pub mod serde;
pub mod session;
mod struct_;

use std::sync::Arc;

/// The logical types of elements in Vortex arrays.
///
/// `DType` represents the different logical data types that can be represented in a Vortex array.
///
/// This is different from physical types, which represent the actual layout of data (compressed or
/// uncompressed). The set of physical types/formats (or data layout) is surjective into the set of
/// logical types (or in other words, all physical types map to a single logical type).
///
/// Note that a `DType` represents the logical type of the elements in the `Array`s, **not** the
/// logical type of the `Array` itself.
///
/// For example, an array with [`DType::Primitive`]([`I32`], [`NonNullable`]) could be physically
/// encoded as any of the following:
///
/// - A flat array of `i32` values.
/// - A run-length encoded sequence.
/// - Dictionary encoded values with bitpacked codes.
///
/// All of these physical encodings preserve the same logical [`I32`] type, even if the physical
/// data is different.
///
/// [`I32`]: PType::I32
/// [`NonNullable`]: Nullability::NonNullable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    /// A logical null type.
    ///
    /// `Null` only has a single value, `null`.
    Null,

    /// A logical boolean type.
    ///
    /// `Bool` can be `true` or `false` if non-nullable. It can be `true`, `false`, or `null` if
    /// nullable.
    Bool(Nullability),

    /// A logical fixed-width numeric type.
    ///
    /// This can be unsigned, signed, or floating point. See [`PType`] for more information.
    Primitive(PType, Nullability),

    /// Logical real numbers with fixed precision and scale.
    ///
    /// See [`DecimalDType`] for more information.
    Decimal(DecimalDType, Nullability),

    /// Logical UTF-8 strings.
    Utf8(Nullability),

    /// Logical binary data.
    Binary(Nullability),

    /// A logical variable-length list type.
    ///
    /// This is parameterized by a single `DType` that represents the element type of the inner
    /// lists.
    List(Arc<DType>, Nullability),

    /// A logical fixed-size list type.
    ///
    /// This is parameterized by a `DType` that represents the element type of the inner lists, as
    /// well as a `u32` size that determines the fixed length of each `FixedSizeList` scalar.
    FixedSizeList(Arc<DType>, u32, Nullability),

    /// A logical struct type.
    ///
    /// A `Struct` type is composed of an ordered list of fields, each with a corresponding name and
    /// `DType`. See [`StructFields`] for more information.
    Struct(StructFields, Nullability),

    /// A user-defined extension type.
    ///
    /// See [`ExtDTypeRef`] for more information.
    Extension(ExtDTypeRef),
}

pub use bigint::*;
pub use decimal::*;
pub use dtype_impl::NativeDType;
pub use extension::ExtDType;
pub use extension::ExtDTypeRef;
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
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_session::VortexSession;

    use crate::dtype::session::DTypeSession;

    pub(crate) static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<DTypeSession>());
}
