// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A type system for Vortex.
//!
//! This crate contains the core logical type system for Vortex, including the definition of data types,
//! and (optionally) logic for their serialization and deserialization.
//!
//! Vortex dtypes are logical domains, not physical layouts. Encodings are tracked separately on
//! arrays, so the same dtype may be backed by canonical buffers, dictionary codes, compressed
//! children, or a lazy expression array.
//!
//! Every non-null logical dtype carries [`Nullability`] directly. This differs from Apache Arrow,
//! where nullability is usually field metadata.

#[cfg(feature = "arbitrary")]
mod arbitrary;
pub mod arrow;
mod bigint;
mod coercion;
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
#[allow(
    clippy::derived_hash_with_manual_eq,
    reason = "manual PartialEq adds Arc::ptr_eq fast path only"
)]
#[derive(Debug, Clone, Eq, Hash)]
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

    /// A logical union (sum) type.
    ///
    /// `Union` is reserved for values that are drawn from one of several possible child domains.
    /// The exact child-type metadata and canonical storage are not yet part of the stable public
    /// Rust API, so most callers should prefer [`DType::Variant`] for dynamically typed values or
    /// [`DType::Struct`] for fixed schemas.
    Union(Nullability),

    /// Dynamically typed values stored as Vortex scalars.
    ///
    /// A variant value preserves its full logical value in variant storage and may also carry a
    /// typed, row-aligned shredded representation for selected paths when materialized as a
    /// [`VariantArray`](crate::arrays::VariantArray).
    Variant(Nullability),

    /// A user-defined extension type.
    ///
    /// See [`ExtDTypeRef`] for more information.
    Extension(ExtDTypeRef),
}

impl PartialEq for DType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Primitive(pa, na), Self::Primitive(pb, nb)) => pa == pb && na == nb,
            (Self::Decimal(da, na), Self::Decimal(db, nb)) => da == db && na == nb,
            (Self::Utf8(a), Self::Utf8(b)) => a == b,
            (Self::Binary(a), Self::Binary(b)) => a == b,
            (Self::List(da, na), Self::List(db, nb)) => {
                na == nb && (Arc::ptr_eq(da, db) || da == db)
            }
            (Self::FixedSizeList(da, sa, na), Self::FixedSizeList(db, sb, nb)) => {
                sa == sb && na == nb && (Arc::ptr_eq(da, db) || da == db)
            }
            // StructFields handles its own Arc::ptr_eq in its PartialEq impl.
            (Self::Struct(a, na), Self::Struct(b, nb)) => na == nb && a == b,
            (Self::Union(a), Self::Union(b)) => a == b,
            (Self::Variant(a), Self::Variant(b)) => a == b,
            (Self::Extension(a), Self::Extension(b)) => a == b,
            // Every variant is listed in the first position so that adding a new
            // variant produces a non-exhaustive match compile error.
            (Self::Null, _)
            | (Self::Bool(_), _)
            | (Self::Primitive(..), _)
            | (Self::Decimal(..), _)
            | (Self::Utf8(_), _)
            | (Self::Binary(_), _)
            | (Self::List(..), _)
            | (Self::FixedSizeList(..), _)
            | (Self::Struct(..), _)
            | (Self::Union(..), _)
            | (Self::Variant(_), _)
            | (Self::Extension(_), _) => false,
        }
    }
}

pub use bigint::*;
pub use decimal::*;
pub use dtype_impl::NativeDType;
pub use f16::*;
pub use field::*;
pub use field_mask::*;
pub use field_names::*;
pub use half;
pub use nullability::*;
pub use ptype::*;
pub use struct_::*;

use crate::dtype::extension::ExtDTypeRef;

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

    pub(crate) static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);
}
