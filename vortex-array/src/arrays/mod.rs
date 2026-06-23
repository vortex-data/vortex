// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in array encodings.
//!
//! Canonical arrays are the default uncompressed representation for a logical dtype:
//! [`NullArray`], [`BoolArray`], [`PrimitiveArray`], [`DecimalArray`], [`VarBinViewArray`],
//! [`ListViewArray`], [`FixedSizeListArray`], [`StructArray`], [`ExtensionArray`], and
//! [`VariantArray`].
//!
//! Utility and lazy arrays represent common transformations without immediately materializing
//! their result. Examples include [`ChunkedArray`] for concatenation, [`ConstantArray`] for repeated
//! values, [`DictArray`] for dictionary encoding, [`FilterArray`] for masked rows, [`SliceArray`]
//! for views, and [`ScalarFnArray`] for deferred scalar-function execution.
//!
//! Some public arrays are primarily internal building blocks. Their constructors and extension
//! traits document the stable contract; avoid depending on undocumented slot order or metadata
//! details.

#[cfg(any(test, feature = "_test-harness"))]
mod assertions;

#[cfg(any(test, feature = "_test-harness"))]
pub use assertions::assert_arrays_eq_impl;

#[cfg(test)]
mod validation_tests;

#[cfg(any(test, feature = "_test-harness"))]
pub mod dict_test;

pub mod bool;
pub use bool::Bool;
pub use bool::BoolArray;

pub mod chunked;
pub use chunked::Chunked;
pub use chunked::ChunkedArray;

pub mod constant;
pub use constant::Constant;
pub use constant::ConstantArray;

pub mod datetime;
pub use datetime::TemporalArray;

pub mod decimal;
pub use decimal::Decimal;
pub use decimal::DecimalArray;

pub mod dict;
pub use dict::Dict;
pub use dict::DictArray;

pub mod extension;
pub use extension::Extension;
pub use extension::ExtensionArray;

pub mod filter;
pub use filter::Filter;
pub use filter::FilterArray;

pub mod fixed_size_list;
pub use fixed_size_list::FixedSizeList;
pub use fixed_size_list::FixedSizeListArray;

pub mod interleave;
pub use interleave::Interleave;
pub use interleave::InterleaveArray;

pub mod list;
pub use list::List;
pub use list::ListArray;

pub mod listview;
pub use listview::ListView;
pub use listview::ListViewArray;

pub mod masked;
pub use masked::Masked;
pub use masked::MaskedArray;

pub mod null;
pub use null::Null;
pub use null::NullArray;

pub mod patched;
pub use patched::Patched;
pub use patched::PatchedArray;

pub mod primitive;
pub use primitive::Primitive;
pub use primitive::PrimitiveArray;

pub mod scalar_fn;
pub use scalar_fn::ScalarFn;
pub use scalar_fn::ScalarFnArray;

pub mod shared;
pub use shared::Shared;
pub use shared::SharedArray;

pub mod slice;
pub use slice::Slice;
pub use slice::SliceArray;

pub mod struct_;
pub use struct_::Struct;
pub use struct_::StructArray;

pub mod varbin;
pub use varbin::VarBin;
pub use varbin::VarBinArray;

pub mod varbinview;
pub use varbinview::VarBinView;
pub use varbinview::VarBinViewArray;

pub mod variant;
pub use variant::Variant;
pub use variant::VariantArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    bool::initialize(session);
    chunked::initialize(session);
    decimal::initialize(session);
    dict::initialize(session);
    extension::initialize(session);
    filter::initialize(session);
    fixed_size_list::initialize(session);
    list::initialize(session);
    listview::initialize(session);
    patched::initialize(session);
    primitive::initialize(session);
    struct_::initialize(session);
    varbin::initialize(session);
    varbinview::initialize(session);
    variant::initialize(session);
}

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
