// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! All the built-in encoding schemes and arrays.

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

pub mod primitive;
pub use primitive::Primitive;
pub use primitive::PrimitiveArray;

pub mod scalar_fn;
pub use scalar_fn::ScalarFnArray;
pub use scalar_fn::ScalarFnVTable;

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

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
