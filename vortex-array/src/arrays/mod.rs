// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! All the built-in encoding schemes and arrays.

#[cfg(any(test, feature = "_test-harness"))]
mod assertions;

#[cfg(any(test, feature = "_test-harness"))]
pub use assertions::format_indices;

#[cfg(test)]
mod validation_tests;

#[cfg(any(test, feature = "_test-harness"))]
pub mod dict_test;

pub mod bool;
pub use bool::BoolArray;
pub use bool::BoolVTable;

pub mod chunked;
pub use chunked::ChunkedArray;
pub use chunked::ChunkedVTable;

pub mod constant;
pub use constant::ConstantArray;
pub use constant::ConstantVTable;

pub mod datetime;
pub use datetime::TemporalArray;

pub mod decimal;
pub use decimal::DecimalArray;
pub use decimal::DecimalVTable;

pub mod dict;
pub use dict::DictArray;
pub use dict::DictVTable;

pub mod extension;
pub use extension::ExtensionArray;
pub use extension::ExtensionVTable;

pub mod filter;
pub use filter::FilterArray;
pub use filter::FilterVTable;

pub mod fixed_size_list;
pub use fixed_size_list::FixedSizeListArray;
pub use fixed_size_list::FixedSizeListVTable;

pub mod list;
pub use list::ListArray;
pub use list::ListVTable;

pub mod listview;
pub use listview::ListViewArray;
pub use listview::ListViewVTable;

pub mod masked;
pub use masked::MaskedArray;
pub use masked::MaskedVTable;

pub mod null;
pub use null::NullArray;
pub use null::NullVTable;

pub mod primitive;
pub use primitive::PrimitiveArray;
pub use primitive::PrimitiveVTable;

pub mod scalar_fn;
pub use scalar_fn::ScalarFnArray;
pub use scalar_fn::ScalarFnVTable;

pub mod shared;
pub use shared::SharedArray;
pub use shared::SharedVTable;

pub mod slice;
pub use slice::SliceArray;
pub use slice::SliceVTable;

pub mod struct_;
pub use struct_::StructArray;
pub use struct_::StructVTable;

pub mod varbin;
pub use varbin::VarBinArray;
pub use varbin::VarBinVTable;

pub mod varbinview;
pub use varbinview::VarBinViewArray;
pub use varbinview::VarBinViewVTable;

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
