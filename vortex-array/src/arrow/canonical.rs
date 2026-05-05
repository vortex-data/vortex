// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`CanonicalArrowEncoder`] — fallback Vortex → Arrow encoder for canonical encodings.
//!
//! The canonical encoder handles every `DataType` that maps directly to a canonical Vortex
//! encoding (`Bool`, `Primitive`, `VarBinView`, `ListView`, `Struct`, `FixedSizeList`,
//! `Decimal`, `Extension`). For now it also handles a few non-canonical optimizations
//! (offset-based byte/list arrays); those are slated to move into encoding-keyed
//! [`ArrowEncoder`](super::ArrowEncoder) plugins in subsequent work.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::types::*;
use arrow_schema::DataType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrow::ArrowEncoder;
use crate::arrow::ArrowSession;
use crate::arrow::executor::bool::to_arrow_bool;
use crate::arrow::executor::byte::to_arrow_byte_array;
use crate::arrow::executor::byte_view::to_arrow_byte_view;
use crate::arrow::executor::decimal::to_arrow_decimal;
use crate::arrow::executor::dictionary::to_arrow_dictionary;
use crate::arrow::executor::fixed_size_list::to_arrow_fixed_list;
use crate::arrow::executor::list::to_arrow_list;
use crate::arrow::executor::list_view::to_arrow_list_view;
use crate::arrow::executor::null::to_arrow_null;
use crate::arrow::executor::primitive::to_arrow_primitive;
use crate::arrow::executor::run_end::to_arrow_run_end;
use crate::arrow::executor::struct_::to_arrow_struct;
use crate::arrow::executor::temporal::to_arrow_temporal;
use crate::dtype::DType;

/// Returns the canonical Arrow [`DataType`] for a Vortex [`DType`].
///
/// This mirrors [`DType::to_arrow_dtype`] but is kept separate so encoders can use it without
/// touching the deprecated dtype shim.
pub fn canonical_arrow_type_for_dtype(dtype: &DType) -> VortexResult<DataType> {
    dtype.to_arrow_dtype()
}

/// The default canonical Vortex → Arrow encoder. Registered automatically in
/// [`crate::arrow::ArrowSession::default`].
#[derive(Debug, Default)]
pub struct CanonicalArrowEncoder;

impl ArrowEncoder for CanonicalArrowEncoder {
    fn preferred_arrow_type(
        &self,
        array: &ArrayRef,
        _session: &ArrowSession,
    ) -> VortexResult<Option<DataType>> {
        // The canonical encoder mirrors `DType::to_arrow_dtype()`. Encoding-specific
        // shortcuts (e.g. VarBin → Utf8 instead of Utf8View) live on their own plugins.
        Ok(Some(canonical_arrow_type_for_dtype(array.dtype())?))
    }

    fn to_arrow_array(
        &self,
        array: ArrayRef,
        target: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        let len = array.len();
        let arrow = match target {
            DataType::Null => to_arrow_null(array, ctx),
            DataType::Boolean => to_arrow_bool(array, ctx),
            DataType::Int8 => to_arrow_primitive::<Int8Type>(array, ctx),
            DataType::Int16 => to_arrow_primitive::<Int16Type>(array, ctx),
            DataType::Int32 => to_arrow_primitive::<Int32Type>(array, ctx),
            DataType::Int64 => to_arrow_primitive::<Int64Type>(array, ctx),
            DataType::UInt8 => to_arrow_primitive::<UInt8Type>(array, ctx),
            DataType::UInt16 => to_arrow_primitive::<UInt16Type>(array, ctx),
            DataType::UInt32 => to_arrow_primitive::<UInt32Type>(array, ctx),
            DataType::UInt64 => to_arrow_primitive::<UInt64Type>(array, ctx),
            DataType::Float16 => to_arrow_primitive::<Float16Type>(array, ctx),
            DataType::Float32 => to_arrow_primitive::<Float32Type>(array, ctx),
            DataType::Float64 => to_arrow_primitive::<Float64Type>(array, ctx),
            DataType::Timestamp(..)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_) => to_arrow_temporal(array, target, ctx),
            DataType::Binary => to_arrow_byte_array::<BinaryType>(array, ctx),
            DataType::LargeBinary => to_arrow_byte_array::<LargeBinaryType>(array, ctx),
            DataType::Utf8 => to_arrow_byte_array::<Utf8Type>(array, ctx),
            DataType::LargeUtf8 => to_arrow_byte_array::<LargeUtf8Type>(array, ctx),
            DataType::BinaryView => to_arrow_byte_view::<BinaryViewType>(array, ctx),
            DataType::Utf8View => to_arrow_byte_view::<StringViewType>(array, ctx),
            DataType::List(elements_field) => to_arrow_list::<i32>(array, elements_field, ctx),
            DataType::LargeList(elements_field) => to_arrow_list::<i64>(array, elements_field, ctx),
            DataType::FixedSizeList(elements_field, list_size) => {
                to_arrow_fixed_list(array, *list_size, elements_field, ctx)
            }
            DataType::ListView(elements_field) => {
                to_arrow_list_view::<i32>(array, elements_field, ctx)
            }
            DataType::LargeListView(elements_field) => {
                to_arrow_list_view::<i64>(array, elements_field, ctx)
            }
            DataType::Struct(fields) => to_arrow_struct(array, Some(fields), ctx),
            DataType::Dictionary(codes_type, values_type) => {
                to_arrow_dictionary(array, codes_type, values_type, ctx)
            }
            dt @ (DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)) => to_arrow_decimal(array, dt, ctx),
            DataType::RunEndEncoded(ends_type, values_type) => {
                to_arrow_run_end(array, ends_type.data_type(), values_type, ctx)
            }
            DataType::FixedSizeBinary(_)
            | DataType::Map(..)
            | DataType::Duration(_)
            | DataType::Interval(_)
            | DataType::Union(..) => {
                vortex_bail!("Conversion to Arrow type {target} is not supported");
            }
        }?;

        vortex_ensure!(
            arrow.len() == len,
            "Arrow array length does not match Vortex array length after conversion to {:?}",
            arrow
        );
        Ok(Some(arrow))
    }
}
