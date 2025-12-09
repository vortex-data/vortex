// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion logic between Vortex vector types and Arrow types.

use arrow_array::types::Decimal128Type;
use arrow_array::types::Decimal256Type;
use arrow_array::types::Decimal32Type;
use arrow_array::types::Decimal64Type;
use arrow_array::types::Float16Type;
use arrow_array::types::Float32Type;
use arrow_array::types::Float64Type;
use arrow_array::types::Int16Type;
use arrow_array::types::Int32Type;
use arrow_array::types::Int64Type;
use arrow_array::types::Int8Type;
use arrow_array::types::StringViewType;
use arrow_array::types::UInt16Type;
use arrow_array::types::UInt32Type;
use arrow_array::types::UInt64Type;
use arrow_array::types::UInt8Type;
use arrow_array::Array;
use arrow_array::BooleanArray;
use arrow_array::FixedSizeListArray;
use arrow_array::GenericByteViewArray;
use arrow_array::NullArray;
use arrow_array::PrimitiveArray;
use arrow_array::StructArray;
use arrow_schema::DataType;
use vortex_error::vortex_bail;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorMutOps;

mod binaryview;
mod bool;
mod decimal;
mod fixed_size_list;
mod list;
mod null;
mod primitive;
mod struct_;
mod vector;

/// Trait for converting Vortex vector objects into Arrow.
pub trait IntoArrow {
    /// The output Arrow type.
    type Output;

    /// Convert the Vortex vector object into an Arrow object.
    fn into_arrow(self) -> VortexResult<Self::Output>;
}

/// Trait for converting Arrow objects into Vortex vector objects.
pub trait IntoVector {
    /// The output Vortex vector type.
    type Output;

    /// Convert the Arrow object into a Vortex vector object.
    fn into_vector(self) -> VortexResult<Self::Output>;
}

impl IntoArrow for Datum {
    type Output = Box<dyn arrow_array::Datum>;

    fn into_arrow(self) -> VortexResult<Self::Output> {
        match self {
            Datum::Scalar(s) => Ok(Box::new(arrow_array::Scalar::new(
                s.repeat(1).freeze().into_arrow()?,
            ))),
            Datum::Vector(v) => Ok(Box::new(v.into_arrow()?)),
        }
    }
}

impl IntoVector for &dyn Array {
    type Output = Vector;

    #[allow(clippy::unwrap_used)]
    fn into_vector(self) -> VortexResult<Self::Output> {
        // The downcast_ref calls below are guaranteed to succeed because we match on data_type()
        // first and each branch only attempts to downcast to the corresponding Arrow type.
        match self.data_type() {
            DataType::Null => self
                .as_any()
                .downcast_ref::<NullArray>()
                .vortex_expect("NullArray downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Boolean => self
                .as_any()
                .downcast_ref::<BooleanArray>()
                .vortex_expect("BooleanArray downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Int8 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Int8Type>>()
                .vortex_expect("Int8Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Int16 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Int16Type>>()
                .vortex_expect("Int16Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Int32 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Int32Type>>()
                .vortex_expect("Int32Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Int64 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Int64Type>>()
                .vortex_expect("Int64Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::UInt8 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<UInt8Type>>()
                .vortex_expect("UInt8Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::UInt16 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<UInt16Type>>()
                .vortex_expect("UInt16Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::UInt32 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<UInt32Type>>()
                .vortex_expect("UInt32Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::UInt64 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<UInt64Type>>()
                .vortex_expect("UInt64Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Float16 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Float16Type>>()
                .vortex_expect("Float16Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Float32 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Float32Type>>()
                .vortex_expect("Float32Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Float64 => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Float64Type>>()
                .vortex_expect("Float64Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Timestamp(..)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Duration(_)
            | DataType::Interval(_) => {
                vortex_bail!("Temporal types not yet supported: {}", self.data_type())
            }
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => {
                vortex_bail!("Binary types not yet supported: {}", self.data_type())
            }
            DataType::BinaryView => {
                vortex_bail!("BinaryView not yet supported: {}", self.data_type())
            }
            DataType::Utf8 | DataType::LargeUtf8 => {
                vortex_bail!("Utf8/LargeUtf8 not yet supported: {}", self.data_type())
            }
            DataType::Utf8View => self
                .as_any()
                .downcast_ref::<GenericByteViewArray<StringViewType>>()
                .vortex_expect("StringViewArray downcast")
                .into_vector()
                .map(Vector::from),
            DataType::List(_)
            | DataType::ListView(_)
            | DataType::LargeList(_)
            | DataType::LargeListView(_) => {
                vortex_bail!("List types not yet supported: {}", self.data_type())
            }
            DataType::FixedSizeList(..) => self
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .vortex_expect("FixedSizeListArray downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Struct(_) => self
                .as_any()
                .downcast_ref::<StructArray>()
                .vortex_expect("StructArray downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Union(..) => {
                vortex_bail!("Union type not supported: {}", self.data_type())
            }
            DataType::Dictionary(..) => {
                vortex_bail!("Dictionary type not supported: {}", self.data_type())
            }
            DataType::Decimal32(..) => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Decimal32Type>>()
                .vortex_expect("Decimal32Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Decimal64(..) => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Decimal64Type>>()
                .vortex_expect("Decimal64Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Decimal128(..) => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Decimal128Type>>()
                .vortex_expect("Decimal128Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Decimal256(..) => self
                .as_any()
                .downcast_ref::<PrimitiveArray<Decimal256Type>>()
                .vortex_expect("Decimal256Array downcast")
                .into_vector()
                .map(Vector::from),
            DataType::Map(..) => {
                vortex_bail!("Map type not supported: {}", self.data_type())
            }
            DataType::RunEndEncoded(..) => {
                vortex_bail!("RunEndEncoded type not supported: {}", self.data_type())
            }
        }
    }
}

/// Converts an Arrow [`NullBuffer`](arrow_buffer::NullBuffer) to a Vortex [`Mask`](vortex_mask::Mask).
pub(crate) fn nulls_to_mask(
    nulls: Option<&arrow_buffer::NullBuffer>,
    len: usize,
) -> vortex_mask::Mask {
    use vortex_buffer::BitBuffer;
    use vortex_mask::Mask;

    match nulls {
        None => Mask::AllTrue(len),
        Some(nulls) => {
            let inner = nulls.inner();
            // Arrow stores validity as "1 = valid, 0 = null" which matches our Mask semantics
            let bit_buffer = BitBuffer::from(inner.clone());
            Mask::from_buffer(bit_buffer)
        }
    }
}
