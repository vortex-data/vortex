// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion logic between Vortex vector types and Arrow types.

use crate::Datum;
use crate::ScalarOps;
use crate::Vector;
use crate::VectorMutOps;
use crate::binaryview::BinaryViewVector;
use crate::binaryview::StringType;
use crate::bool::BoolVector;
use crate::decimal::DecimalVector;
use crate::fixed_size_list::FixedSizeListVector;
use crate::null::NullVector;
use crate::primitive::PrimitiveVector;
use crate::struct_::StructVector;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexError;
use vortex_error::vortex_bail;

mod binaryview;
mod bool;
mod decimal;
mod fixed_size_list;
mod list;
mod mask;
mod null;
mod primitive;
mod struct_;
mod vector;

impl TryFrom<Datum> for Box<dyn arrow_array::Datum> {
    type Error = VortexError;

    fn try_from(value: Datum) -> Result<Self, Self::Error> {
        match value {
            Datum::Scalar(s) => Ok(Box::new(arrow_array::Scalar::new(ArrayRef::try_from(
                s.repeat(1).freeze(),
            )?))),
            Datum::Vector(v) => Ok(Box::new(ArrayRef::try_from(v)?)),
        }
    }
}

impl TryFrom<&dyn Array> for Vector {
    type Error = VortexError;

    fn try_from(value: &dyn Array) -> Result<Self, Self::Error> {
        match value.data_type() {
            DataType::Null => NullVector::try_from(value).map(Vector::from),
            DataType::Boolean => BoolVector::try_from(value).map(Vector::from),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64 => PrimitiveVector::try_from(value).map(Vector::from),
            DataType::Timestamp(..)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Duration(_)
            | DataType::Interval(_) => {
                vortex_bail!("Temporal types not yet supported: {}", value.data_type())
            }
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => {
                vortex_bail!("Binary types not yet supported: {}", value.data_type())
            }
            DataType::BinaryView => {
                vortex_bail!("BinaryView not yet supported: {}", value.data_type())
            }
            DataType::Utf8 | DataType::LargeUtf8 => {
                vortex_bail!("Utf8/LargeUtf8 not yet supported: {}", value.data_type())
            }
            DataType::Utf8View => BinaryViewVector::<StringType>::try_from(value).map(Vector::from),
            DataType::List(_)
            | DataType::ListView(_)
            | DataType::LargeList(_)
            | DataType::LargeListView(_) => {
                vortex_bail!("List types not yet supported: {}", value.data_type())
            }
            DataType::FixedSizeList(..) => FixedSizeListVector::try_from(value).map(Vector::from),
            DataType::Struct(_) => StructVector::try_from(value).map(Vector::from),
            DataType::Union(..) => {
                vortex_bail!("Union type not supported: {}", value.data_type())
            }
            DataType::Dictionary(..) => {
                vortex_bail!("Dictionary type not supported: {}", value.data_type())
            }
            DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..) => DecimalVector::try_from(value).map(Vector::from),
            DataType::Map(..) => {
                vortex_bail!("Map type not supported: {}", value.data_type())
            }
            DataType::RunEndEncoded(..) => {
                vortex_bail!("RunEndEncoded type not supported: {}", value.data_type())
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
