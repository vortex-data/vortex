// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::ArrowPrimitiveType;
use arrow_array::BooleanArray;
use arrow_array::NullArray;
use arrow_array::PrimitiveArray;
use arrow_array::RecordBatch;
use arrow_array::StructArray;
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use vortex_compute::arrow::IntoArrow;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_vector::VectorOps;

use crate::Array;
use crate::ArrayRef;
use crate::VectorExecutor;
use crate::arrays::ListVTable;
use crate::arrays::VarBinVTable;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;

/// Trait for executing a Vortex array to produce an Arrow array.
pub trait ArrowArrayExecutor {
    /// Execute the array to produce an Arrow array.
    ///
    /// If a [`DataType`] is given, the array will be converted to the desired Arrow type.
    fn execute_arrow(
        &self,
        // TODO(ngates): do we even want optional data type? Or do we make it required and tell
        //  users to call `DType::into_arrow` to get a default logical type? I'm inclined to think
        //  the latter is preferable. Although there's a world where the user may want the minimal
        //  conversion to Arrow without knowing what that conversion is. In which case, should
        //  DictionaryArray and non-logical types be supported? Should the user provide a list of
        //  supported Arrow arrays? I dunno...
        field: Option<&Field>,
        session: &VortexSession,
    ) -> VortexResult<ArrowArrayRef>;

    fn execute_record_batch(
        &self,
        schema: &Schema,
        session: &VortexSession,
    ) -> VortexResult<RecordBatch> {
        let array = self.execute_arrow(
            // TODO(ngates): don't really want to create a new field each time?
            Some(&Field::new(
                "",
                DataType::Struct(schema.fields.clone()),
                false,
            )),
            session,
        )?;
        RecordBatch::try_from(array.as_struct()).map_err(VortexError::from)
    }
}

impl ArrowArrayExecutor for ArrayRef {
    fn execute_arrow(
        &self,
        field: Option<&Field>,
        session: &VortexSession,
    ) -> VortexResult<ArrowArrayRef> {
        match field {
            None => {
                // Special-case the Arrow-shaped encodings that are not part of our vector API
                // to avoid unnecessary conversions.
                if let Some(_varbin) = self.as_opt::<VarBinVTable>() {
                    // Convert directly to preferred Arrow VarBin array.
                }
                if let Some(_list) = self.as_opt::<ListVTable>() {
                    // Convert directly to preferred Arrow List array.
                }

                let vector = self.execute_vector(session)?;
                vector.into_arrow()
            }

            // Once we know the target Arrow DataType, how do we get there? Should we allow crates
            // to register Arrow conversion kernels? Should we wrap up the Vortex array in a
            // cast expression and re-run the optimizer? Should we just execute to a vector and
            // then convert?
            Some(field) => {
                let nullability: Nullability = field.is_nullable().into();
                match field.data_type() {
                    DataType::Null => to_arrow_null(self, session),
                    DataType::Boolean => to_arrow_bool(self, nullability, session),
                    DataType::Int8 => to_arrow_primitive::<UInt8Type>(self, nullability, session),
                    DataType::Int16 => to_arrow_primitive::<UInt16Type>(self, nullability, session),
                    DataType::Int32 => to_arrow_primitive::<UInt32Type>(self, nullability, session),
                    DataType::Int64 => to_arrow_primitive::<UInt64Type>(self, nullability, session),
                    DataType::UInt8 => to_arrow_primitive::<Int8Type>(self, nullability, session),
                    DataType::UInt16 => to_arrow_primitive::<Int16Type>(self, nullability, session),
                    DataType::UInt32 => to_arrow_primitive::<Int32Type>(self, nullability, session),
                    DataType::UInt64 => to_arrow_primitive::<Int64Type>(self, nullability, session),
                    DataType::Float16 => {
                        to_arrow_primitive::<Float16Type>(self, nullability, session)
                    }
                    DataType::Float32 => {
                        to_arrow_primitive::<Float32Type>(self, nullability, session)
                    }
                    DataType::Float64 => {
                        to_arrow_primitive::<Float64Type>(self, nullability, session)
                    }
                    DataType::Timestamp(..) => {
                        todo!()
                    }
                    DataType::Date32 => {
                        todo!()
                    }
                    DataType::Date64 => {
                        todo!()
                    }
                    DataType::Time32(_) => {
                        todo!()
                    }
                    DataType::Time64(_) => {
                        todo!()
                    }
                    DataType::Duration(_) => {
                        todo!()
                    }
                    DataType::Interval(_) => {
                        todo!()
                    }
                    DataType::Binary => {
                        todo!()
                    }
                    DataType::FixedSizeBinary(_) => {
                        todo!()
                    }
                    DataType::LargeBinary => {
                        todo!()
                    }
                    DataType::BinaryView => {
                        todo!()
                    }
                    DataType::Utf8 => {
                        todo!()
                    }
                    DataType::LargeUtf8 => {
                        todo!()
                    }
                    DataType::Utf8View => {
                        todo!()
                    }
                    DataType::List(_) => {
                        todo!()
                    }
                    DataType::ListView(_) => {
                        todo!()
                    }
                    DataType::FixedSizeList(..) => {
                        todo!()
                    }
                    DataType::LargeList(_) => {
                        todo!()
                    }
                    DataType::LargeListView(_) => {
                        todo!()
                    }
                    DataType::Struct(fields) => to_arrow_struct(self, fields, nullability, session),
                    DataType::Union(..) => {
                        todo!()
                    }
                    DataType::Dictionary(..) => {
                        todo!()
                    }
                    DataType::Decimal32(..) => {
                        todo!()
                    }
                    DataType::Decimal64(..) => {
                        todo!()
                    }
                    DataType::Decimal128(..) => {
                        todo!()
                    }
                    DataType::Decimal256(..) => {
                        todo!()
                    }
                    DataType::Map(..) => {
                        todo!()
                    }
                    DataType::RunEndEncoded(..) => {
                        todo!()
                    }
                }
            }
        }
    }
}

fn to_arrow_null(array: &ArrayRef, session: &VortexSession) -> VortexResult<ArrowArrayRef> {
    let null_vector = array
        .execute_vector(session)?
        .into_null_opt()
        .ok_or_else(|| vortex_err!("Failed to convert array to Null vector"))?;
    Ok(Arc::new(NullArray::new(null_vector.len())))
}

fn to_arrow_bool(
    array: &ArrayRef,
    nullability: Nullability,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    let bool_vector = array
        .execute_vector(session)?
        .into_bool_opt()
        .ok_or_else(|| vortex_err!("Failed to convert array to Bool vector"))?;
    let (bits, validity) = bool_vector.into_parts();

    vortex_ensure!(
        nullability.is_nullable() || validity.all_true(),
        "Cannot convert to non-nullable Boolean array with nulls present"
    );

    Ok(Arc::new(BooleanArray::new(
        bits.into(),
        to_null_buffer(validity),
    )))
}

fn to_arrow_primitive<T: ArrowPrimitiveType>(
    array: &ArrayRef,
    nullability: Nullability,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    let array = array.cast(DType::Primitive(T::Native::PTYPE, nullability))?;
    let vector = array.execute_vector(session)?.into_primitive();
    let (buffer, validity) = vector.downcast::<T::Native>().into_parts();
    let null_buffer = to_null_buffer(validity);
    let buffer = buffer.into_arrow_scalar_buffer();
    Ok(Arc::new(PrimitiveArray::<T>::new(buffer, null_buffer)))
}

fn to_arrow_struct(
    array: &ArrayRef,
    fields: &Fields,
    nullability: Nullability,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    let validity = array.validity_mask();
    vortex_ensure!(
        nullability.is_nullable() || validity.all_true(),
        "Cannot convert to non-nullable Struct array with nulls present"
    );

    let mut arrow_fields = Vec::with_capacity(fields.len());
    for field in fields.iter() {
        arrow_fields.push(
            array
                .get_item(field.name().as_str())?
                .execute_arrow(Some(field), session)?,
        );
    }

    Ok(Arc::new(unsafe {
        StructArray::new_unchecked(
            fields.clone(),
            arrow_fields.into(),
            to_null_buffer(validity),
        )
    }))
}
