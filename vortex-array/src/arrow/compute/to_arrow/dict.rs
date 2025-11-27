// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::make_array;
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::Array;
use crate::arrays::DictArray;
use crate::arrays::DictVTable;
use crate::arrow::IntoArrowArray;
use crate::arrow::compute::ToArrowKernel;
use crate::arrow::compute::ToArrowKernelAdapter;
use crate::register_kernel;

register_kernel!(ToArrowKernelAdapter(DictVTable).lift());

impl ToArrowKernel for DictVTable {
    fn to_arrow(
        &self,
        array: &DictArray,
        arrow_type: Option<&DataType>,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        let (arrow_keys, arrow_values) = match arrow_type {
            None => (
                array.codes().clone().into_arrow_preferred()?,
                match array.values().dtype() {
                    DType::Utf8(_) => {
                        // If the values are Utf8, we force conversion into
                        // arrow Utf8 rather than Utf8View, since this would
                        // effectively double-dictionary encode otherwise.
                        array.values().clone().into_arrow(&DataType::Utf8)?
                    }
                    _ => array.values().clone().into_arrow_preferred()?,
                },
            ),
            Some(DataType::Dictionary(codes_type, values_type)) => (
                array.codes().clone().into_arrow(codes_type)?,
                array.values().clone().into_arrow(values_type)?,
            ),
            _ => {
                // Unsupported type.
                return Ok(None);
            }
        };
        let keys_data = arrow_keys.to_data();
        Ok(Some(make_array(
            ArrayDataBuilder::new(DataType::Dictionary(
                Box::new(arrow_keys.data_type().clone()),
                Box::new(arrow_values.data_type().clone()),
            ))
            .len(keys_data.len())
            .add_buffers(keys_data.buffers().iter().cloned())
            .nulls(keys_data.nulls().cloned())
            .add_child_data(arrow_values.to_data())
            .build()?,
        )))
    }
}
