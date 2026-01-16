// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod canonical;
mod list;
mod temporal;
mod varbin;

use std::any::Any;
use std::sync::LazyLock;

use arcref::ArcRef;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
pub(crate) use canonical::to_arrow_decimal32;
pub(crate) use canonical::to_arrow_decimal64;
pub(crate) use canonical::to_arrow_decimal128;
pub(crate) use canonical::to_arrow_decimal256;
use vortex_dtype::DType;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::Array;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::array::ArrowArray;
use crate::compute::ComputeFn;
use crate::compute::ComputeFnVTable;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Options;
use crate::compute::Output;
use crate::vtable::VTable;

/// Convert a Vortex array to an Arrow array with the encoding's preferred `DataType`.
///
/// For example, a `VarBinArray` will be converted to an Arrow `VarBin` array, instead of the
/// canonical `VarBinViewArray`.
///
/// Warning: do not use this to convert a Vortex [`crate::stream::ArrayStream`] since each array
/// may have a different preferred Arrow type. Use [`to_arrow`] instead.
pub fn to_arrow_preferred(array: &dyn Array) -> VortexResult<ArrowArrayRef> {
    to_arrow_opts(array, &ToArrowOptions { arrow_type: None })
}

/// Convert a Vortex array to an Arrow array of the given type.
pub fn to_arrow(array: &dyn Array, arrow_type: &DataType) -> VortexResult<ArrowArrayRef> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    array.to_array().execute_arrow(arrow_type, &mut ctx)
}

pub fn to_arrow_opts(array: &dyn Array, options: &ToArrowOptions) -> VortexResult<ArrowArrayRef> {
    let data_type = if let Some(data_type) = &options.arrow_type {
        data_type.clone()
    } else {
        array.dtype().to_arrow_dtype()?
    };
    let arrow = to_arrow(array, &data_type)?;

    vortex_ensure!(
        &data_type == arrow.data_type(),
        "to arrow returned array with data_type {}, expected {}",
        arrow.data_type(),
        data_type
    );

    Ok(arrow)
}

pub struct ToArrowOptions {
    /// The Arrow data type to convert to, if specified.
    pub arrow_type: Option<DataType>,
}

impl Options for ToArrowOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

//
// pub struct ToArrowArgs<'a> {
//     array: &'a dyn Array,
//     arrow_type: Option<&'a DataType>,
// }
//
// impl<'a> TryFrom<&InvocationArgs<'a>> for ToArrowArgs<'a> {
//     type Error = VortexError;
//
//     fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
//         if value.inputs.len() != 1 {
//             vortex_bail!("Expected 1 input, found {}", value.inputs.len());
//         }
//         let array = value.inputs[0]
//             .array()
//             .ok_or_else(|| vortex_err!("Expected input 0 to be an array"))?;
//         let options = value
//             .options
//             .as_any()
//             .downcast_ref::<ToArrowOptions>()
//             .vortex_expect("Expected options to be ToArrowOptions");
//
//         Ok(ToArrowArgs {
//             array,
//             arrow_type: options.arrow_type.as_ref(),
//         })
//     }
// }
//
// pub struct ToArrowKernelRef(pub ArcRef<dyn Kernel>);
// inventory::collect!(ToArrowKernelRef);
//
// pub trait ToArrowKernel: VTable {
//     fn to_arrow(
//         &self,
//         arr: &Self::Array,
//         arrow_type: Option<&DataType>,
//     ) -> VortexResult<Option<ArrowArrayRef>>;
// }
//
// #[derive(Debug)]
// pub struct ToArrowKernelAdapter<V: VTable>(pub V);
//
// impl<V: VTable + ToArrowKernel> ToArrowKernelAdapter<V> {
//     pub const fn lift(&'static self) -> ToArrowKernelRef {
//         ToArrowKernelRef(ArcRef::new_ref(self))
//     }
// }
//
// impl<V: VTable + ToArrowKernel> Kernel for ToArrowKernelAdapter<V> {
//     fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
//         let inputs = ToArrowArgs::try_from(args)?;
//         let Some(array) = inputs.array.as_opt::<V>() else {
//             return Ok(None);
//         };
//
//         let Some(arrow_array) = V::to_arrow(&self.0, array, inputs.arrow_type)? else {
//             return Ok(None);
//         };
//
//         Ok(Some(
//             ArrowArray::new(arrow_array, array.dtype().nullability())
//                 .to_array()
//                 .into(),
//         ))
//     }
// }

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::ArrayRef;
    use arrow_array::PrimitiveArray;
    use arrow_array::StringViewArray;
    use arrow_array::StructArray;
    use arrow_array::types::Int32Type;
    use arrow_buffer::NullBuffer;

    use super::to_arrow;
    use crate::IntoArray;
    use crate::arrays;

    #[test]
    fn test_to_arrow() {
        let array = arrays::StructArray::from_fields(
            vec![
                (
                    "a",
                    arrays::PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2)])
                        .into_array(),
                ),
                (
                    "b",
                    arrays::VarBinViewArray::from_iter_str(vec!["a", "b", "c"]).into_array(),
                ),
            ]
            .as_slice(),
        )
        .unwrap();

        let arrow_array: ArrayRef = Arc::new(
            StructArray::try_from(vec![
                (
                    "a",
                    Arc::new(PrimitiveArray::<Int32Type>::from_iter_values_with_nulls(
                        vec![1, 0, 2],
                        Some(NullBuffer::from(vec![true, false, true])),
                    )) as ArrayRef,
                ),
                (
                    "b",
                    Arc::new(StringViewArray::from(vec![Some("a"), Some("b"), Some("c")])),
                ),
            ])
            .unwrap(),
        );

        assert_eq!(
            &to_arrow(array.as_ref(), &array.dtype().to_arrow_dtype().unwrap()).unwrap(),
            &arrow_array
        );
    }
}
