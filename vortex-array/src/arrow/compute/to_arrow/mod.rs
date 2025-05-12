mod canonical;
mod temporal;
mod varbin;

use std::any::Any;
use std::sync::LazyLock;

use arcref::ArcRef;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::arrow::array::{ArrowArray, ArrowVTable};
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Options, Output};
use crate::vtable::VTable;
use crate::{Array, ArrayExt};

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
    to_arrow_opts(
        array,
        &ToArrowOptions {
            arrow_type: Some(arrow_type.clone()),
        },
    )
}

pub fn to_arrow_opts(array: &dyn Array, options: &ToArrowOptions) -> VortexResult<ArrowArrayRef> {
    let arrow = TO_ARROW_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options,
        })?
        .unwrap_array()?
        .as_opt::<ArrowVTable>()
        .ok_or_else(|| vortex_err!("ToArrow compute kernels must return a Vortex ArrowArray"))?
        .inner()
        .clone();

    if let Some(arrow_type) = &options.arrow_type {
        if arrow.data_type() != arrow_type {
            vortex_bail!(
                "Arrow array type mismatch: expected {:?}, got {:?}",
                &options.arrow_type,
                arrow.data_type()
            );
        }
    }

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

struct ToArrow;

impl ComputeFnVTable for ToArrow {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let ToArrowArgs { array, arrow_type } = ToArrowArgs::try_from(args)?;

        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = array.invoke(&TO_ARROW_FN, args)? {
            return Ok(output);
        }

        // Fall back to canonicalizing and then converting.
        if !array.is_canonical() {
            let canonical_array = array.to_canonical()?;
            let arrow_array = to_arrow_opts(
                canonical_array.as_ref(),
                &ToArrowOptions {
                    arrow_type: arrow_type.cloned(),
                },
            )?;
            return Ok(ArrowArray::new(arrow_array, array.dtype().nullability())
                .to_array()
                .into());
        }

        vortex_bail!(
            "Failed to convert array {} to Arrow {:?}",
            array.encoding_id(),
            arrow_type
        );
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let ToArrowArgs { array, arrow_type } = ToArrowArgs::try_from(args)?;
        Ok(arrow_type
            .map(|arrow_type| DType::from_arrow((arrow_type, array.dtype().nullability())))
            .unwrap_or_else(|| array.dtype().clone()))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let ToArrowArgs { array, .. } = ToArrowArgs::try_from(args)?;
        Ok(array.len())
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

pub static TO_ARROW_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("to_arrow".into(), ArcRef::new_ref(&ToArrow));

    // Register the kernels we ship ourselves
    compute.register_kernel(ArcRef::new_ref(&canonical::ToArrowCanonical));
    compute.register_kernel(ArcRef::new_ref(&temporal::ToArrowTemporal));

    for kernel in inventory::iter::<ToArrowKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub struct ToArrowArgs<'a> {
    array: &'a dyn Array,
    arrow_type: Option<&'a DataType>,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for ToArrowArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 1 {
            vortex_bail!("Expected 1 input, found {}", value.inputs.len());
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 0 to be an array"))?;
        let options = value
            .options
            .as_any()
            .downcast_ref::<ToArrowOptions>()
            .vortex_expect("Expected options to be ToArrowOptions");

        Ok(ToArrowArgs {
            array,
            arrow_type: options.arrow_type.as_ref(),
        })
    }
}

pub struct ToArrowKernelRef(pub ArcRef<dyn Kernel>);
inventory::collect!(ToArrowKernelRef);

pub trait ToArrowKernel: VTable {
    fn to_arrow(
        &self,
        arr: &Self::Array,
        arrow_type: Option<&DataType>,
    ) -> VortexResult<Option<ArrowArrayRef>>;
}

#[derive(Debug)]
pub struct ToArrowKernelAdapter<V: VTable>(pub V);

impl<V: VTable + ToArrowKernel> ToArrowKernelAdapter<V> {
    pub const fn lift(&'static self) -> ToArrowKernelRef {
        ToArrowKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + ToArrowKernel> Kernel for ToArrowKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = ToArrowArgs::try_from(args)?;
        let Some(array) = inputs.array.as_opt::<V>() else {
            return Ok(None);
        };

        let Some(arrow_array) = V::to_arrow(&self.0, array, inputs.arrow_type)? else {
            return Ok(None);
        };

        Ok(Some(
            ArrowArray::new(arrow_array, array.dtype().nullability())
                .to_array()
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, PrimitiveArray, StringViewArray, StructArray};
    use arrow_buffer::NullBuffer;

    use super::to_arrow;
    use crate::{IntoArray, arrays};

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
