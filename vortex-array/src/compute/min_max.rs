use std::sync::{Arc, LazyLock};

use arcref::ArcRef;
use vortex_dtype::{DType, Nullability, StructDType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output, UnaryArgs};
use crate::stats::{Precision, Stat, StatsProviderExt};
use crate::{Array, Encoding};

/// Computes the min & max of an array, returning the (min, max) values
/// The return values are (min, max) scalars, where None indicates that the value is non-existent
/// (e.g. for an empty array).
/// The return value dtype is the non-nullable version of the array dtype.
///
/// This will update the stats set of this array (as a side effect).
pub fn min_max(array: &dyn Array) -> VortexResult<Option<MinMaxResult>> {
    let scalar = MIN_MAX_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options: &(),
        })?
        .unwrap_scalar()?;
    MinMaxResult::from_scalar(scalar)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinMaxResult {
    pub min: Scalar,
    pub max: Scalar,
}

impl MinMaxResult {
    pub fn from_scalar(scalar: Scalar) -> VortexResult<Option<Self>> {
        if scalar.is_null() {
            Ok(None)
        } else {
            let min = scalar.as_struct().field_by_idx(0)?;
            let max = scalar.as_struct().field_by_idx(1)?;
            Ok(Some(MinMaxResult { min, max }))
        }
    }
}

pub struct MinMax;

impl ComputeFnVTable for MinMax {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;

        let return_dtype = self.return_dtype(args)?;

        match min_max_impl(array, kernels)? {
            None => Ok(Scalar::null(return_dtype).into()),
            Some(MinMaxResult { min, max }) => {
                assert!(
                    min <= max,
                    "min > max: min={} max={} encoding={}",
                    min,
                    max,
                    array.encoding()
                );

                // Update the stats set with the computed min/max
                array
                    .statistics()
                    .set(Stat::Min, Precision::Exact(min.value().clone()));
                array
                    .statistics()
                    .set(Stat::Max, Precision::Exact(max.value().clone()));

                // Return the min/max as a struct scalar
                Ok(Scalar::struct_(return_dtype, vec![min, max]).into())
            }
        }
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;

        // We return a min/max struct scalar, where the overall struct is nullable in the case
        // that the array is all null or empty.
        Ok(DType::Struct(
            Arc::new(StructDType::new(
                ["min".into(), "max".into()].into(),
                vec![array.dtype().clone(), array.dtype().clone()],
            )),
            Nullability::Nullable,
        ))
    }

    fn return_len(&self, _args: &InvocationArgs) -> VortexResult<usize> {
        Ok(1)
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

fn min_max_impl(
    array: &dyn Array,
    kernels: &[ArcRef<dyn Kernel>],
) -> VortexResult<Option<MinMaxResult>> {
    if array.is_empty() || array.valid_count()? == 0 {
        return Ok(None);
    }

    let min = array
        .statistics()
        .get_scalar(Stat::Min, array.dtype())
        .and_then(Precision::as_exact);
    let max = array
        .statistics()
        .get_scalar(Stat::Max, array.dtype())
        .and_then(Precision::as_exact);

    if let Some((min, max)) = min.zip(max) {
        return Ok(Some(MinMaxResult { min, max }));
    }

    let args = InvocationArgs {
        inputs: &[array.into()],
        options: &(),
    };
    for kernel in kernels {
        if let Some(output) = kernel.invoke(&args)? {
            return MinMaxResult::from_scalar(output.unwrap_scalar()?);
        }
    }
    if let Some(output) = array.invoke(&MIN_MAX_FN, &args)? {
        return MinMaxResult::from_scalar(output.unwrap_scalar()?);
    }

    if !array.is_canonical() {
        let array = array.to_canonical()?;
        return min_max(array.as_ref());
    }

    vortex_bail!(NotImplemented: "min_max", array.encoding());
}

/// Computes the min and max of an array, returning the (min, max) values
/// If the array is empty or has only nulls, the result is `None`.
pub trait MinMaxKernel: Encoding {
    fn min_max(&self, array: &Self::Array) -> VortexResult<Option<MinMaxResult>>;
}

pub struct MinMaxKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(MinMaxKernelRef);

#[derive(Debug)]
pub struct MinMaxKernelAdapter<E: Encoding>(pub E);

impl<E: Encoding + MinMaxKernel> MinMaxKernelAdapter<E> {
    pub const fn lift(&'static self) -> MinMaxKernelRef {
        MinMaxKernelRef(ArcRef::new_ref(self))
    }
}

impl<E: Encoding + MinMaxKernel> Kernel for MinMaxKernelAdapter<E> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = UnaryArgs::<()>::try_from(args)?;
        let Some(array) = inputs.array.as_any().downcast_ref::<E::Array>() else {
            return Ok(None);
        };
        let dtype = DType::Struct(
            Arc::new(StructDType::new(
                ["min".into(), "max".into()].into(),
                vec![array.dtype().clone(), array.dtype().clone()],
            )),
            Nullability::Nullable,
        );
        Ok(Some(match E::min_max(&self.0, array)? {
            None => Scalar::null(dtype).into(),
            Some(MinMaxResult { min, max }) => Scalar::struct_(dtype, vec![min, max]).into(),
        }))
    }
}

pub static MIN_MAX_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("min_max".into(), ArcRef::new_ref(&MinMax));
    for kernel in inventory::iter::<MinMaxKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use vortex_buffer::buffer;

    use crate::arrays::{BoolArray, NullArray, PrimitiveArray};
    use crate::compute::{MinMaxResult, min_max};
    use crate::validity::Validity;

    #[test]
    fn test_prim_max() {
        let p = PrimitiveArray::new(buffer![1, 2, 3], Validity::NonNullable);
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: 1.into(),
                max: 3.into()
            })
        );
    }

    #[test]
    fn test_bool_max() {
        let p = BoolArray::new(
            BooleanBuffer::from([true, true, true].as_slice()),
            Validity::NonNullable,
        );
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: true.into(),
                max: true.into()
            })
        );

        let p = BoolArray::new(
            BooleanBuffer::from([false, false, false].as_slice()),
            Validity::NonNullable,
        );
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: false.into(),
                max: false.into()
            })
        );

        let p = BoolArray::new(
            BooleanBuffer::from([false, true, false].as_slice()),
            Validity::NonNullable,
        );
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: false.into(),
                max: true.into()
            })
        );
    }

    #[test]
    fn test_null() {
        let p = NullArray::new(1);
        assert_eq!(min_max(&p).unwrap(), None);
    }
}
