// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::{DType, Nullability, StructFields};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::Array;
use crate::arrays::ConstantVTable;
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output, UnaryArgs};
use crate::stats::{Precision, Stat, StatsProvider};
use crate::vtable::VTable;

static MIN_MAX_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("min_max".into(), ArcRef::new_ref(&MinMax));
    for kernel in inventory::iter::<MinMaxKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub(crate) fn warm_up_vtable() -> usize {
    MIN_MAX_FN.kernels().len()
}

/// The minimum and maximum non-null values of an array, or None if there are no non-null values.
///
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
            let min = scalar
                .as_struct()
                .field_by_idx(0)
                .vortex_expect("missing min field");
            let max = scalar
                .as_struct()
                .field_by_idx(1)
                .vortex_expect("missing max field");
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
                    array.encoding_id()
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
            StructFields::new(
                ["min", "max"].into(),
                vec![array.dtype().clone(), array.dtype().clone()],
            ),
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
    if array.is_empty() || array.valid_count() == 0 {
        return Ok(None);
    }

    if let Some(array) = array.as_opt::<ConstantVTable>()
        && !array.scalar().is_null()
    {
        return Ok(Some(MinMaxResult {
            min: array.scalar().clone(),
            max: array.scalar().clone(),
        }));
    }

    let min = array
        .statistics()
        .get(Stat::Min)
        .and_then(Precision::as_exact);
    let max = array
        .statistics()
        .get(Stat::Max)
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
        let array = array.to_canonical();
        return min_max(array.as_ref());
    }

    vortex_bail!(NotImplemented: "min_max", array.encoding_id());
}

/// The minimum and maximum non-null values of an array, or None if there are no non-null values.
pub trait MinMaxKernel: VTable {
    fn min_max(&self, array: &Self::Array) -> VortexResult<Option<MinMaxResult>>;
}

pub struct MinMaxKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(MinMaxKernelRef);

#[derive(Debug)]
pub struct MinMaxKernelAdapter<V: VTable>(pub V);

impl<V: VTable + MinMaxKernel> MinMaxKernelAdapter<V> {
    pub const fn lift(&'static self) -> MinMaxKernelRef {
        MinMaxKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + MinMaxKernel> Kernel for MinMaxKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = UnaryArgs::<()>::try_from(args)?;
        let Some(array) = inputs.array.as_opt::<V>() else {
            return Ok(None);
        };
        let dtype = DType::Struct(
            StructFields::new(
                ["min", "max"].into(),
                vec![array.dtype().clone(), array.dtype().clone()],
            ),
            Nullability::Nullable,
        );
        Ok(Some(match V::min_max(&self.0, array)? {
            None => Scalar::null(dtype).into(),
            Some(MinMaxResult { min, max }) => Scalar::struct_(dtype, vec![min, max]).into(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{BitBuffer, buffer};

    use crate::arrays::{BoolArray, NullArray, PrimitiveArray};
    use crate::compute::{MinMaxResult, min_max};
    use crate::validity::Validity;

    #[test]
    fn test_prim_max() {
        let p = PrimitiveArray::new(buffer![1, 2, 3], Validity::NonNullable);
        assert_eq!(
            min_max(p.as_ref()).unwrap(),
            Some(MinMaxResult {
                min: 1.into(),
                max: 3.into()
            })
        );
    }

    #[test]
    fn test_bool_max() {
        let p = BoolArray::from_bit_buffer(
            BitBuffer::from([true, true, true].as_slice()),
            Validity::NonNullable,
        );
        assert_eq!(
            min_max(p.as_ref()).unwrap(),
            Some(MinMaxResult {
                min: true.into(),
                max: true.into()
            })
        );

        let p = BoolArray::from_bit_buffer(
            BitBuffer::from([false, false, false].as_slice()),
            Validity::NonNullable,
        );
        assert_eq!(
            min_max(p.as_ref()).unwrap(),
            Some(MinMaxResult {
                min: false.into(),
                max: false.into()
            })
        );

        let p = BoolArray::from_bit_buffer(
            BitBuffer::from([false, true, false].as_slice()),
            Validity::NonNullable,
        );
        assert_eq!(
            min_max(p.as_ref()).unwrap(),
            Some(MinMaxResult {
                min: false.into(),
                max: true.into()
            })
        );
    }

    #[test]
    fn test_null() {
        let p = NullArray::new(1);
        assert_eq!(min_max(p.as_ref()).unwrap(), None);
    }
}
