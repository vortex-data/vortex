// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray as _;
use crate::arrays::ConstantVTable;
use crate::compute::ComputeFn;
use crate::compute::ComputeFnVTable;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Output;
use crate::compute::UnaryArgs;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::vtable::VTable;

static NAMES: LazyLock<FieldNames> = LazyLock::new(|| FieldNames::from(["min", "max"]));

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
pub fn min_max(array: &ArrayRef) -> VortexResult<Option<MinMaxResult>> {
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
        let array = array.to_array();

        let return_dtype = self.return_dtype(args)?;

        match min_max_impl(&array, kernels)? {
            None => Ok(Scalar::null(return_dtype).into()),
            Some(MinMaxResult { min, max }) => {
                assert!(
                    min <= max,
                    "min > max: min={} max={} encoding={}",
                    min,
                    max,
                    array.encoding_id()
                );

                // Update the stats set with the computed min/max.
                if let Some(min_value) = min.value() {
                    array
                        .statistics()
                        .set(Stat::Min, Precision::Exact(min_value.clone()));
                }
                if let Some(max_value) = max.value() {
                    array
                        .statistics()
                        .set(Stat::Max, Precision::Exact(max_value.clone()));
                }

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
                NAMES.clone(),
                vec![
                    array.dtype().as_nonnullable(),
                    array.dtype().as_nonnullable(),
                ],
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
    array: &ArrayRef,
    kernels: &[ArcRef<dyn Kernel>],
) -> VortexResult<Option<MinMaxResult>> {
    if array.is_empty() || array.valid_count()? == 0 {
        return Ok(None);
    }

    if let Some(array) = array.as_opt::<ConstantVTable>() {
        return ConstantVTable.min_max(array);
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
        let non_nullable_dtype = array.dtype().as_nonnullable();
        return Ok(Some(MinMaxResult {
            min: min.cast(&non_nullable_dtype)?,
            max: max.cast(&non_nullable_dtype)?,
        }));
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

    if !array.is_canonical() {
        let array = array.to_canonical()?.into_array();
        return min_max(&array);
    }

    vortex_bail!(NotImplemented: "min_max", array.encoding_id());
}

/// The minimum and maximum non-null values of an array, or None if there are no non-null/or non-nan
/// values.
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
        let non_nullable_dtype = array.dtype().as_nonnullable();
        let dtype = DType::Struct(
            StructFields::new(
                NAMES.clone(),
                vec![non_nullable_dtype.clone(), non_nullable_dtype],
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
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;

    use crate::IntoArray as _;
    use crate::arrays::BoolArray;
    use crate::arrays::NullArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::MinMaxResult;
    use crate::compute::min_max;
    use crate::validity::Validity;

    #[test]
    fn test_prim_max() {
        let p = PrimitiveArray::new(buffer![1, 2, 3], Validity::NonNullable).into_array();
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
            BitBuffer::from([true, true, true].as_slice()),
            Validity::NonNullable,
        )
        .into_array();
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: true.into(),
                max: true.into()
            })
        );

        let p = BoolArray::new(
            BitBuffer::from([false, false, false].as_slice()),
            Validity::NonNullable,
        )
        .into_array();
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: false.into(),
                max: false.into()
            })
        );

        let p = BoolArray::new(
            BitBuffer::from([false, true, false].as_slice()),
            Validity::NonNullable,
        )
        .into_array();
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
        let p = NullArray::new(1).into_array();
        assert_eq!(min_max(&p).unwrap(), None);
    }
}
