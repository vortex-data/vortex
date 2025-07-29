// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Array;
use crate::compute::{
    ComputeFn, ComputeFnVTable, IS_SORTED_FN, InvocationArgs, IsSortedKernelRef, Kernel, Options,
    Output, is_sorted_opts,
};
use crate::stats::{Precision, Stat, StatsProvider};
use crate::vtable::VTable;
use arcref::ArcRef;
use std::any::Any;
use std::sync::LazyLock;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

pub fn array_equals(array: &dyn Array) -> VortexResult<bool> {
    is_sorted_opts(array, false)
}

pub fn array_equals_opts(array: &dyn Array, ignore_nullability: bool) -> VortexResult<bool> {
    Ok(IS_SORTED_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options: &ArrayEqualsOptions { ignore_nullability },
        })?
        .unwrap_scalar()?
        .as_bool()
        .value()
        .vortex_expect("non-nullable"))
}

#[derive(Clone, Copy)]
struct ArrayEqualsOptions {
    ignore_nullability: bool,
}

impl Options for ArrayEqualsOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub static ARRAY_EQUALS_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("array_equals".into(), ArcRef::new_ref(&ArrayEquals));
    for kernel in inventory::iter::<IsSortedKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct ArrayEquals;
impl ComputeFnVTable for ArrayEquals {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let ArrayEqualsArgs {
            left,
            right,
            ignore_nullability,
        } = ArrayEqualsArgs::try_from(args)?;

        if ignore_nullability && !left.dtype().eq_ignore_nullability(right.dtype()) {
            return Ok(Scalar::from(false).into());
        }

        if !ignore_nullability && !left.dtype().eq(right.dtype()) {
            return Ok(Scalar::from(false).into());
        }

        if left.len() != right.len() {
            return Ok(Scalar::from(false).into());
        }

        if let Some(l_scalar) = left.as_constant()
            && let Some(r_scalar) = right.as_constant()
        {
            return Ok(Scalar::from(l_scalar.eq(&r_scalar)).into());
        }

        if left.is_empty() && right.is_empty() {
            return Ok(Scalar::from(true).into());
        }

        for stat in [
            Stat::IsConstant,
            Stat::IsSorted,
            Stat::IsStrictSorted,
            Stat::Max, // todo: can we do that with e.g. float errors?
            Stat::Min,
            Stat::Sum,
            Stat::NullCount,
            Stat::NaNCount,
            // No Stat::UncompressedSizeInBytes because arrays may physically differ and has a different metric
        ] {
            let Some(Precision::Exact(left_v)) = left.statistics().get(stat) else {
                continue;
            };

            let Some(Precision::Exact(right_v)) = right.statistics().get(stat) else {
                continue;
            };

            if !left_v.eq(&right_v) {
                return Ok(Scalar::from(false).into());
            }
        }

        let args = InvocationArgs {
            inputs: &[left.into(), right.into()],
            options: &ArrayEqualsOptions { ignore_nullability },
        };

        for kernel in kernels {
            if let Some(output) = kernel.invoke(&args)? {
                return Ok(output);
            }
        }

        if let Some(output) = left.invoke(&ARRAY_EQUALS_FN, &args)? {
            todo!();
        }
        // swap...

        todo!();

        // if no kernels matched, default running per element comparison
        todo!();
    }

    fn return_dtype(&self, _args: &InvocationArgs) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn return_len(&self, _args: &InvocationArgs) -> VortexResult<usize> {
        Ok(1)
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

// todo: statistics
pub trait ArrayEqualsKernel: VTable {
    fn compare_array(
        &self,
        array: &Self::Array,
        other: &dyn Array,
        ignore_nullability: bool,
    ) -> VortexResult<Option<bool>>;
}

struct ArrayEqualsArgs<'a> {
    left: &'a dyn Array,
    right: &'a dyn Array,
    ignore_nullability: bool,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for ArrayEqualsArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 3 {
            vortex_bail!(
                "ArrayEquals function requires three one arguments, got {}",
                value.inputs.len()
            );
        }
        let left = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("First argument must be an array"))?;

        let right = value.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("Second argument must be an array"))?;

        let options = value
            .options
            .as_any()
            .downcast_ref::<ArrayEqualsOptions>()
            .ok_or_else(|| vortex_err!("Invalid options type for array equals function"))?;

        Ok(ArrayEqualsArgs {
            left,
            right,
            ignore_nullability: options.ignore_nullability,
        })
    }
}

#[derive(Debug)]
pub struct ArrayEqualsKernelAdapter<V: VTable>(pub V);

pub struct ArrayEqualsKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(ArrayEqualsKernelRef);

impl<V: VTable + ArrayEqualsKernel> ArrayEqualsKernelAdapter<V> {
    pub const fn lift(&'static self) -> ArrayEqualsKernelRef {
        ArrayEqualsKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + ArrayEqualsKernel> Kernel for ArrayEqualsKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let ArrayEqualsArgs {
            left,
            right,
            ignore_nullability,
        } = ArrayEqualsArgs::try_from(args)?;

        let Some(left) = left.as_opt::<V>() else {
            return Ok(None);
        };

        let is_equal = V::compare_array(&self.0, left, right, ignore_nullability)?;
        Ok(is_equal.map(|b| Scalar::from(b).into()))
    }
}
