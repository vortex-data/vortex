use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output, cast};
use crate::vtable::VTable;
use crate::{Array, ArrayRef, IntoArray};

pub fn fill_null(array: &dyn Array, fill_value: &Scalar) -> VortexResult<ArrayRef> {
    FILL_NULL_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), fill_value.into()],
            options: &(),
        })?
        .unwrap_array()
}

pub trait FillNullKernel: VTable {
    fn fill_null(&self, array: &Self::Array, fill_value: &Scalar) -> VortexResult<ArrayRef>;
}

pub struct FillNullKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(FillNullKernelRef);

#[derive(Debug)]
pub struct FillNullKernelAdapter<V: VTable>(pub V);

impl<V: VTable + FillNullKernel> FillNullKernelAdapter<V> {
    pub const fn lift(&'static self) -> FillNullKernelRef {
        FillNullKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + FillNullKernel> Kernel for FillNullKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = FillNullArgs::try_from(args)?;
        let Some(array) = inputs.array.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(Some(
            V::fill_null(&self.0, array, inputs.fill_value)?.into(),
        ))
    }
}

pub static FILL_NULL_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("fill_null".into(), ArcRef::new_ref(&FillNull));
    for kernel in inventory::iter::<FillNullKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct FillNull;

impl ComputeFnVTable for FillNull {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let FillNullArgs { array, fill_value } = FillNullArgs::try_from(args)?;

        if !array.dtype().is_nullable() {
            return Ok(array.to_array().into());
        }

        if array.invalid_count()? == 0 {
            return Ok(cast(array, fill_value.dtype())?.into());
        }

        if fill_value.is_null() {
            vortex_bail!("Cannot fill_null with a null value")
        }

        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = array.invoke(&FILL_NULL_FN, args)? {
            return Ok(output);
        }

        log::debug!("FillNullFn not implemented for {}", array.encoding_id());
        if !array.is_canonical() {
            let canonical_arr = array.to_canonical()?.into_array();
            return Ok(fill_null(canonical_arr.as_ref(), fill_value)?.into());
        }

        vortex_bail!("fill null not implemented for DType {}", array.dtype())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let FillNullArgs { array, fill_value } = FillNullArgs::try_from(args)?;
        if !array.dtype().eq_ignore_nullability(fill_value.dtype()) {
            vortex_bail!("FillNull value must match array type (ignoring nullability)");
        }
        Ok(fill_value.dtype().clone())
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let FillNullArgs { array, .. } = FillNullArgs::try_from(args)?;
        Ok(array.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct FillNullArgs<'a> {
    array: &'a dyn Array,
    fill_value: &'a Scalar,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for FillNullArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 2 {
            vortex_bail!("FillNull requires 2 arguments");
        }

        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("FillNull requires an array"))?;
        let fill_value = value.inputs[1]
            .scalar()
            .ok_or_else(|| vortex_err!("FillNull requires a scalar"))?;

        Ok(FillNullArgs { array, fill_value })
    }
}
