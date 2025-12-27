// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_dtype::ExtDType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::canonical::ToCanonical;
use crate::compute::ComputeFn;
use crate::compute::ComputeFnVTable;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Output;
use crate::compute::cast;
use crate::vtable::VTable;

static FILL_NULL_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("fill_null".into(), ArcRef::new_ref(&FillNull));
    for kernel in inventory::iter::<FillNullKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub(crate) fn warm_up_vtable() -> usize {
    FILL_NULL_FN.kernels().len()
}

/// Replace nulls in the array with another value.
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::{PrimitiveArray};
/// use vortex_array::compute::{fill_null};
/// use vortex_scalar::Scalar;
///
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let array = fill_null(array.as_ref(), &Scalar::from(42i32)).unwrap();
/// assert_eq!(array.display_values().to_string(), "[0i32, 42i32, 1i32, 42i32, 2i32]");
/// ```
pub fn fill_null(array: &dyn Array, fill_value: &Scalar) -> VortexResult<ArrayRef> {
    FILL_NULL_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), fill_value.into()],
            options: &(),
        })?
        .unwrap_array()
}

pub trait FillNullKernel: VTable {
    /// Kernel for replacing null values in an array with a fill value.
    ///
    /// TODO(connor): Actually enforce these constraints (so that casts do not fail).
    ///
    /// Implementations can assume that:
    /// - The array has at least one null value (not all valid, not all invalid).
    /// - The fill value is non-null.
    /// - For decimal arrays, the fill value can be successfully cast to the array's storage type.
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

struct FillNull;

impl ComputeFnVTable for FillNull {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let FillNullArgs { array, fill_value } = FillNullArgs::try_from(args)?;

        if !array.dtype().is_nullable() || array.all_valid() {
            return Ok(cast(array, fill_value.dtype())?.into());
        }

        if array.all_invalid() {
            return Ok(ConstantArray::new(fill_value.clone(), array.len())
                .into_array()
                .into());
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

        tracing::debug!("FillNullFn not implemented for {}", array.encoding_id());
        if !array.is_canonical() {
            let canonical_arr = array.to_canonical().into_array();
            return Ok(fill_null(canonical_arr.as_ref(), fill_value)?.into());
        }

        if matches!(array.dtype(), DType::Extension(..)) {
            let filled_storage = fill_null(
                array.to_extension().storage(),
                &fill_value.as_extension().storage(),
            )?;

            if filled_storage.dtype().nullability()
                == array
                    .to_extension()
                    .ext_dtype()
                    .storage_dtype()
                    .nullability()
            {
                return Ok(ExtensionArray::new(
                    array.to_extension().ext_dtype().clone(),
                    filled_storage,
                )
                .into_array()
                .into());
            } else {
                let new_ext_dtype = Arc::new(ExtDType::new(
                    array.to_extension().ext_dtype().id().clone(),
                    Arc::new(filled_storage.dtype().clone()),
                    array.to_extension().ext_dtype().metadata().cloned(),
                ));
                return Ok(ExtensionArray::new(new_ext_dtype, filled_storage)
                    .into_array()
                    .into());
            }
        }
        // TODO(joe): update fuzzer when fixed
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
