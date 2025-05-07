use std::sync::LazyLock;

use arcref::ArcRef;
use arrow_array::BooleanArray;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output, cast};
use crate::encoding::Encoding;
use crate::{Array, ArrayRef};

/// Replace values with null where the mask is true.
///
/// The returned array is nullable but otherwise has the same dtype and length as `array`.
///
/// # Examples
///
/// ```
/// use vortex_array::IntoArray;
/// use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// use vortex_array::compute::{ mask};
/// use vortex_mask::Mask;
/// use vortex_scalar::Scalar;
///
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let mask_array = Mask::try_from(
///     &BoolArray::from_iter([true, false, false, false, true]),
/// )
/// .unwrap();
///
/// let masked = mask(&array, &mask_array).unwrap();
/// assert_eq!(masked.len(), 5);
/// assert!(!masked.is_valid(0).unwrap());
/// assert!(!masked.is_valid(1).unwrap());
/// assert_eq!(masked.scalar_at(2).unwrap(), Scalar::from(Some(1)));
/// assert!(!masked.is_valid(3).unwrap());
/// assert!(!masked.is_valid(4).unwrap());
/// ```
///
pub fn mask(array: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    MASK_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), mask.into()],
            options: &(),
        })?
        .unwrap_array()
}

pub struct MaskKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(MaskKernelRef);

pub trait MaskKernel: Encoding {
    /// Replace masked values with null in array.
    fn mask(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef>;
}

#[derive(Debug)]
pub struct MaskKernelAdapter<E: Encoding>(pub E);

impl<E: Encoding + MaskKernel> MaskKernelAdapter<E> {
    pub const fn lift(&'static self) -> MaskKernelRef {
        MaskKernelRef(ArcRef::new_ref(self))
    }
}

impl<E: Encoding + MaskKernel> Kernel for MaskKernelAdapter<E> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = MaskArgs::try_from(args)?;
        let Some(array) = inputs.array.as_any().downcast_ref::<E::Array>() else {
            return Ok(None);
        };
        Ok(Some(E::mask(&self.0, array, inputs.mask)?.into()))
    }
}

pub static MASK_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("mask".into(), ArcRef::new_ref(&MaskFn));
    for kernel in inventory::iter::<MaskKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct MaskFn;

impl ComputeFnVTable for MaskFn {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let MaskArgs { array, mask } = MaskArgs::try_from(args)?;

        if matches!(mask, Mask::AllFalse(_)) {
            // Fast-path for empty mask
            return Ok(cast(array, &array.dtype().as_nullable())?.into());
        }

        if matches!(mask, Mask::AllTrue(_)) {
            // Fast-path for full mask.
            return Ok(ConstantArray::new(
                Scalar::null(array.dtype().clone().as_nullable()),
                array.len(),
            )
            .into_array()
            .into());
        }

        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = array.invoke(&MASK_FN, args)? {
            return Ok(output);
        }

        // Fallback: implement using Arrow kernels.
        log::debug!("No mask implementation found for {}", array.encoding());

        let array_ref = array.to_array().into_arrow_preferred()?;
        let mask = BooleanArray::new(mask.to_boolean_buffer(), None);

        let masked = arrow_select::nullif::nullif(array_ref.as_ref(), &mask)?;

        Ok(ArrayRef::from_arrow(masked, true).into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let MaskArgs { array, .. } = MaskArgs::try_from(args)?;
        Ok(array.dtype().as_nullable())
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let MaskArgs { array, mask } = MaskArgs::try_from(args)?;

        if mask.len() != array.len() {
            vortex_bail!(
                "mask.len() is {}, does not equal array.len() of {}",
                mask.len(),
                array.len()
            );
        }

        Ok(mask.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct MaskArgs<'a> {
    array: &'a dyn Array,
    mask: &'a Mask,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for MaskArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 2 {
            vortex_bail!("Mask function requires 2 arguments");
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 0 to be an array"))?;
        let mask = value.inputs[1]
            .mask()
            .ok_or_else(|| vortex_err!("Expected input 1 to be a mask"))?;

        Ok(MaskArgs { array, mask })
    }
}
