// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_mask::{AllOr, Mask};

use super::{ComputeFnVTable, InvocationArgs, Output, cast};
use crate::builders::{ArrayBuilder, builder_with_capacity};
use crate::compute::{ComputeFn, Kernel};
use crate::vtable::VTable;
use crate::{Array, ArrayRef};

/// Performs element-wise conditional selection between two arrays based on a mask.
///
/// Returns a new array where `result[i] = if_true[i]` when `mask[i]` is true,
/// otherwise `result[i] = if_false[i]`.
pub fn zip(if_true: &dyn Array, if_false: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    ZIP_FN
        .invoke(&InvocationArgs {
            inputs: &[if_true.into(), if_false.into(), mask.into()],
            options: &(),
        })?
        .unwrap_array()
}

pub static ZIP_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("zip".into(), ArcRef::new_ref(&Zip));
    for kernel in inventory::iter::<ZipKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct Zip;

impl ComputeFnVTable for Zip {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let ZipArgs {
            if_true,
            if_false,
            mask,
        } = ZipArgs::try_from(args)?;

        if mask.all_true() {
            return Ok(cast(if_true, &zip_return_dtype(if_true, if_false))?.into());
        }

        if mask.all_false() {
            return Ok(cast(if_false, &zip_return_dtype(if_true, if_false))?.into());
        }

        // check if if_true supports zip directly
        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }

        if let Some(output) = if_true.invoke(&ZIP_FN, args)? {
            return Ok(output);
        }

        // TODO(os): add invert_mask opt and check if if_false has a kernel like:
        //           kernel.invoke(Args(if_false, if_true, mask, invert_mask = true))

        Ok(zip_impl(
            if_true.to_canonical()?.as_ref(),
            if_false.to_canonical()?.as_ref(),
            mask,
        )?
        .into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let ZipArgs {
            if_true, if_false, ..
        } = ZipArgs::try_from(args)?;

        if !if_true.dtype().eq_ignore_nullability(if_false.dtype()) {
            vortex_bail!("input arrays to zip must have the same dtype");
        }
        Ok(zip_return_dtype(if_true, if_false))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let ZipArgs { if_true, mask, .. } = ZipArgs::try_from(args)?;
        // ComputeFn::invoke asserts if_true.len() == if_false.len(), because zip is elementwise
        if if_true.len() != mask.len() {
            vortex_bail!("input arrays must have the same length as the mask");
        }
        Ok(if_true.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct ZipArgs<'a> {
    if_true: &'a dyn Array,
    if_false: &'a dyn Array,
    mask: &'a Mask,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for ZipArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 3 {
            vortex_bail!("Expected 3 inputs for zip, found {}", value.inputs.len());
        }
        let if_true = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 0 to be an array"))?;

        let if_false = value.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 1 to be an array"))?;

        let mask = value.inputs[2]
            .mask()
            .ok_or_else(|| vortex_err!("Expected input 2 to be a mask"))?;

        Ok(Self {
            if_true,
            if_false,
            mask,
        })
    }
}

pub trait ZipKernel: VTable {
    fn zip(
        &self,
        if_true: &Self::Array,
        if_false: &dyn Array,
        mask: &Mask,
    ) -> VortexResult<Option<ArrayRef>>;
}

pub struct ZipKernelRef(pub ArcRef<dyn Kernel>);
inventory::collect!(ZipKernelRef);

#[derive(Debug)]
pub struct ZipKernelAdapter<V: VTable>(pub V);

impl<V: VTable + ZipKernel> ZipKernelAdapter<V> {
    pub const fn lift(&'static self) -> ZipKernelRef {
        ZipKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + ZipKernel> Kernel for ZipKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let ZipArgs {
            if_true,
            if_false,
            mask,
        } = ZipArgs::try_from(args)?;
        let Some(if_true) = if_true.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(V::zip(&self.0, if_true, if_false, mask)?.map(Into::into))
    }
}

pub(crate) fn zip_return_dtype(if_true: &dyn Array, if_false: &dyn Array) -> DType {
    if_true
        .dtype()
        .union_nullability(if_false.dtype().nullability())
}

fn zip_impl(if_true: &dyn Array, if_false: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    // if_true.len() == if_false.len() from ComputeFn::invoke
    let builder = builder_with_capacity(&zip_return_dtype(if_true, if_false), if_true.len());
    zip_impl_with_builder(if_true, if_false, mask, builder)
}

pub(crate) fn zip_impl_with_builder(
    if_true: &dyn Array,
    if_false: &dyn Array,
    mask: &Mask,
    mut builder: Box<dyn ArrayBuilder>,
) -> VortexResult<ArrayRef> {
    match mask.slices() {
        AllOr::All => Ok(if_true.to_array()),
        AllOr::None => Ok(if_false.to_array()),
        AllOr::Some(slices) => {
            for (start, end) in slices {
                builder.extend_from_array(&if_false.slice(builder.len(), *start))?;
                builder.extend_from_array(&if_true.slice(*start, *end))?;
            }
            if builder.len() < if_false.len() {
                builder.extend_from_array(&if_false.slice(builder.len(), if_false.len()))?;
            }
            Ok(builder.finish())
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{BoolArray, PrimitiveArray};
    use vortex_array::compute::zip;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_mask::Mask;

    #[test]
    fn test_zip_basic() {
        let mask =
            Mask::try_from(&BoolArray::from_iter([true, false, false, true, false])).unwrap();
        let if_true = PrimitiveArray::from_iter([10, 20, 30, 40, 50]).into_array();
        let if_false = PrimitiveArray::from_iter([1, 2, 3, 4, 5]).into_array();

        let result = zip(&if_true, &if_false, &mask).unwrap();
        let expected = PrimitiveArray::from_iter([10, 2, 3, 40, 5]);

        assert_eq!(
            result.to_primitive().unwrap().as_slice::<i32>(),
            expected.as_slice::<i32>()
        );
    }

    #[test]
    fn test_zip_all_true() {
        let mask = Mask::new_true(4);
        let if_true = PrimitiveArray::from_iter([10, 20, 30, 40]).into_array();
        let if_false =
            PrimitiveArray::from_option_iter([Some(1), Some(2), Some(3), None]).into_array();

        let result = zip(&if_true, &if_false, &mask).unwrap();

        assert_eq!(
            result.to_primitive().unwrap().as_slice::<i32>(),
            if_true.to_primitive().unwrap().as_slice::<i32>()
        );

        // result must be nullable even if_true was not
        assert_eq!(result.dtype(), if_false.dtype())
    }

    #[test]
    #[should_panic]
    fn test_invalid_lengths() {
        let mask = Mask::new_false(4);
        let if_true = PrimitiveArray::from_iter([10, 20, 30]).into_array();
        let if_false = PrimitiveArray::from_iter([1, 2, 3, 4]).into_array();

        zip(&if_true, &if_false, &mask).unwrap();
    }
}
