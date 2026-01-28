// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::ComputeFnVTable;
use super::InvocationArgs;
use super::Output;
use super::cast;
use crate::Array;
use crate::ArrayRef;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity;
use crate::compute::ComputeFn;
use crate::compute::Kernel;
use crate::vtable::VTable;

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

pub(crate) fn warm_up_vtable() -> usize {
    ZIP_FN.kernels().len()
}

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

        if !if_true.is_canonical() || !if_false.is_canonical() {
            return zip(
                if_true.to_canonical()?.as_ref(),
                if_false.to_canonical()?.as_ref(),
                mask,
            )
            .map(Into::into);
        }

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
    assert_eq!(
        if_true.len(),
        if_false.len(),
        "ComputeFn::invoke checks that arrays have the same size"
    );

    let return_type = zip_return_dtype(if_true, if_false);
    zip_impl_with_builder(
        if_true,
        if_false,
        mask,
        builder_with_capacity(&return_type, if_true.len()),
    )
}

fn zip_impl_with_builder(
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
                builder.extend_from_array(&if_false.slice(builder.len()..*start)?);
                builder.extend_from_array(&if_true.slice(*start..*end)?);
            }
            if builder.len() < if_false.len() {
                builder.extend_from_array(&if_false.slice(builder.len()..if_false.len())?);
            }
            Ok(builder.finish())
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_select::zip::zip as arrow_zip;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewVTable;
    use crate::arrow::IntoArrowArray;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::BufferGrowthStrategy;
    use crate::builders::VarBinViewBuilder;
    use crate::compute::zip;

    #[test]
    fn test_zip_basic() {
        let mask = Mask::from_iter([true, false, false, true, false]);
        let if_true = buffer![10, 20, 30, 40, 50].into_array();
        let if_false = buffer![1, 2, 3, 4, 5].into_array();

        let result = zip(&if_true, &if_false, &mask).unwrap();
        let expected = buffer![10, 2, 3, 40, 5].into_array();

        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_zip_all_true() {
        let mask = Mask::new_true(4);
        let if_true = buffer![10, 20, 30, 40].into_array();
        let if_false =
            PrimitiveArray::from_option_iter([Some(1), Some(2), Some(3), None]).into_array();

        let result = zip(&if_true, &if_false, &mask).unwrap();
        let expected =
            PrimitiveArray::from_option_iter([Some(10), Some(20), Some(30), Some(40)]).into_array();

        assert_arrays_eq!(result, expected);

        // result must be nullable even if_true was not
        assert_eq!(result.dtype(), if_false.dtype())
    }

    #[test]
    #[should_panic]
    fn test_invalid_lengths() {
        let mask = Mask::new_false(4);
        let if_true = buffer![10, 20, 30].into_array();
        let if_false = buffer![1, 2, 3, 4].into_array();

        zip(&if_true, &if_false, &mask).unwrap();
    }

    #[test]
    fn test_fragmentation() {
        let len = 100;

        let const1 = ConstantArray::new(
            Scalar::utf8("hello_this_is_a_longer_string", Nullability::Nullable),
            len,
        )
        .to_array();

        let const2 = ConstantArray::new(
            Scalar::utf8("world_this_is_another_string", Nullability::Nullable),
            len,
        )
        .to_array();

        // Create a mask that alternates frequently to cause fragmentation
        // Pattern: take from const1 at even indices, const2 at odd indices
        let indices: Vec<usize> = (0..len).step_by(2).collect();
        let mask = Mask::from_indices(len, indices);

        let result = zip(&const1, &const2, &mask).unwrap();

        insta::assert_snapshot!(result.display_tree(), @r"
        root: vortex.varbinview(utf8?, len=100) nbytes=1.66 kB (100.00%) [all_valid]
          metadata: EmptyMetadata
          buffer (align=1): 29 B (1.75%)
          buffer (align=1): 28 B (1.69%)
          buffer (align=16): 1.60 kB (96.56%)
        ");

        // test wrapped in a struct
        let wrapped1 = StructArray::try_from_iter([("nested", const1)])
            .unwrap()
            .to_array();
        let wrapped2 = StructArray::try_from_iter([("nested", const2)])
            .unwrap()
            .to_array();

        let wrapped_result = zip(&wrapped1, &wrapped2, &mask).unwrap();
        insta::assert_snapshot!(wrapped_result.display_tree(), @r"
        root: vortex.struct({nested=utf8?}, len=100) nbytes=1.66 kB (100.00%)
          metadata: EmptyMetadata
          nested: vortex.varbinview(utf8?, len=100) nbytes=1.66 kB (100.00%) [all_valid]
            metadata: EmptyMetadata
            buffer (align=1): 29 B (1.75%)
            buffer (align=1): 28 B (1.69%)
            buffer (align=16): 1.60 kB (96.56%)
        ");
    }

    #[test]
    fn test_varbinview_zip() {
        let if_true = {
            let mut builder = VarBinViewBuilder::new(
                DType::Utf8(Nullability::NonNullable),
                10,
                Default::default(),
                BufferGrowthStrategy::fixed(64 * 1024),
                0.0,
            );
            for _ in 0..100 {
                builder.append_value("Hello");
                builder.append_value("Hello this is a long string that won't be inlined.");
            }
            builder.finish()
        };

        let if_false = {
            let mut builder = VarBinViewBuilder::new(
                DType::Utf8(Nullability::NonNullable),
                10,
                Default::default(),
                BufferGrowthStrategy::fixed(64 * 1024),
                0.0,
            );
            for _ in 0..100 {
                builder.append_value("Hello2");
                builder.append_value("Hello2 this is a long string that won't be inlined.");
            }
            builder.finish()
        };

        // [1,2,4,5,7,8,..]
        let mask = Mask::from_indices(200, (0..100).filter(|i| i % 3 != 0).collect());

        let zipped = zip(&if_true, &if_false, &mask).unwrap();
        let zipped = zipped.as_opt::<VarBinViewVTable>().unwrap();
        assert_eq!(zipped.nbuffers(), 2);

        // assert the result is the same as arrow
        let expected = arrow_zip(
            mask.into_array()
                .into_arrow_preferred()
                .unwrap()
                .as_boolean(),
            &if_true.into_arrow_preferred().unwrap(),
            &if_false.into_arrow_preferred().unwrap(),
        )
        .unwrap();

        let actual = zipped.clone().into_array().into_arrow_preferred().unwrap();
        assert_eq!(actual.as_ref(), expected.as_ref());
    }
}
