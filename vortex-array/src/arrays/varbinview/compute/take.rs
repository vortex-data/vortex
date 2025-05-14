use std::ops::Deref;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::arrays::{BinaryView, VarBinViewArray, VarBinViewVTable};
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

/// Take involves creating a new array that references the old array, just with the given set of views.
impl TakeKernel for VarBinViewVTable {
    fn take(&self, array: &VarBinViewArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        // Compute the new validity

        // This is valid since all elements (of all arrays) even null values are inside must be the
        // min-max valid range.
        let validity = array.validity().take(indices)?;
        let indices = indices.to_primitive()?;

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |$I| {
        // This is valid since all elements even null values are inside the min-max valid range.
            take_views(array.views(), indices.as_slice::<$I>())
        });

        Ok(VarBinViewArray::try_new(
            views_buffer,
            array.buffers().to_vec(),
            array
                .dtype()
                .union_nullability(indices.dtype().nullability()),
            validity,
        )?
        .into_array())
    }
}

register_kernel!(TakeKernelAdapter(VarBinViewVTable).lift());

fn take_views<I: AsPrimitive<usize>>(
    views: &Buffer<BinaryView>,
    indices: &[I],
) -> Buffer<BinaryView> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    Buffer::<BinaryView>::from_iter(indices.iter().map(|i| views_ref[i.as_()]))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::accessor::ArrayAccessor;
    use crate::array::Array;
    use crate::arrays::VarBinViewArray;
    use crate::canonical::ToCanonical;
    use crate::compute::take;

    #[test]
    fn take_nullable() {
        let arr = VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            None,
            Some("six"),
        ]);

        let taken = take(arr.as_ref(), &buffer![0, 3].into_array()).unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken
                .to_varbinview()
                .unwrap()
                .with_iterator(|it| it
                    .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                    .collect::<Vec<_>>())
                .unwrap(),
            [Some("one".to_string()), Some("four".to_string())]
        );
    }
}
