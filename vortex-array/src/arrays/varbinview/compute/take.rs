use std::ops::Deref;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::arrays::{BinaryView, VarBinViewArray, VarBinViewEncoding};
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, ToCanonical};

/// Take involves creating a new array that references the old array, just with the given set of views.
impl TakeFn<&VarBinViewArray> for VarBinViewEncoding {
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
            array.dtype().with_nullability(
                (array.dtype().is_nullable() || indices.dtype().is_nullable()).into(),
            ),
            validity,
        )?
        .into_array())
    }
}

fn take_views<I: AsPrimitive<usize>>(
    views: &Buffer<BinaryView>,
    indices: &[I],
) -> Buffer<BinaryView> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    Buffer::<BinaryView>::from_iter(indices.iter().map(|i| views_ref[i.as_()]))
}
