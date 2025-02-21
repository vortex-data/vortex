use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::PrimitiveEncoding;
use crate::compute::MaskFn;
use crate::variants::PrimitiveArrayTrait as _;
use crate::{Array, ArrayRef, IntoArray};

impl MaskFn<&PrimitiveArray> for PrimitiveEncoding {
    fn mask(&self, array: &PrimitiveArray, mask: Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().mask(&mask)?;
        Ok(
            PrimitiveArray::from_byte_buffer(array.byte_buffer().clone(), array.ptype(), validity)
                .into_array(),
        )
    }
}
