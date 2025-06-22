use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, match_each_integer_ptype};
use vortex_error::VortexResult;

pub fn cast_canonical_array(array: &ArrayRef, target: &DType) -> VortexResult<Option<ArrayRef>> {
    // TODO(joe): support more casting options
    if !target.is_int() || !array.dtype().is_int() {
        return Ok(None);
    }
    Ok(Some(match_each_integer_ptype!(
        array.dtype().as_ptype(),
        |In| {
            match_each_integer_ptype!(target.as_ptype(), |Out| {
                // Since the cast itself would truncate.
                #[allow(clippy::cast_possible_truncation)]
                PrimitiveArray::new(
                    array
                        .to_primitive()?
                        .as_slice::<In>()
                        .iter()
                        .map(|v| *v as Out)
                        .collect::<Buffer<Out>>(),
                    Validity::from_mask(array.validity_mask()?, target.nullability()),
                )
                .to_array()
            })
        }
    )))
}
