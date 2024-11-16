use arrow_buffer::BooleanBuffer;
use num_traits::AsPrimitive;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::array::BoolArray;
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

impl TakeFn for BoolArray {
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        let validity = self.validity();
        let indices = indices.clone().into_primitive()?;
        match_each_integer_ptype!(indices.ptype(), |$I| {
            Ok(BoolArray::try_new(
                take_bool(&self.boolean_buffer(), indices.maybe_null_slice::<$I>()),
                validity.take(indices.as_ref())?,
            )?.into_array())
        })
    }
}

fn take_bool<I: AsPrimitive<usize>>(bools: &BooleanBuffer, indices: &[I]) -> BooleanBuffer {
    BooleanBuffer::collect_bool(indices.len(), |idx| bools.value(indices[idx].as_()))
}

#[cfg(test)]
mod test {
    use crate::array::primitive::PrimitiveArray;
    use crate::array::BoolArray;
    use crate::compute::take;

    #[test]
    fn take_nullable() {
        let reference = BoolArray::from_iter(vec![
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(false),
        ]);

        let b = BoolArray::try_from(take(&reference, PrimitiveArray::from(vec![0, 3, 4])).unwrap())
            .unwrap();
        assert_eq!(
            b.boolean_buffer(),
            BoolArray::from_iter(vec![Some(false), None, Some(false)]).boolean_buffer()
        );
    }
}
