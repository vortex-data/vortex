use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray, IntoArrayVariant};

impl TakeFn<BoolArray> for BoolEncoding {
    fn take(&self, array: &BoolArray, indices: &Array) -> VortexResult<Array> {
        let validity = array.validity();
        let indices = indices.clone().into_primitive()?;

        // For boolean arrays that roughly fit into a single page (at least, on Linux), it's worth
        // the overhead to convert to a Vec<bool>.
        let buffer = if array.len() <= 4096 {
            let bools = array.boolean_buffer().into_iter().collect_vec();
            match_each_integer_ptype!(indices.ptype(), |$I| {
                take_byte_bool(bools, indices.as_slice::<$I>())
            })
        } else {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                take_bool(&array.boolean_buffer(), indices.as_slice::<$I>())
            })
        };

        Ok(BoolArray::try_new(buffer, validity.take(indices.as_ref())?)?.into_array())
    }

    unsafe fn take_unchecked(&self, array: &BoolArray, indices: &Array) -> VortexResult<Array> {
        let validity = array.validity();
        let indices = indices.clone().into_primitive()?;

        // For boolean arrays that roughly fit into a single page (at least, on Linux), it's worth
        // the overhead to convert to a Vec<bool>.
        let buffer = if array.len() <= 4096 {
            let bools = array.boolean_buffer().into_iter().collect_vec();
            match_each_integer_ptype!(indices.ptype(), |$I| {
                take_byte_bool_unchecked(bools, indices.as_slice::<$I>())
            })
        } else {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                take_bool_unchecked(&array.boolean_buffer(), indices.as_slice::<$I>())
            })
        };

        // SAFETY: caller enforces indices are valid for array, and array has same len as validity.
        let validity = unsafe { validity.take_unchecked(indices.as_ref())? };
        Ok(BoolArray::try_new(buffer, validity)?.into_array())
    }
}

fn take_byte_bool<I: AsPrimitive<usize>>(bools: Vec<bool>, indices: &[I]) -> BooleanBuffer {
    BooleanBuffer::collect_bool(indices.len(), |idx| {
        bools[unsafe { (*indices.get_unchecked(idx)).as_() }]
    })
}

fn take_byte_bool_unchecked<I: AsPrimitive<usize>>(
    bools: Vec<bool>,
    indices: &[I],
) -> BooleanBuffer {
    BooleanBuffer::collect_bool(indices.len(), |idx| unsafe {
        *bools.get_unchecked((*indices.get_unchecked(idx)).as_())
    })
}

fn take_bool<I: AsPrimitive<usize>>(bools: &BooleanBuffer, indices: &[I]) -> BooleanBuffer {
    BooleanBuffer::collect_bool(indices.len(), |idx| {
        // We can always take from the indices unchecked since collect_bool just iterates len.
        bools.value(unsafe { (*indices.get_unchecked(idx)).as_() })
    })
}

fn take_bool_unchecked<I: AsPrimitive<usize>>(
    bools: &BooleanBuffer,
    indices: &[I],
) -> BooleanBuffer {
    BooleanBuffer::collect_bool(indices.len(), |idx| unsafe {
        // We can always take from the indices unchecked since collect_bool just iterates len.
        bools.value_unchecked((*indices.get_unchecked(idx)).as_())
    })
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

        let b =
            BoolArray::try_from(take(&reference, PrimitiveArray::from_iter([0, 3, 4])).unwrap())
                .unwrap();
        assert_eq!(
            b.boolean_buffer(),
            BoolArray::from_iter(vec![Some(false), None, Some(false)]).boolean_buffer()
        );
    }
}
