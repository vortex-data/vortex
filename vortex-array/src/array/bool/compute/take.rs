use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::{TakeFn, TakeOptions};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};

impl TakeFn<BoolArray> for BoolEncoding {
    fn take(
        &self,
        array: &BoolArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        let validity = array.validity();
        let indices = indices.clone().into_primitive()?;

        // For boolean arrays that roughly fit into a single page (at least, on Linux), it's worth
        // the overhead to convert to a Vec<bool>.
        let buffer = if array.len() <= 4096 {
            let bools = array.boolean_buffer().into_iter().collect_vec();
            match_each_integer_ptype!(indices.ptype(), |$I| {
                if options.skip_bounds_check {
                    take_byte_bool_unchecked(bools, indices.maybe_null_slice::<$I>())
                } else {
                    take_byte_bool(bools, indices.maybe_null_slice::<$I>())
                }
            })
        } else {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                if options.skip_bounds_check {
                    take_bool_unchecked(&array.boolean_buffer(), indices.maybe_null_slice::<$I>())
                } else {
                    take_bool(&array.boolean_buffer(), indices.maybe_null_slice::<$I>())
                }
            })
        };

        Ok(BoolArray::try_new(buffer, validity.take(indices.as_ref(), options)?)?.into_array())
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
    use crate::compute::{take, TakeOptions};

    #[test]
    fn take_nullable() {
        let reference = BoolArray::from_iter(vec![
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(false),
        ]);

        let b = BoolArray::try_from(
            take(
                &reference,
                PrimitiveArray::from(vec![0, 3, 4]),
                TakeOptions::default(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            b.boolean_buffer(),
            BoolArray::from_iter(vec![Some(false), None, Some(false)]).boolean_buffer()
        );
    }
}
