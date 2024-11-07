use std::cmp::min;

use arrow_buffer::buffer::BooleanBuffer;
use arrow_buffer::BooleanBufferBuilder;
use num_traits::{AsPrimitive, FromPrimitive};
use vortex_array::array::{BoolArray, PrimitiveArray};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{vortex_panic, VortexExpect as _, VortexResult};

pub fn runend_bool_encode(elements: &BoolArray) -> (PrimitiveArray, bool) {
    let (arr, start) = runend_bool_encode_slice(&elements.boolean_buffer());
    (arr.into(), start)
}

pub fn runend_bool_encode_slice(elements: &BooleanBuffer) -> (Vec<u64>, bool) {
    let mut iter = elements.set_slices();
    let Some((start, end)) = iter.next() else {
        return (vec![elements.len() as u64], false);
    };

    let mut ends = Vec::new();
    let first_bool = start == 0;
    if !first_bool {
        ends.push(start as u64)
    }
    ends.push(end as u64);
    for (s, e) in iter {
        ends.push(s as u64);
        ends.push(e as u64);
    }

    let last_end = ends.last().vortex_expect(
        "RunEndBoolArray cannot have empty run ends (by construction); this should be impossible",
    );
    if *last_end != elements.len() as u64 {
        ends.push(elements.len() as u64)
    }

    (ends, first_bool)
}

pub fn runend_bool_decode(
    run_ends: &PrimitiveArray,
    start: bool,
    validity: Validity,
    offset: usize,
    length: usize,
) -> VortexResult<BoolArray> {
    match_each_integer_ptype!(run_ends.ptype(), |$E| {
        let bools = runend_bool_decode_slice::<$E>(run_ends.maybe_null_slice(), start, offset, length);
        BoolArray::try_new(bools, validity)
    })
}

pub fn runend_bool_decode_slice<E: NativePType + AsPrimitive<usize> + FromPrimitive + Ord>(
    run_ends: &[E],
    start: bool,
    offset: usize,
    length: usize,
) -> BooleanBuffer {
    let offset_e = E::from_usize(offset).unwrap_or_else(|| {
        vortex_panic!(
            "offset {} cannot be converted to {}",
            offset,
            std::any::type_name::<E>()
        )
    });
    let length_e = E::from_usize(length).unwrap_or_else(|| {
        vortex_panic!(
            "length {} cannot be converted to {}",
            length,
            std::any::type_name::<E>()
        )
    });
    let trimmed_ends = run_ends
        .iter()
        .map(|v| *v - offset_e)
        .map(|v| min(v, length_e));

    let mut decoded = BooleanBufferBuilder::new(length);
    for (idx, end) in trimmed_ends.enumerate() {
        decoded.append_n(end.as_() - decoded.len(), value_at_index(idx, start));
    }
    BooleanBuffer::from(decoded)
}

pub fn value_at_index(idx: usize, start: bool) -> bool {
    if idx % 2 == 0 {
        start
    } else {
        !start
    }
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use itertools::Itertools;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::array::BoolArray;
    use vortex_array::compute::SliceFn;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayVariant;

    use crate::compress::{
        runend_bool_decode, runend_bool_decode_slice, runend_bool_encode, runend_bool_encode_slice,
    };

    #[test]
    fn encode_bool() {
        let encoded =
            runend_bool_encode_slice(&BooleanBuffer::from([true, true, false, true].as_slice()));
        assert_eq!(encoded, (vec![2, 3, 4], true))
    }

    #[test]
    fn encode_bool_false_true_end() {
        let mut input = vec![false; 66];
        input.extend([true, true]);
        let encoded = runend_bool_encode_slice(&BooleanBuffer::from(input));
        assert_eq!(encoded, (vec![66, 68], false))
    }

    #[test]
    fn encode_bool_false() {
        let encoded =
            runend_bool_encode_slice(&BooleanBuffer::from([false, false, true, false].as_slice()));
        assert_eq!(encoded, (vec![2, 3, 4], false))
    }

    #[test]
    fn encode_decode_bool() {
        let input = [true, true, false, true, true, false];
        let (ends, start) = runend_bool_encode_slice(&BooleanBuffer::from(input.as_slice()));

        let decoded = runend_bool_decode_slice(ends.as_slice(), start, 0, input.len());
        assert_eq!(decoded, BooleanBuffer::from(input.as_slice()))
    }

    #[test]
    fn encode_decode_bool_false_start() {
        let input = [false, false, true, true, false, true, true, false];
        let (ends, start) = runend_bool_encode_slice(&BooleanBuffer::from(input.as_slice()));

        let decoded = runend_bool_decode_slice(ends.as_slice(), start, 0, input.len());
        assert_eq!(decoded, BooleanBuffer::from(input.as_slice()))
    }

    #[test]
    fn encode_decode_bool_false_start_long() {
        let input = vec![true; 1024];
        // input.extend([false, true, false, true].as_slice());
        let (ends, start) = runend_bool_encode_slice(&BooleanBuffer::from(input.as_slice()));

        let decoded = runend_bool_decode_slice(ends.as_slice(), start, 0, input.len());
        assert_eq!(decoded, BooleanBuffer::from(input.as_slice()))
    }

    #[test]
    fn encode_decode_random() {
        let mut rng = StdRng::seed_from_u64(4352);
        let input = (0..1024 * 4).map(|_x| rng.gen::<bool>()).collect_vec();
        let (ends, start) = runend_bool_encode_slice(&BooleanBuffer::from(input.as_slice()));

        let decoded = runend_bool_decode_slice(ends.as_slice(), start, 0, input.len());
        assert_eq!(decoded, BooleanBuffer::from(input.as_slice()))
    }

    #[test]
    fn encode_decode_offset_array() {
        let mut rng = StdRng::seed_from_u64(39451);
        let input = (0..1024 * 8 - 61).map(|_x| rng.gen::<bool>()).collect_vec();
        let b = BoolArray::from(input.clone());
        let b = b.slice(3, 1024 * 8 - 66).unwrap().into_bool().unwrap();
        let (ends, start) = runend_bool_encode(&b);

        let decoded = runend_bool_decode(&ends, start, Validity::NonNullable, 0, 1024 * 8 - 69)
            .unwrap()
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .iter()
            .collect_vec();
        assert_eq!(input[3..1024 * 8 - 66], decoded)
    }
}
