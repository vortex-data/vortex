// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::arrays::PrimitiveArray;
use crate::match_each_ptype_width;

cfg_if::cfg_if! {
    if #[cfg(target_feature = "avx2")] {
        pub const IS_CONST_LANE_WIDTH: usize = 32;
    } else {
        pub const IS_CONST_LANE_WIDTH: usize = 16;
    }
}

/// Checks whether every value equals the first value, comparing in groups of `WIDTH`.
///
/// Callers that require bitwise equality must pass byte records or integer bit representations.
/// The slice must be nonempty and `WIDTH` must be nonzero.
pub fn compute_is_constant<T: Copy + PartialEq, const WIDTH: usize>(values: &[T]) -> bool {
    let first_value = values[0];
    let first_vec = &[first_value; WIDTH];

    let (chunks, remainder) = values[1..].as_chunks::<WIDTH>();
    for chunk in chunks {
        if first_vec != chunk {
            return false;
        }
    }

    for value in remainder {
        if *value != first_value {
            return false;
        }
    }

    true
}

pub(super) fn check_primitive_constant(array: &PrimitiveArray) -> bool {
    let values = array
        .buffer_handle()
        .as_host_opt()
        .vortex_expect("is_constant requires host values");

    match_each_ptype_width!(
        array.ptype(),
        |W| {
            compute_is_constant::<_, { IS_CONST_LANE_WIDTH / W }>(
                values.as_slice().as_chunks::<W>().0,
            )
        },
        _ => { unreachable!("primitive widths are specialized") },
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::check_primitive_constant;
    use crate::arrays::PrimitiveArray;

    #[rstest]
    #[case::zero(&[0], true)]
    #[case::signed_zero(&[0, 0x80000000], false)]
    #[case::infinity(&[0x7f800000], true)]
    #[case::same_nan(&[0x7fc00001], true)]
    #[case::different_nan(&[0x7fc00001, 0x7fc00002], false)]
    fn compares_stored_bits(
        #[case] bits: &[u32],
        #[case] expected: bool,
        #[values(2, 16, 17, 65)] len: usize,
    ) {
        let values = bits.iter().copied().cycle().take(len).map(f32::from_bits);
        let array = PrimitiveArray::from_iter(values);
        assert_eq!(check_primitive_constant(&array), expected);
    }
}
