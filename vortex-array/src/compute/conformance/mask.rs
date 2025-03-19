use vortex_error::VortexUnwrap;
use vortex_mask::Mask;

use crate::Array;
use crate::arrays::BoolArray;
use crate::compute::{mask, scalar_at};

pub fn test_mask(array: &dyn Array) {
    assert_eq!(array.len(), 5);
    test_heterogenous_mask(array);
    test_empty_mask(array);
    test_full_mask(array);
}

fn test_heterogenous_mask(array: &dyn Array) {
    let mask_array =
        Mask::try_from(&BoolArray::from_iter([true, false, false, true, true])).vortex_unwrap();
    let masked = mask(array, mask_array).vortex_unwrap();
    assert_eq!(masked.len(), array.len());
    assert!(!masked.is_valid(0).vortex_unwrap());
    assert_eq!(
        scalar_at(&masked, 1).vortex_unwrap(),
        scalar_at(array, 1).vortex_unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 2).vortex_unwrap(),
        scalar_at(array, 2).vortex_unwrap().into_nullable()
    );
    assert!(!masked.is_valid(3).vortex_unwrap());
    assert!(!masked.is_valid(4).vortex_unwrap());
}

fn test_empty_mask(array: &dyn Array) {
    let all_unmasked =
        Mask::try_from(&BoolArray::from_iter([false, false, false, false, false])).vortex_unwrap();
    let masked = mask(array, all_unmasked).vortex_unwrap();
    assert_eq!(masked.len(), array.len());
    assert_eq!(
        scalar_at(&masked, 0).vortex_unwrap(),
        scalar_at(array, 0).vortex_unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 1).vortex_unwrap(),
        scalar_at(array, 1).vortex_unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 2).vortex_unwrap(),
        scalar_at(array, 2).vortex_unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 3).vortex_unwrap(),
        scalar_at(array, 3).vortex_unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 4).vortex_unwrap(),
        scalar_at(array, 4).vortex_unwrap().into_nullable()
    );
}

fn test_full_mask(array: &dyn Array) {
    let all_masked =
        Mask::try_from(&BoolArray::from_iter([true, true, true, true, true])).vortex_unwrap();
    let masked = mask(array, all_masked).vortex_unwrap();
    assert_eq!(masked.len(), array.len());
    assert!(!masked.is_valid(0).vortex_unwrap());
    assert!(!masked.is_valid(1).vortex_unwrap());
    assert!(!masked.is_valid(2).vortex_unwrap());
    assert!(!masked.is_valid(3).vortex_unwrap());
    assert!(!masked.is_valid(4).vortex_unwrap());

    let mask1 =
        Mask::try_from(&BoolArray::from_iter([true, false, false, true, true])).vortex_unwrap();
    let mask2 =
        Mask::try_from(&BoolArray::from_iter([false, true, false, false, true])).vortex_unwrap();
    let first = mask(array, mask1).vortex_unwrap();
    let double_masked = mask(&first, mask2).vortex_unwrap();
    assert_eq!(double_masked.len(), array.len());
    assert!(!double_masked.is_valid(0).vortex_unwrap());
    assert!(!double_masked.is_valid(1).vortex_unwrap());
    assert_eq!(
        scalar_at(&double_masked, 2).vortex_unwrap(),
        scalar_at(array, 2).vortex_unwrap().into_nullable()
    );
    assert!(!double_masked.is_valid(3).vortex_unwrap());
    assert!(!double_masked.is_valid(4).vortex_unwrap());
}
