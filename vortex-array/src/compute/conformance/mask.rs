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
        Mask::try_from(&BoolArray::from_iter([true, false, false, true, true])).unwrap();
    let masked = mask(array, mask_array).unwrap();
    assert_eq!(masked.len(), array.len());
    assert!(!masked.is_valid(0).unwrap());
    assert_eq!(
        scalar_at(&masked, 1).unwrap(),
        scalar_at(array, 1).unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 2).unwrap(),
        scalar_at(array, 2).unwrap().into_nullable()
    );
    assert!(!masked.is_valid(3).unwrap());
    assert!(!masked.is_valid(4).unwrap());
}

fn test_empty_mask(array: &dyn Array) {
    let all_unmasked =
        Mask::try_from(&BoolArray::from_iter([false, false, false, false, false])).unwrap();
    let masked = mask(array, all_unmasked).unwrap();
    assert_eq!(masked.len(), array.len());
    assert_eq!(
        scalar_at(&masked, 0).unwrap(),
        scalar_at(array, 0).unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 1).unwrap(),
        scalar_at(array, 1).unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 2).unwrap(),
        scalar_at(array, 2).unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 3).unwrap(),
        scalar_at(array, 3).unwrap().into_nullable()
    );
    assert_eq!(
        scalar_at(&masked, 4).unwrap(),
        scalar_at(array, 4).unwrap().into_nullable()
    );
}

fn test_full_mask(array: &dyn Array) {
    let all_masked =
        Mask::try_from(&BoolArray::from_iter([true, true, true, true, true])).unwrap();
    let masked = mask(array, all_masked).unwrap();
    assert_eq!(masked.len(), array.len());
    assert!(!masked.is_valid(0).unwrap());
    assert!(!masked.is_valid(1).unwrap());
    assert!(!masked.is_valid(2).unwrap());
    assert!(!masked.is_valid(3).unwrap());
    assert!(!masked.is_valid(4).unwrap());

    let mask1 =
        Mask::try_from(&BoolArray::from_iter([true, false, false, true, true])).unwrap();
    let mask2 =
        Mask::try_from(&BoolArray::from_iter([false, true, false, false, true])).unwrap();
    let first = mask(array, mask1).unwrap();
    let double_masked = mask(&first, mask2).unwrap();
    assert_eq!(double_masked.len(), array.len());
    assert!(!double_masked.is_valid(0).unwrap());
    assert!(!double_masked.is_valid(1).unwrap());
    assert_eq!(
        scalar_at(&double_masked, 2).unwrap(),
        scalar_at(array, 2).unwrap().into_nullable()
    );
    assert!(!double_masked.is_valid(3).unwrap());
    assert!(!double_masked.is_valid(4).unwrap());
}