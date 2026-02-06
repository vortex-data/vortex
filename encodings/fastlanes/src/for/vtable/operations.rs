// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;

    use crate::FoRArray;

    #[test]
    fn for_scalar_at() {
        let for_arr =
            FoRArray::encode(PrimitiveArray::from_iter([-100, 1100, 1500, 1900])).unwrap();
        let expected = PrimitiveArray::from_iter([-100, 1100, 1500, 1900]);
        assert_arrays_eq!(for_arr, expected);
    }
}
