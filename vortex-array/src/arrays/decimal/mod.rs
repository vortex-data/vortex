// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::DecimalArrayParts;
pub use array::DecimalData;
pub use vtable::DecimalArray;

pub(crate) mod compute;

mod vtable;
pub use compute::rules::DecimalMaskedValidityRule;
pub use vtable::Decimal;

mod utils;
pub use utils::*;

#[cfg(test)]
mod tests {
    use arrow_array::Decimal128Array;

    #[test]
    fn test_decimal() {
        // They pass it b/c the DType carries the information. No other way to carry a
        // dtype except via the array.
        let value = Decimal128Array::new_null(100);
        let numeric = value.value(10);
        assert_eq!(numeric, 0i128);
    }
}
