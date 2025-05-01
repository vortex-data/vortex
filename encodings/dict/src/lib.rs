//! Implementation of Dictionary encoding.
//!
//! Expose a [DictArray] which is zero-copy equivalent to Arrow's
//! [DictionaryArray](https://docs.rs/arrow/latest/arrow/array/struct.DictionaryArray.html).
pub use array::*;

mod array;
pub mod builders;
mod compute;
mod serde;
#[cfg(feature = "test-harness")]
pub mod test;
mod variants;

#[cfg(test)]
mod test {
    use std::str::from_utf8;

    use rand::distr::Alphanumeric;
    use rand::prelude::IndexedRandom;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::{ConstantArray, VarBinArray};
    use vortex_array::compute::{Operator, compare, slice};

    use crate::builders::dict_encode;

    #[test]
    fn foo() {
        let codes_len = 10_000;
        let values_len = 10_000;

        let varbin_arr = VarBinArray::from(gen_varbin_words(codes_len.max(values_len), values_len));
        let dict = dict_encode(&varbin_arr).unwrap();
        let dict = slice(&dict, 0, codes_len).unwrap();
        let bytes = varbin_arr
            .with_iterator(|i| i.next().unwrap().unwrap().to_vec())
            .unwrap();
        let value = from_utf8(bytes.as_slice()).unwrap();

        compare(&dict, &ConstantArray::new(value, codes_len), Operator::Eq).unwrap();
    }

    pub fn gen_varbin_words(len: usize, unique_count: usize) -> Vec<String> {
        let rng = &mut StdRng::seed_from_u64(0);
        let dict: Vec<String> = (0..unique_count)
            .map(|_| {
                rng.sample_iter(&Alphanumeric)
                    .take(16)
                    .map(char::from)
                    .collect()
            })
            .collect();

        (0..len)
            .map(|_| dict.choose(rng).unwrap().clone())
            .collect()
    }
}
