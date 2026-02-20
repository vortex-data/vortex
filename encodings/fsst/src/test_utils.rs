// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use rand::Rng;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexExpect;

use crate::fsst_compress;
use crate::fsst_train_compressor;

pub fn gen_fsst_test_data(len: usize, avg_str_len: usize, unique_chars: u8) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let mut strings = Vec::with_capacity(len);

    for _ in 0..len {
        // Generate a random string with length around `avg_len`. The number of possible
        // characters within the random string is defined by `unique_chars`.
        let len = avg_str_len * rng.random_range(50..=150) / 100;
        strings.push(Some(
            (0..len)
                .map(|_| rng.random_range(b'a'..(b'a' + unique_chars)))
                .collect::<Vec<u8>>(),
        ));
    }

    let varbin = VarBinArray::from_iter(
        strings
            .into_iter()
            .map(|opt_s| opt_s.map(Vec::into_boxed_slice)),
        DType::Binary(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);

    fsst_compress(varbin, &compressor).into_array()
}

pub fn gen_dict_fsst_test_data<T: NativePType>(
    len: usize,
    unique_values: usize,
    str_len: usize,
    unique_char_count: u8,
) -> DictArray {
    let values = gen_fsst_test_data(len, str_len, unique_char_count);
    let mut rng = StdRng::seed_from_u64(0);
    let codes = (0..len)
        .map(|_| T::from(rng.random_range(0..unique_values)).unwrap())
        .collect::<PrimitiveArray>();
    DictArray::try_new(codes.into_array(), values)
        .vortex_expect("DictArray::try_new should succeed for test data")
}
