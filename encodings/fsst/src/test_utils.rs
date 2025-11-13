// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::arrays::{DictArray, PrimitiveArray, VarBinArray};
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::VortexUnwrap;

use crate::{fsst_compress, fsst_train_compressor};

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
    DictArray::try_new(codes.into_array(), values).vortex_unwrap()
}
