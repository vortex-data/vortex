#![allow(clippy::unwrap_used)]

use rand::distributions::{Alphanumeric, Distribution, Standard};
use rand::prelude::{SliceRandom, StdRng};
use rand::{Rng, SeedableRng};
use vortex_array::arrays::{ChunkedArray, PrimitiveArray, VarBinArray};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{VortexResult, VortexUnwrap};
use vortex_fsst::{fsst_compress, fsst_train_compressor};

use crate::DictArray;

pub fn gen_primitive_for_dict<T: NativePType>(len: usize, unique_count: usize) -> PrimitiveArray
where
    Standard: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..unique_count)
        .map(|_| rng.gen::<T>())
        .collect::<Vec<T>>();
    let data = (0..len)
        .map(|_| *values.choose(&mut rng).unwrap())
        .collect::<Buffer<_>>();
    PrimitiveArray::new(data, Validity::NonNullable)
}

pub fn gen_primitive_dict<V: NativePType, C: NativePType>(
    len: usize,
    unique_count: usize,
) -> VortexResult<DictArray>
where
    Standard: Distribution<V>,
{
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..unique_count)
        .map(|_| rng.gen::<V>())
        .collect::<PrimitiveArray>();

    let codes = (0..len)
        .map(|_| C::from(rng.gen_range(0..unique_count)).unwrap())
        .collect::<PrimitiveArray>();

    DictArray::try_new(codes.into_array(), values.into_array())
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

pub fn gen_fsst_test_data(len: usize, avg_str_len: usize, unique_chars: u8) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let mut strings = Vec::with_capacity(len);

    for _ in 0..len {
        // Generate a random string with length around `avg_len`. The number of possible
        // characters within the random string is defined by `unique_chars`.
        let len = avg_str_len * rng.gen_range(50..=150) / 100;
        strings.push(Some(
            (0..len)
                .map(|_| rng.gen_range(b'a'..(b'a' + unique_chars)))
                .collect::<Vec<u8>>(),
        ));
    }

    let varbin = VarBinArray::from_iter(
        strings
            .into_iter()
            .map(|opt_s| opt_s.map(Vec::into_boxed_slice)),
        DType::Binary(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin).vortex_unwrap();

    fsst_compress(&varbin, &compressor)
        .vortex_unwrap()
        .into_array()
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
        .map(|_| T::from(rng.gen_range(0..unique_values)).unwrap())
        .collect::<PrimitiveArray>();
    DictArray::try_new(codes.into_array(), values).vortex_unwrap()
}

pub fn gen_dict_primitive_chunks<T: NativePType, O: NativePType>(
    len: usize,
    unique_values: usize,
    chunk_count: usize,
) -> ArrayRef
where
    Standard: Distribution<T>,
{
    (0..chunk_count)
        .map(|_| {
            gen_primitive_dict::<T, O>(len, unique_values)
                .vortex_unwrap()
                .into_array()
        })
        .collect::<ChunkedArray>()
        .into_array()
}
