use rand::distributions::{Alphanumeric, Distribution, Standard};
use rand::prelude::{SliceRandom, StdRng};
use rand::Rng;
use vortex_array::array::{PrimitiveArray, VarBinArray};
use vortex_array::validity::Validity;
use vortex_array::{Array, IntoArray};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap};
use vortex_fsst::{fsst_compress, fsst_train_compressor};

use crate::DictArray;

pub fn gen_primitive_for_dict<T: NativePType>(
    rng: &mut StdRng,
    len: usize,
    unique_count: usize,
) -> PrimitiveArray
where
    Standard: Distribution<T>,
{
    let values = (0..unique_count)
        .map(|_| rng.gen::<T>())
        .collect::<Vec<T>>();
    let data = (0..len)
        .map(|_| *values.choose(rng).vortex_expect("not empty"))
        .collect::<Buffer<_>>();
    PrimitiveArray::new(data, Validity::NonNullable)
}

pub fn gen_primitive_dict<T: NativePType, O: NativePType>(
    rng: &mut StdRng,
    len: usize,
    unique_count: usize,
) -> VortexResult<DictArray>
where
    Standard: Distribution<T>,
{
    let values = (0..unique_count)
        .map(|_| rng.gen::<T>())
        .collect::<PrimitiveArray>();
    let codes = (0..len)
        .map(|_| O::from(rng.gen_range(0..unique_count)).vortex_expect("valid value"))
        .collect::<PrimitiveArray>();

    DictArray::try_new(codes.into_array(), values.into_array())
}

pub fn gen_varbin_words(rng: &mut StdRng, len: usize, unique_count: usize) -> Vec<String> {
    let dict: Vec<String> = (0..unique_count)
        .map(|_| {
            rng.sample_iter(&Alphanumeric)
                .take(16)
                .map(char::from)
                .collect()
        })
        .collect();
    (0..len)
        .map(|_| dict.choose(rng).vortex_expect("non empty").clone())
        .collect()
}

pub fn generate_fsst_test_data(
    rng: &mut StdRng,
    len: usize,
    avg_str_len: usize,
    unique_chars: u8,
) -> Array {
    let mut strings = Vec::with_capacity(len);

    for _ in 0..len {
        // Generate a random string with length around `avg_len`. The number of possible
        // characters within the random string is defined by `unique_chars`.
        let len = avg_str_len * rng.gen_range(50..=150) / 100;
        strings.push(Some(
            (0..len)
                .map(|_| rng.gen_range(b'a'..(b'a' + unique_chars)) as char)
                .collect::<String>()
                .into_bytes(),
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

pub fn generate_dict_fsst_test_data<O: NativePType>(
    rng: &mut StdRng,
    len: usize,
    unique_values: usize,
    str_len: usize,
    unique_char_count: u8,
) -> DictArray {
    let values = generate_fsst_test_data(rng, len, str_len, unique_char_count);
    let codes = (0..len)
        .map(|_| O::from(rng.gen_range(0..unique_values)).vortex_expect("valid value"))
        .collect::<PrimitiveArray>();
    DictArray::try_new(codes.into_array(), values).vortex_unwrap()
}
