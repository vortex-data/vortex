use rand::distributions::{Alphanumeric, Distribution, Standard};
use rand::prelude::{SliceRandom, StdRng};
use rand::Rng;
use vortex_array::array::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::IntoArray;
use vortex_buffer::Buffer;
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexResult};

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
