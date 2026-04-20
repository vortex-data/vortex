// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Alphanumeric;
use rand::distr::Distribution;
use rand::distr::StandardUniform;
use rand::prelude::IndexedRandom;
use rand::prelude::StdRng;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::validity::Validity;

pub fn gen_primitive_for_dict<T: NativePType>(len: usize, unique_count: usize) -> PrimitiveArray
where
    StandardUniform: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..unique_count)
        .map(|_| rng.random::<T>())
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
    StandardUniform: Distribution<V>,
{
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..unique_count)
        .map(|_| rng.random::<V>())
        .collect::<PrimitiveArray>();

    let codes = (0..len)
        .map(|_| C::from(rng.random_range(0..unique_count)).unwrap())
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

pub fn gen_dict_primitive_chunks<T: NativePType, O: NativePType>(
    len: usize,
    unique_values: usize,
    chunk_count: usize,
) -> ArrayRef
where
    StandardUniform: Distribution<T>,
{
    (0..chunk_count)
        .map(|_| {
            gen_primitive_dict::<T, O>(len, unique_values)
                .vortex_expect("operation should succeed in test")
                .into_array()
        })
        .collect::<ChunkedArray>()
        .into_array()
}
