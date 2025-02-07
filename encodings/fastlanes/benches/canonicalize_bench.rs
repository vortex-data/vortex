use divan::Bencher;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::array::ChunkedArray;
use vortex_array::builders::{ArrayBuilder, PrimitiveBuilder};
use vortex_array::{Array, IntoArray, IntoArrayVariant, IntoCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_fastlanes::bitpack_to_best_bit_width;

fn main() {
    divan::main();
}

fn make_array<T: NativePType>(len: usize) -> Array {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..len)
        .map(|_| T::from(rng.gen_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .into_primitive()
        .unwrap();

    bitpack_to_best_bit_width(values).unwrap().into_array()
}

#[divan::bench()]
fn test() {
    let chunks = (0..10).map(|_| make_array::<i32>(100)).collect::<Vec<_>>();
    let arr = make_array::<i32>(1);
    let chunked = ChunkedArray::try_new(chunks, arr.dtype().clone())
        .unwrap()
        .into_array();

    let into_ca = chunked
        .clone()
        .into_canonical()
        .unwrap()
        .into_primitive()
        .unwrap();
    let mut primitive_builder =
        PrimitiveBuilder::<i32>::with_capacity(arr.dtype().nullability(), 10 * 100);
    chunked
        .clone()
        .canonicalize_into(&mut primitive_builder)
        .unwrap();
    let ca_into = primitive_builder.finish().unwrap();

    assert_eq!(
        into_ca.as_slice::<i32>(),
        ca_into.into_primitive().unwrap().as_slice::<i32>()
    );

    let mut primitive_builder =
        PrimitiveBuilder::<i32>::with_capacity(arr.dtype().nullability(), 10 * 100);
    primitive_builder.extend_from_array(chunked).unwrap();
    let ca_into = primitive_builder.finish().unwrap();

    assert_eq!(
        into_ca.as_slice::<i32>(),
        ca_into.into_primitive().unwrap().as_slice::<i32>()
    );
}

#[divan::bench(
    types = [u32],
    args = [
        // (1000, 100),
        // (100000, 100),
        // (1000000, 100),
        // (100000, 1000),
        (100000, 3),
    ]
)]
fn into_canonical<T: NativePType>(bencher: Bencher, (arr_len, chunk_count): (usize, usize)) {
    let chunks = (0..chunk_count)
        .map(|_| make_array::<T>(arr_len))
        .collect::<Vec<_>>();
    let arr = make_array::<T>(1);
    let chunked = ChunkedArray::try_new(chunks, arr.dtype().clone()).unwrap();
    // println!("chunked: {}", chunked.tree_display());

    bencher.bench(|| chunked.clone().into_canonical().unwrap().len());
}

#[divan::bench(
    types = [u32],
    args = [
        // (1000, 100),
        // (100000, 100),
        // (1000000, 100),
        // (100000, 1000),
        (100000, 3),
    ]
)]
fn canonical_into<T: NativePType>(bencher: Bencher, (arr_len, chunk_count): (usize, usize)) {
    let chunks = (0..chunk_count)
        .map(|_| make_array::<T>(arr_len))
        .collect::<Vec<_>>();
    let arr = make_array::<T>(1);
    let chunked = ChunkedArray::try_new(chunks, arr.dtype().clone())
        .unwrap()
        .into_array();

    // println!("chunked: {}", chunked.tree_display());

    bencher.bench(|| {
        let mut primitive_builder =
            PrimitiveBuilder::<T>::with_capacity(arr.dtype().nullability(), arr_len * chunk_count);
        chunked
            .clone()
            .canonicalize_into(&mut primitive_builder)
            .unwrap();
        primitive_builder.finish().unwrap().len()
    });
}
