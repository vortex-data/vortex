// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use mimalloc::MiMalloc;
use vortex_array::compute::warm_up_vtables;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    warm_up_vtables();
    divan::main();
}

// TODO(ngates): bring back benchmarks once operator API is stable.
// #[divan::bench(types = [i8, i16, i32, i64], args = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])]
// pub fn aligned_step_kernel<T>(bencher: Bencher, fraction_kept: f64)
// where
//     T: NativePType + Element,
//     T: PhysicalPType,
//     T::Physical: fastlanes::BitPacking + Element,
// {
//     let mut rng = StdRng::seed_from_u64(0);
//     let values = (0..N)
//         .map(|_| T::from(rng.random_range(0..127)).unwrap())
//         .collect::<PrimitiveArray>();
//     let array = bitpack_to_best_bit_width(&values).unwrap();
//
//     // Create the aligned kernel - offset = 0
//     let packed_stride = array.bit_width() as usize
//         * <<T as PhysicalPType>::Physical as fastlanes::FastLanes>::LANES;
//     let buffer = Buffer::<<T as PhysicalPType>::Physical>::from_byte_buffer(
//         array.packed().clone().into_byte_buffer(),
//     );
//     let kernel = BitPackedKernel::<T>::new(array.bit_width() as usize, packed_stride, buffer, 0);
//
//     let mask = (0..N)
//         .map(|_| rng.random_bool(fraction_kept))
//         .collect::<BooleanBuffer>();
//     let mut mask_data = [0usize; N_WORDS];
//     for (i, chunk) in mask.bit_chunks().iter().enumerate() {
//         if i < N_WORDS {
//             mask_data[i] = usize::try_from(chunk).unwrap();
//         }
//     }
//
//     // Create mask with all true values to test maximum unpacking
//     let ctx = KernelContext::default();
//     let mut output_data = vec![T::default(); N];
//     let mut output = ViewMut::new(&mut output_data, None);
//
//     bencher
//         .with_inputs(|| (BitView::new(&mask_data), kernel.clone()))
//         .bench_local_values(|(bit_view, mut kernel)| {
//             kernel.step(&ctx, bit_view, &mut output).unwrap()
//         });
// }

// #[divan::bench(types = [i8, i16, i32, i64], args = [(8, 0.01), (512, 0.01), (8, 0.05), (512, 0.05), (8, 0.1), (512, 0.1), (8, 0.3), (512, 0.3), (8, 0.5), (512, 0.5), (8, 0.7), (512, 0.7), (8, 0.9), (512, 0.9), (8, 1.0), (512, 1.0)])]
// pub fn unaligned_step_kernel<T>(bencher: Bencher, (offset, fraction_kept): (usize, f64))
// where
//     T: NativePType + Element,
//     T: PhysicalPType,
//     T::Physical: fastlanes::BitPacking + Element,
// {
//     let mut rng = StdRng::seed_from_u64(0);
//     let values = (0..N + offset)
//         .map(|_| T::from(rng.random_range(0..127)).unwrap())
//         .collect::<PrimitiveArray>();
//     let array = bitpack_to_best_bit_width(&values).unwrap();
//
//     let packed_stride = array.bit_width() as usize
//         * <<T as PhysicalPType>::Physical as fastlanes::FastLanes>::LANES;
//     let buffer = Buffer::<<T as PhysicalPType>::Physical>::from_byte_buffer(
//         array.packed().clone().into_byte_buffer(),
//     );
//     let kernel = BitPackedUnalignedKernel::<T>::new(
//         array.bit_width() as usize,
//         packed_stride,
//         buffer,
//         0,
//         offset.try_into().unwrap(),
//     );
//
//     let mask = (0..N)
//         .map(|_| rng.random_bool(fraction_kept))
//         .collect::<Mask>();
//
//     let expect = filter(&array.as_ref().slice(offset..offset + N), &mask)
//         .unwrap()
//         .to_primitive();
//
//     let mut mask_data = [0usize; N_WORDS];
//     for (i, chunk) in mask.to_bit_buffer().chunks().iter().enumerate() {
//         if i < N_WORDS {
//             mask_data[i] = usize::try_from(chunk).unwrap();
//         }
//     }
//     let ctx = KernelContext::default();
//     let mut output_data = vec![T::default(); N];
//     let mut output = ViewMut::new(&mut output_data, None);
//
//     bencher
//         .with_inputs(|| (BitView::new(&mask_data), kernel.clone()))
//         .bench_local_values(|(bit_view, mut kernel)| {
//             kernel.step(&ctx, bit_view, &mut output).unwrap();
//
//             assert_eq!(
//                 output.as_slice::<T>()[..mask.true_count()],
//                 *expect.as_slice::<T>()
//             );
//         });
// }
