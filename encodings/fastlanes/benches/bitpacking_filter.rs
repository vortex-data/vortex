// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures selective filtering around the sparse extraction thresholds.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPackedData;
use vortex_mask::Mask;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

const NUM_ARRAY_CHUNKS: usize = 64;
// Keep the array density below the outer full-decode policy.
const NUM_SELECTED_CHUNKS: usize = 8;
const CHUNK_SIZE: usize = 1_024;
const LEN: usize = NUM_ARRAY_CHUNKS * CHUNK_SIZE;

trait BenchInt: NativePType {
    fn from_counter(value: u64) -> Self;
}

macro_rules! impl_bench_int {
    ($($T:ty),+) => {
        $(impl BenchInt for $T {
            fn from_counter(value: u64) -> Self {
                value as $T
            }
        })+
    };
}

impl_bench_int!(u8, u16, u32, u64);

fn fixture<T: BenchInt>(bit_width: usize, selected_per_chunk: usize) -> (ArrayRef, Mask) {
    let limit = if bit_width == 64 {
        u64::MAX
    } else {
        1_u64 << bit_width
    };
    let values: BufferMut<T> = (0..LEN)
        .map(|index| T::from_counter(index as u64 % limit))
        .collect();
    let packed = BitPackedData::encode(
        &PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
        bit_width as u8,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap()
    .into_array();
    let indices = (0..NUM_SELECTED_CHUNKS).flat_map(|chunk| {
        (0..selected_per_chunk)
            .map(move |index| chunk * CHUNK_SIZE + index * CHUNK_SIZE / selected_per_chunk)
    });
    (packed, Mask::from_indices(LEN, indices))
}

macro_rules! bench_width {
    ($module:ident, $T:ty, $bit_width:expr, [$($selected:expr),+ $(,)?]) => {
        mod $module {
            use super::*;

            #[vortex_bench_support::cpu_features]
            #[divan::bench(args = [$($selected),+])]
            fn filter(bencher: Bencher, selected_per_chunk: usize) {
                let (packed, mask) = fixture::<$T>($bit_width, selected_per_chunk);
                bencher
                    .counter(ItemsCount::new(LEN))
                    .with_inputs(|| (mask.clone(), SESSION.create_execution_ctx()))
                    .bench_refs(|(mask, ctx)| {
                        packed
                            .filter(mask.clone())
                            .unwrap()
                            .execute::<PrimitiveArray>(ctx)
                            .unwrap()
                    });
            }
        }
    };
}

macro_rules! bench_type {
    ($module:ident, $T:ty, [$(($width_module:ident, $bit_width:expr)),+ $(,)?], $selected:tt) => {
        mod $module {
            use super::*;

            $(bench_width!($width_module, $T, $bit_width, $selected);)+
        }
    };
}

bench_type!(u8, u8, [(width1, 1), (width4, 4), (width7, 7)], [8, 16, 24]);
bench_type!(
    u16,
    u16,
    [(width1, 1), (width8, 8), (width15, 15)],
    [8, 32, 48]
);
bench_type!(
    u32,
    u32,
    [(width1, 1), (width16, 16), (width31, 31)],
    [8, 64, 80, 96]
);
bench_type!(
    u64,
    u64,
    [(width1, 1), (width32, 32), (width63, 63)],
    [8, 128, 160, 192]
);
