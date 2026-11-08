// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the arrays the compressor builds with RLE against those it builds with RunEnd.
//!
//! Each input is an `i32` column of runs with random 30-bit values. The `without_runend`
//! variant compresses it with RunEnd excluded, so RLE is the run encoding available to the
//! compressor; `without_rle` excludes RLE instead. Both variants keep every other scheme, so the
//! benchmarks measure the trees a writer would actually produce, children included.
//!
//! Run lengths are drawn uniformly from `1..2 * RUN_LEN`, which models the irregular runs of real
//! data. Fixed-length runs make RunEnd decoding branch-predictable and flatter it.

#![expect(clippy::unwrap_used)]

use mimalloc::MiMalloc;

#[cfg(not(codspeed))]
mod benchmarks {
    use std::fmt;
    use std::sync::LazyLock;

    use divan::Bencher;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::session::AggregateFnSession;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::memory::MemorySession;
    use vortex_array::optimizer::kernels::KernelSession;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_btrblocks::BtrBlocksCompressorBuilder;
    use vortex_btrblocks::CompressionSession;
    use vortex_btrblocks::SchemeExt;
    use vortex_btrblocks::schemes::integer::IntRLEScheme;
    use vortex_btrblocks::schemes::integer::RunEndScheme;
    use vortex_buffer::Buffer;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    pub(super) static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        VortexSession::empty()
            .with::<ArraySession>()
            .with::<KernelSession>()
            .with::<AggregateFnSession>()
            .with::<MemorySession>()
            .with::<CompressionSession>()
    });

    const LEN: usize = 65_536;

    /// Average run lengths. RLE and RunEnd are both only considered from an average of 4.
    const RUN_LENS: &[usize] = &[4, 8, 16, 64, 256];

    #[derive(Debug, Clone, Copy)]
    enum Variant {
        WithoutRunEnd,
        WithoutRle,
    }

    impl fmt::Display for Variant {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::WithoutRunEnd => write!(f, "without_runend"),
                Self::WithoutRle => write!(f, "without_rle"),
            }
        }
    }

    const VARIANTS: &[Variant] = &[Variant::WithoutRunEnd, Variant::WithoutRle];

    fn runs(run_len: usize) -> PrimitiveArray {
        let mut rng = StdRng::seed_from_u64(run_len as u64);
        let mut values = Vec::with_capacity(LEN);
        while values.len() < LEN {
            let len = rng.random_range(1..2 * run_len);
            let value = rng.random_range(0..1i32 << 30);
            values.extend(std::iter::repeat_n(value, len));
        }
        values.truncate(LEN);
        PrimitiveArray::new(Buffer::from(values), Validity::NonNullable)
    }

    fn compressed(run_len: usize, variant: Variant) -> ArrayRef {
        let excluded = match variant {
            Variant::WithoutRunEnd => RunEndScheme.id(),
            Variant::WithoutRle => IntRLEScheme.id(),
        };
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .exclude_schemes([excluded])
            .build();
        compressor
            .compress(
                &runs(run_len).into_array(),
                &mut SESSION.create_execution_ctx(),
            )
            .unwrap()
    }

    fn random_mask(selectivity: f64) -> Mask {
        let mut rng = StdRng::seed_from_u64(7);
        Mask::from_iter((0..LEN).map(|_| rng.random_bool(selectivity)))
    }

    #[divan::bench(args = VARIANTS, consts = RUN_LENS)]
    fn decode<const RUN_LEN: usize>(bencher: Bencher, variant: Variant) {
        let array = compressed(RUN_LEN, variant);
        bencher
            .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
            .bench_values(|(array, mut ctx)| array.execute::<Canonical>(&mut ctx).unwrap());
    }

    #[divan::bench(args = VARIANTS, consts = RUN_LENS)]
    fn filter_1pct<const RUN_LEN: usize>(bencher: Bencher, variant: Variant) {
        let array = compressed(RUN_LEN, variant);
        let mask = random_mask(0.01);
        bencher
            .with_inputs(|| (array.clone(), mask.clone(), SESSION.create_execution_ctx()))
            .bench_values(|(array, mask, mut ctx)| {
                array
                    .filter(mask)
                    .unwrap()
                    .execute::<Canonical>(&mut ctx)
                    .unwrap()
            });
    }

    #[divan::bench(args = VARIANTS, consts = RUN_LENS)]
    fn take_1pct<const RUN_LEN: usize>(bencher: Bencher, variant: Variant) {
        let array = compressed(RUN_LEN, variant);
        let mut rng = StdRng::seed_from_u64(11);
        let indices: ArrayRef = (0..LEN / 100)
            .map(|_| rng.random_range(0..u32::try_from(LEN).unwrap()))
            .collect::<Buffer<u32>>()
            .into_array();
        bencher
            .with_inputs(|| {
                (
                    array.clone(),
                    indices.clone(),
                    SESSION.create_execution_ctx(),
                )
            })
            .bench_values(|(array, indices, mut ctx)| {
                array
                    .take(indices)
                    .unwrap()
                    .execute::<Canonical>(&mut ctx)
                    .unwrap()
            });
    }
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    #[cfg(not(codspeed))]
    std::sync::LazyLock::force(&benchmarks::SESSION);
    divan::main()
}
