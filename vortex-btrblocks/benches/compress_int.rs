// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end compression of 4 MiB integer arrays, the uncompressed size of a writer chunk.

#![expect(clippy::unwrap_used)]

use mimalloc::MiMalloc;

#[cfg(not(codspeed))]
mod benchmarks {
    use std::sync::LazyLock;

    use divan::Bencher;
    use divan::counter::BytesCount;
    use num_traits::AsPrimitive;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_btrblocks::BtrBlocksCompressorBuilder;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// The uncompressed size of each array, matching a writer chunk.
    const BYTES: usize = 4 << 20;

    #[derive(Debug, Copy, Clone)]
    enum Distribution {
        Constant,
        LowCardinality,
        ShortRuns,
        LongRuns,
        Sequence,
        SmallRange,
        WideRandom,
    }

    impl Distribution {
        fn name(self) -> &'static str {
            match self {
                Self::Constant => "Constant",
                Self::LowCardinality => "LowCardinality",
                Self::ShortRuns => "ShortRuns",
                Self::LongRuns => "LongRuns",
                Self::Sequence => "Sequence",
                Self::SmallRange => "SmallRange",
                Self::WideRandom => "WideRandom",
            }
        }
    }

    const DISTRIBUTIONS: [Distribution; 7] = [
        Distribution::Constant,
        Distribution::LowCardinality,
        Distribution::ShortRuns,
        Distribution::LongRuns,
        Distribution::Sequence,
        Distribution::SmallRange,
        Distribution::WideRandom,
    ];

    /// A seeded xorshift generator, so that every build compresses the same data.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }
    }

    fn runs(len: usize, max_run: u32, distinct: u32) -> Vec<u32> {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut output = Vec::with_capacity(len);
        while output.len() < len {
            let value = rng.next() % distinct;
            let run = (rng.next() % max_run).max(1) as usize;
            output.extend(std::iter::repeat_n(value, run.min(len - output.len())));
        }
        output
    }

    fn generate<T>(distribution: Distribution) -> Buffer<T>
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        let len = BYTES / size_of::<T>();
        let len32 = u32::try_from(len).unwrap();
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let values: Vec<u32> = match distribution {
            Distribution::Constant => vec![7; len],
            Distribution::LowCardinality => (0..len32).map(|i| (i * 7919) % 50).collect(),
            Distribution::ShortRuns => runs(len, 4, 1024),
            Distribution::LongRuns => runs(len, 512, 1024),
            Distribution::Sequence => (0..len32).map(|i| i.wrapping_mul(3)).collect(),
            Distribution::SmallRange => (0..len).map(|_| 1000 + rng.next() % 200).collect(),
            Distribution::WideRandom => (0..len).map(|_| rng.next() % 1_000_000).collect(),
        };
        values.into_iter().map(|v| v.as_()).collect()
    }

    fn bench_compress<T>(bencher: Bencher, distribution: Distribution, nullable: bool)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        let values = generate::<T>(distribution);
        let validity = if nullable {
            let mut rng = Rng(0xD1B5_4A32_D192_ED03);
            Validity::from(BitBuffer::from_iter(
                (0..values.len()).map(|_| !rng.next().is_multiple_of(10)),
            ))
        } else {
            Validity::NonNullable
        };
        // This session registers no encodings, so allow every scheme, as a writer's session that
        // registers every encoding does.
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();

        // Record the compressed size once, to check that every build compresses alike.
        let array = PrimitiveArray::new(values.clone(), validity.clone()).into_array();
        let compressed = compressor
            .compress(&array, &mut SESSION.create_execution_ctx())
            .unwrap();
        eprintln!(
            "SIZE {} {} {} {}",
            std::any::type_name::<T>(),
            distribution.name(),
            nullable,
            compressed.nbytes()
        );

        // A fresh array per iteration, so statistics cached on the array by a previous
        // iteration are not reused.
        bencher
            .counter(BytesCount::new(BYTES))
            .with_inputs(|| {
                (
                    PrimitiveArray::new(values.clone(), validity.clone()).into_array(),
                    SESSION.create_execution_ctx(),
                )
            })
            .bench_refs(|(array, ctx)| compressor.compress(array, ctx).unwrap());
    }

    #[divan::bench(types = [u8, u16, u32, u64, i32, i64], args = DISTRIBUTIONS)]
    fn compress<T>(bencher: Bencher, distribution: Distribution)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        bench_compress::<T>(bencher, distribution, false);
    }

    #[divan::bench(types = [u8, u16, u32, u64, i32, i64], args = DISTRIBUTIONS)]
    fn compress_nullable<T>(bencher: Bencher, distribution: Distribution)
    where
        T: NativePType,
        u32: AsPrimitive<T>,
    {
        bench_compress::<T>(bencher, distribution, true);
    }
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main()
}
