// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use mimalloc::MiMalloc;

#[cfg(not(codspeed))]
mod benchmarks {
    use std::sync::LazyLock;

    use divan::Bencher;
    use divan::counter::BytesCount;
    use divan::counter::ItemsCount;
    use rand::Rng;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::session::AggregateFnSession;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::memory::MemorySession;
    use vortex_array::optimizer::kernels::KernelSession;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_btrblocks::BtrBlocksCompressorBuilder;
    use vortex_btrblocks::CompressionSession;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer_mut;
    use vortex_edition::EDITION_DECLARATIONS;
    use vortex_edition::EDITION_FAMILIES;
    use vortex_edition::EditionSession;
    use vortex_edition::EditionSessionExt;
    use vortex_edition::declarations::core::CORE_2026_08_3;
    use vortex_session::VortexSession;
    use vortex_utils::aliases::hash_set::HashSet;

    pub(super) static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        VortexSession::empty()
            .with::<ArraySession>()
            .with::<KernelSession>()
            .with::<AggregateFnSession>()
            .with::<MemorySession>()
            .with::<CompressionSession>()
    });

    /// A session that allows only what a default writer may produce, set up as the golden tests
    /// set it up: the encodings the compressor emits, limited by the default `core` edition.
    static WRITE_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<EditionSession>();
        vortex_alp::initialize(&session);
        vortex_datetime_parts::initialize(&session);
        vortex_decimal_byte_parts::initialize(&session);
        vortex_fastlanes::initialize(&session);
        vortex_fsst::initialize(&session);
        vortex_onpair::initialize(&session);
        vortex_sequence::initialize(&session);
        vortex_sparse::initialize(&session);
        vortex_zigzag::initialize(&session);
        for family in EDITION_FAMILIES {
            session.editions().declare_family(family).unwrap();
        }
        for declaration in EDITION_DECLARATIONS {
            session.register_edition(declaration).unwrap();
        }
        session.enable_edition(CORE_2026_08_3).unwrap();
        session
    });

    fn make_clickbench_window_name() -> ArrayRef {
        // A test that's meant to mirror the WindowName column from ClickBench.
        let mut values = buffer_mut![-1i32; 65_536];
        let mut visited = HashSet::new();
        let mut rng = StdRng::seed_from_u64(1u64);
        while visited.len() < 223 {
            let random = (rng.next_u32() as usize) % 65_536;
            if visited.contains(&random) {
                continue;
            }
            visited.insert(random);
            // Pick 100 random values to insert.
            values[random] = 5 * (rng.next_u64() % 100) as i32;
        }

        // Ok, now let's compress
        values.freeze().into_array()
    }

    #[divan::bench]
    fn btrblocks(bencher: Bencher) {
        let mut ctx = SESSION.create_execution_ctx();
        let array = make_clickbench_window_name()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();
        bencher
            .with_inputs(|| (&array, SESSION.create_execution_ctx()))
            .input_counter(|(array, _)| ItemsCount::new(array.len()))
            .input_counter(|(array, _)| BytesCount::of_many::<i32>(array.len()))
            .bench_refs(|(array, ctx)| {
                compressor
                    .compress(&array.clone().into_array(), ctx)
                    .unwrap()
            });
    }

    /// Compresses a canonical `Decimal(12, 2)` column whose values are drawn like those of the
    /// golden `decimal_prices` column.
    #[divan::bench]
    fn btrblocks_decimal(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(115);
        let values: Buffer<i64> = (0..65_536)
            .map(|_| rng.random_range(0..10_000_000i64))
            .collect();
        // NB: the default `core` edition does not permit `vortex.decimal_byte_parts.v2`, so the
        // decimal scheme refines to its v1 variant, as in a default write. An unrestricted
        // compressor would refine to v2 instead.
        let compressor = BtrBlocksCompressorBuilder::from_session(&WRITE_SESSION).build();
        bencher
            .with_inputs(|| {
                (
                    DecimalArray::new(
                        values.clone(),
                        DecimalDType::new(12, 2),
                        Validity::NonNullable,
                    )
                    .into_array(),
                    WRITE_SESSION.create_execution_ctx(),
                )
            })
            .input_counter(|(array, _)| ItemsCount::new(array.len()))
            .input_counter(|(array, _)| BytesCount::of_many::<i64>(array.len()))
            .bench_refs(|(array, ctx)| compressor.compress(array, ctx).unwrap());
    }
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    #[cfg(not(codspeed))]
    std::sync::LazyLock::force(&benchmarks::SESSION);
    divan::main()
}
