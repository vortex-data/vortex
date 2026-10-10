// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_runend::compress::runend_decode_primitive;
use vortex_runend::compress::runend_decode_typed_primitive;
use vortex_runend::decompress_bool::runend_decode_bools;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

/// Distribution types for bool benchmarks
#[derive(Clone, Copy)]
enum BoolDistribution {
    /// Alternating true/false (50/50)
    Alternating,
    /// Mostly true (90% true runs)
    MostlyTrue,
    /// Mostly false (90% false runs)
    MostlyFalse,
    /// All true
    AllTrue,
    /// All false
    AllFalse,
}

impl fmt::Display for BoolDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoolDistribution::Alternating => write!(f, "alternating"),
            BoolDistribution::MostlyTrue => write!(f, "mostly_true"),
            BoolDistribution::MostlyFalse => write!(f, "mostly_false"),
            BoolDistribution::AllTrue => write!(f, "all_true"),
            BoolDistribution::AllFalse => write!(f, "all_false"),
        }
    }
}

#[derive(Clone, Copy)]
struct BoolBenchArgs {
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
}

impl fmt::Display for BoolBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}",
            self.total_length, self.avg_run_length, self.distribution
        )
    }
}

/// Creates bool test data with configurable distribution
fn create_bool_test_data(
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
) -> (PrimitiveArray, BoolArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);

        let val = match distribution {
            BoolDistribution::Alternating => run_index.is_multiple_of(2),
            BoolDistribution::MostlyTrue => !run_index.is_multiple_of(10), // 90% true
            BoolDistribution::MostlyFalse => run_index.is_multiple_of(10), // 10% true (90% false)
            BoolDistribution::AllTrue => true,
            BoolDistribution::AllFalse => false,
        };
        values.push(val);
        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::from(BitBuffer::from(values)),
    )
}

// Medium size: 10k elements with various run lengths and distributions
const BOOL_ARGS: &[BoolBenchArgs] = &[
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::AllFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::AllFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::AllFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::AllFalse,
    },
];

#[divan::bench(args = BOOL_ARGS)]
fn decode_bool(bencher: Bencher, args: BoolBenchArgs) {
    let BoolBenchArgs {
        total_length,
        avg_run_length,
        distribution,
    } = args;
    let (ends, values) = create_bool_test_data(total_length, avg_run_length, distribution);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_bools(ends.clone(), values.clone(), 0, total_length, ctx)
        });
}

/// Validity distribution for nullable benchmarks
#[derive(Clone, Copy)]
enum ValidityDistribution {
    /// 90% valid
    MostlyValid,
    /// 50% valid
    HalfValid,
    /// 10% valid
    MostlyNull,
}

impl fmt::Display for ValidityDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidityDistribution::MostlyValid => write!(f, "mostly_valid"),
            ValidityDistribution::HalfValid => write!(f, "half_valid"),
            ValidityDistribution::MostlyNull => write!(f, "mostly_null"),
        }
    }
}

#[derive(Clone, Copy)]
struct NullableBoolBenchArgs {
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
    validity: ValidityDistribution,
}

impl fmt::Display for NullableBoolBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}_{}",
            self.total_length, self.avg_run_length, self.distribution, self.validity
        )
    }
}

/// Creates nullable bool test data with configurable distribution and validity
fn create_nullable_bool_test_data(
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
    validity: ValidityDistribution,
) -> (PrimitiveArray, BoolArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = Vec::with_capacity(total_length / avg_run_length + 1);
    let mut validity_bits = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);

        let val = match distribution {
            BoolDistribution::Alternating => run_index.is_multiple_of(2),
            BoolDistribution::MostlyTrue => !run_index.is_multiple_of(10),
            BoolDistribution::MostlyFalse => run_index.is_multiple_of(10),
            BoolDistribution::AllTrue => true,
            BoolDistribution::AllFalse => false,
        };
        values.push(val);

        let is_valid = match validity {
            ValidityDistribution::MostlyValid => !run_index.is_multiple_of(10),
            ValidityDistribution::HalfValid => run_index.is_multiple_of(2),
            ValidityDistribution::MostlyNull => run_index.is_multiple_of(10),
        };
        validity_bits.push(is_valid);

        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::new(
            BitBuffer::from(values),
            Validity::from(BitBuffer::from(validity_bits)),
        ),
    )
}

const NULLABLE_BOOL_ARGS: &[NullableBoolBenchArgs] = &[
    // Alternating with different validity
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::HalfValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyNull,
    },
    // MostlyTrue with different validity
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
        validity: ValidityDistribution::HalfValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
        validity: ValidityDistribution::MostlyNull,
    },
    // Different run lengths with MostlyValid
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
];

#[divan::bench(args = NULLABLE_BOOL_ARGS)]
fn decode_bool_nullable(bencher: Bencher, args: NullableBoolBenchArgs) {
    let NullableBoolBenchArgs {
        total_length,
        avg_run_length,
        distribution,
        validity,
    } = args;
    let (ends, values) =
        create_nullable_bool_test_data(total_length, avg_run_length, distribution, validity);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_bools(ends.clone(), values.clone(), 0, total_length, ctx)
        });
}

#[derive(Clone, Copy)]
struct PrimitiveBenchArgs {
    run_length: usize,
    valid_percent: usize,
    validity_offset: usize,
}

impl fmt::Display for PrimitiveBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "run_{}_valid_{}_offset_{}",
            self.run_length, self.valid_percent, self.validity_offset
        )
    }
}

const PRIMITIVE_ARGS: [PrimitiveBenchArgs; 30] = {
    let mut args = [PrimitiveBenchArgs {
        run_length: 1,
        valid_percent: 99,
        validity_offset: 0,
    }; 30];
    let lengths = [1, 2, 16, 128, 4096];
    let validities = [10, 50, 99];
    let mut i = 0;
    while i < args.len() {
        args[i] = PrimitiveBenchArgs {
            run_length: lengths[i / 6],
            valid_percent: validities[(i / 2) % 3],
            validity_offset: (i % 2) * 3,
        };
        i += 1;
    }
    args
};

#[divan::bench(types = [u16, i64], args = PRIMITIVE_ARGS)]
fn decode_primitive_nullable<T: NativePType>(bencher: Bencher, args: PrimitiveBenchArgs) {
    let length = 65_536usize;
    let run_count = length.div_ceil(args.run_length);
    let values: Vec<T> = (0..run_count)
        .map(|i| T::from_usize(i % 1024).vortex_expect("benchmark value fits"))
        .collect();
    let validity = BitBuffer::from_iter(
        (0..run_count + args.validity_offset).map(|i| (i * 37) % 100 < args.valid_percent),
    )
    .slice(args.validity_offset..run_count + args.validity_offset);
    let validity = Mask::from(validity);
    bencher.bench(|| {
        runend_decode_typed_primitive(
            (1..=run_count).map(|i| (i * args.run_length).min(length)),
            &values,
            validity.clone(),
            Nullability::Nullable,
            length,
        )
    });
}

#[divan::bench(types = [u16, i64], args = [1usize, 2, 16, 128, 4096])]
fn decode_primitive_non_nullable<T: NativePType>(bencher: Bencher, run_length: usize) {
    let length = 65_536usize;
    let run_count = length.div_ceil(run_length);
    let values: Vec<T> = (0..run_count)
        .map(|i| T::from_usize(i % 1024).vortex_expect("benchmark value fits"))
        .collect();
    bencher.bench(|| {
        runend_decode_typed_primitive(
            (1..=run_count).map(|i| (i * run_length).min(length)),
            &values,
            Mask::AllTrue(run_count),
            Nullability::NonNullable,
            length,
        )
    });
}

#[divan::bench(types = [u16, i64], args = [1usize, 2, 16, 128, 4096])]
fn decode_primitive_array<T: NativePType>(bencher: Bencher, run_length: usize) {
    let length = 65_536usize;
    let run_count = length.div_ceil(run_length);
    let ends =
        PrimitiveArray::from_iter((1..=run_count).map(|i| (i * run_length).min(length) as u32));
    let values = PrimitiveArray::new(
        Buffer::from_iter(
            (0..run_count).map(|i| T::from_usize(i % 1024).vortex_expect("benchmark value fits")),
        ),
        Validity::NonNullable,
    );
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            runend_decode_primitive(ends.clone(), values.clone(), 0, length, ctx)
                .vortex_expect("benchmark decode succeeds")
        });
}
