// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use std::fmt;
use std::ops::Deref;

use divan::Bencher;
#[cfg(not(codspeed))]
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::Rng;
use rand::SeedableRng;
use vortex::array::ArrayRef;
use vortex::array::DynArray;
use vortex::array::IntoArray;
use vortex::array::ToCanonical;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::vtable::ValidityHelper;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::encodings::alp::alp_encode;
use vortex::encodings::datetime_parts::DateTimePartsArray;
use vortex::encodings::datetime_parts::split_temporal;
use vortex::encodings::fastlanes::FoRArray;
use vortex::encodings::fsst::FSSTArray;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;
use vortex::encodings::runend::RunEndArray;
use vortex::extension::datetime::TimeUnit;
use vortex_fastlanes::BitPackedArray;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const NUM_VALUES: u64 = 100_000;

// Helper function to conditionally add counter based on codspeed cfg
fn with_byte_counter<'a, 'b>(bencher: Bencher<'a, 'b>, bytes: u64) -> Bencher<'a, 'b> {
    #[cfg(not(codspeed))]
    return bencher.counter(BytesCount::new(bytes));
    #[cfg(codspeed)]
    {
        _ = bytes; // Consume the bytes value to avoid unused variable warning.
        return bencher;
    }
}

// Encoding tree setup functions

mod setup {
    use rand::rngs::StdRng;

    use super::*;

    fn setup_primitive_arrays() -> (PrimitiveArray, PrimitiveArray, PrimitiveArray) {
        let mut rng = StdRng::seed_from_u64(0);
        let uint_array =
            PrimitiveArray::from_iter((0..NUM_VALUES).map(|_| rng.random_range(42u32..256)));
        let int_array = uint_array
            .clone()
            .into_array()
            .cast(PType::I32.into())
            .unwrap()
            .to_primitive();
        let float_array = uint_array
            .clone()
            .into_array()
            .cast(PType::F64.into())
            .unwrap()
            .to_primitive();
        (uint_array, int_array, float_array)
    }

    /// Create FoR <- BitPacked encoding tree for u64
    pub fn for_bp_u64() -> ArrayRef {
        let (uint_array, ..) = setup_primitive_arrays();
        let compressed = FoRArray::encode(uint_array).unwrap();
        let inner = compressed.encoded();
        let bp = BitPackedArray::encode(inner, 8).unwrap();
        FoRArray::try_new(bp.into_array(), compressed.reference_scalar().clone())
            .unwrap()
            .into_array()
    }

    /// Create ALP <- FoR <- BitPacked encoding tree for f64
    pub fn alp_for_bp_f64() -> ArrayRef {
        let (_, _, float_array) = setup_primitive_arrays();
        let alp_compressed = alp_encode(&float_array, None).unwrap();

        // Manually construct ALP <- FoR <- BitPacked tree
        let for_array = FoRArray::encode(alp_compressed.encoded().to_primitive()).unwrap();
        let inner = for_array.encoded();
        let bp = BitPackedArray::encode(inner, 8).unwrap();
        let for_with_bp =
            FoRArray::try_new(bp.into_array(), for_array.reference_scalar().clone()).unwrap();

        vortex::encodings::alp::ALPArray::try_new(
            for_with_bp.into_array(),
            alp_compressed.exponents(),
            alp_compressed.patches().cloned(),
        )
        .unwrap()
        .into_array()
    }

    /// Create Dict <- VarBinView encoding tree for strings with BitPacked codes
    #[allow(clippy::cast_possible_truncation)]
    pub fn dict_varbinview_string() -> ArrayRef {
        let mut rng = StdRng::seed_from_u64(42);

        // Create unique values (0.005% uniqueness = 50 unique strings)
        let num_unique = ((NUM_VALUES as f64) * 0.00005) as usize;
        let unique_strings: Vec<String> = (0..num_unique)
            .map(|_| {
                (0..8)
                    .map(|_| (rng.random_range(b'a'..=b'z')) as char)
                    .collect()
            })
            .collect();

        // Create codes array (random indices into unique values)
        let codes: Vec<u32> = (0..NUM_VALUES)
            .map(|_| rng.random_range(0..num_unique as u32))
            .collect();
        let codes_prim = PrimitiveArray::from_iter(codes);

        // Compress codes with BitPacked (6 bits should be enough for ~50 unique values)
        let codes_bp = BitPackedArray::encode(&codes_prim.into_array(), 6)
            .unwrap()
            .into_array();

        // Create values array
        let values_array = VarBinViewArray::from_iter_str(unique_strings).into_array();

        DictArray::try_new(codes_bp, values_array)
            .unwrap()
            .into_array()
    }

    /// Create RunEnd <- FoR <- BitPacked encoding tree for u32
    #[allow(clippy::cast_possible_truncation)]
    pub fn runend_for_bp_u32() -> ArrayRef {
        let mut rng = StdRng::seed_from_u64(42);
        // Create data with runs of repeated values
        let mut values = Vec::with_capacity(NUM_VALUES as usize);
        let mut current_value = rng.random_range(0u32..100);
        let mut run_length = 0;

        for _ in 0..NUM_VALUES {
            if run_length == 0 {
                current_value = rng.random_range(0u32..100);
                run_length = rng.random_range(1..1000);
            }
            values.push(current_value);
            run_length -= 1;
        }

        let prim_array = PrimitiveArray::from_iter(values);
        let runend = RunEndArray::encode(prim_array.into_array()).unwrap();

        // Compress the ends with FoR <- BitPacked
        let ends_prim = runend.ends().to_primitive();
        let ends_for = FoRArray::encode(ends_prim).unwrap();
        let ends_inner = ends_for.encoded();
        let ends_bp = BitPackedArray::encode(ends_inner, 8).unwrap();
        let compressed_ends =
            FoRArray::try_new(ends_bp.into_array(), ends_for.reference_scalar().clone())
                .unwrap()
                .into_array();

        // Compress the values with BitPacked
        let values_prim = runend.values().to_primitive();
        let compressed_values = BitPackedArray::encode(&values_prim.into_array(), 8)
            .unwrap()
            .into_array();

        RunEndArray::try_new(compressed_ends, compressed_values)
            .unwrap()
            .into_array()
    }

    /// Create Dict <- FSST <- VarBin encoding tree for strings
    #[allow(clippy::cast_possible_truncation)]
    pub fn dict_fsst_varbin_string() -> ArrayRef {
        let mut rng = StdRng::seed_from_u64(43);

        // Create unique values (1% uniqueness = 10,000 unique strings)
        let num_unique = ((NUM_VALUES as f64) * 0.01) as usize;
        let unique_strings: Vec<String> = (0..num_unique)
            .map(|_| {
                (0..8)
                    .map(|_| (rng.random_range(b'a'..=b'z')) as char)
                    .collect()
            })
            .collect();

        // Train and compress unique values with FSST
        let unique_varbinview = VarBinViewArray::from_iter_str(unique_strings);
        let fsst_compressor = fsst_train_compressor(&unique_varbinview);
        let fsst_values = fsst_compress(&unique_varbinview, &fsst_compressor);

        // Create codes array (random indices into unique values)
        let codes: Vec<u32> = (0..NUM_VALUES)
            .map(|_| rng.random_range(0..num_unique as u32))
            .collect();
        let codes_array = PrimitiveArray::from_iter(codes).into_array();

        DictArray::try_new(codes_array, fsst_values.into_array())
            .unwrap()
            .into_array()
    }

    /// Create Dict <- FSST <- VarBin <- BitPacked encoding tree for strings
    /// Compress the VarBin offsets inside FSST with BitPacked
    #[allow(clippy::cast_possible_truncation)]
    pub fn dict_fsst_varbin_bp_string() -> ArrayRef {
        let mut rng = StdRng::seed_from_u64(45);

        // Create unique values (1% uniqueness = 10,000 unique strings)
        let num_unique = ((NUM_VALUES as f64) * 0.01) as usize;
        let unique_strings: Vec<String> = (0..num_unique)
            .map(|_| {
                (0..8)
                    .map(|_| (rng.random_range(b'a'..=b'z')) as char)
                    .collect()
            })
            .collect();

        // Train and compress unique values with FSST
        let unique_varbinview = VarBinViewArray::from_iter_str(unique_strings);
        let fsst_compressor = fsst_train_compressor(&unique_varbinview);
        let fsst = fsst_compress(&unique_varbinview, &fsst_compressor);

        // Compress the VarBin offsets with BitPacked
        let codes = fsst.codes();
        let offsets_prim = codes.offsets().to_primitive();
        let offsets_bp = BitPackedArray::encode(&offsets_prim.into_array(), 20).unwrap();

        // Rebuild VarBin with compressed offsets
        let compressed_codes = VarBinArray::try_new(
            offsets_bp.into_array(),
            codes.bytes().clone(),
            codes.dtype().clone(),
            codes.validity().clone(),
        )
        .unwrap();

        // Rebuild FSST with compressed codes
        let compressed_fsst = FSSTArray::try_new(
            fsst.dtype().clone(),
            fsst.symbols().clone(),
            fsst.symbol_lengths().clone(),
            compressed_codes,
            fsst.uncompressed_lengths().clone(),
        )
        .unwrap();

        // Create codes array (random indices into unique values)
        let dict_codes: Vec<u32> = (0..NUM_VALUES)
            .map(|_| rng.random_range(0..num_unique as u32))
            .collect();
        let codes_array = PrimitiveArray::from_iter(dict_codes).into_array();

        DictArray::try_new(codes_array, compressed_fsst.into_array())
            .unwrap()
            .into_array()
    }

    /// Create DateTimeParts <- FoR <- BitPacked encoding tree
    pub fn datetime_for_bp() -> ArrayRef {
        // Create timestamp data (microseconds since epoch)
        let mut rng = StdRng::seed_from_u64(123);
        let base_timestamp = 1_600_000_000_000_000i64; // Sept 2020 in microseconds
        let timestamps: Vec<i64> = (0..NUM_VALUES)
            .map(|_| base_timestamp + rng.random_range(0..86_400_000_000)) // Random times within a day
            .collect();

        let ts_array = PrimitiveArray::from_iter(timestamps).into_array();

        // Create TemporalArray with microsecond timestamps
        let temporal_array = TemporalArray::new_timestamp(ts_array, TimeUnit::Microseconds, None);

        // Split into days, seconds, subseconds
        let parts = split_temporal(temporal_array.clone()).unwrap();

        // Compress days with FoR <- BitPacked
        let days_prim = parts.days.to_primitive();
        let days_for = FoRArray::encode(days_prim).unwrap();
        let days_inner = days_for.encoded();
        let days_bp = BitPackedArray::encode(days_inner, 16).unwrap();
        let compressed_days =
            FoRArray::try_new(days_bp.into_array(), days_for.reference_scalar().clone())
                .unwrap()
                .into_array();

        // Compress seconds with FoR <- BitPacked
        let seconds_prim = parts.seconds.to_primitive();
        let seconds_for = FoRArray::encode(seconds_prim).unwrap();
        let seconds_inner = seconds_for.encoded();
        let seconds_bp = BitPackedArray::encode(seconds_inner, 17).unwrap();
        let compressed_seconds = FoRArray::try_new(
            seconds_bp.into_array(),
            seconds_for.reference_scalar().clone(),
        )
        .unwrap()
        .into_array();

        // Compress subseconds with FoR <- BitPacked
        let subseconds_prim = parts.subseconds.to_primitive();
        let subseconds_for = FoRArray::encode(subseconds_prim).unwrap();
        let subseconds_inner = subseconds_for.encoded();
        let subseconds_bp = BitPackedArray::encode(subseconds_inner, 20).unwrap();
        let compressed_subseconds = FoRArray::try_new(
            subseconds_bp.into_array(),
            subseconds_for.reference_scalar().clone(),
        )
        .unwrap()
        .into_array();

        DateTimePartsArray::try_new(
            DType::Extension(temporal_array.ext_dtype()),
            compressed_days,
            compressed_seconds,
            compressed_subseconds,
        )
        .unwrap()
        .into_array()
    }
}

// Complex encoding tree benchmarks

#[derive(Copy, Clone)]
struct SetupFn {
    func: fn() -> ArrayRef,
    name: &'static str,
}

impl SetupFn {
    const fn new(func: fn() -> ArrayRef, name: &'static str) -> Self {
        Self { func, name }
    }
}

impl Deref for SetupFn {
    type Target = fn() -> ArrayRef;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl fmt::Display for SetupFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Macro to construct the `SetupFn` wrapper.
macro_rules! setup_fn {
    ($func:path) => {
        // Stringify and split off the function name.
        // E.g.: `setup::for_bp_u64` => "for_bp_u64"
        SetupFn::new($func, stringify!($func).split("::").last().unwrap())
    };
}

/// Benchmark decompression of various encoding trees
#[divan::bench(
    args = [
        setup_fn!(setup::for_bp_u64),
        setup_fn!(setup::alp_for_bp_f64),
        setup_fn!(setup::dict_varbinview_string),
        setup_fn!(setup::runend_for_bp_u32),
        setup_fn!(setup::dict_fsst_varbin_string),
        setup_fn!(setup::dict_fsst_varbin_bp_string),
        setup_fn!(setup::datetime_for_bp),
    ]
)]
fn decompress(bencher: Bencher, setup_fn: SetupFn) {
    let compressed = setup_fn();
    let nbytes = compressed.nbytes();

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}
