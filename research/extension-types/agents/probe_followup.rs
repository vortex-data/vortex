// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Throwaway follow-up probes for the type research on `develop` at `5fc5cb8`.
//!
//! This file is not part of any crate, so nothing compiles it. To run it, copy it to
//! `vortex-btrblocks/tests/research_probe.rs` and run:
//!
//! ```text
//! cargo test -p vortex-btrblocks --test research_probe -- --nocapture --test-threads=1
//! ```
//!
//! Delete the copy afterwards. Appendix A of `findings.md` records the output of the last run.
//! "Array path" means `execute::<RecursiveCanonical>` on the lazy array, then a read of each row.
//! "Row path" means `execute_scalar` on each row of the lazy array.

#![allow(clippy::all, clippy::disallowed_methods, unused)]

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::expr::ext_storage;
use vortex_array::expr::root;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_btrblocks::CascadingCompressor;
use vortex_btrblocks::Scheme;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_buffer::buffer;
use vortex_datetime_parts::DateTimeParts;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

/// The array session with the `DateTimeParts` and `Sparse` encodings and their kernels registered.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_datetime_parts::initialize(&session);
    vortex_sparse::initialize(&session);
    session
});

/// u64 values where zero is not a valid value.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct NonZero;

impl ExtVTable for NonZero {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = u64;

    fn id(&self) -> ExtId {
        ExtId::new("scratch.non_zero")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(matches!(
            ext_dtype.storage_dtype(),
            DType::Primitive(PType::U64, _)
        ));
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let v = storage_value.as_primitive().cast::<u64>()?;
        if v == 0 {
            vortex_bail!("zero is not a valid NonZero value");
        }
        Ok(v)
    }
}

/// An extension type with no meaningful order (think "opaque handle").
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct Opaque;

impl ExtVTable for Opaque {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = i64;

    fn id(&self) -> ExtId {
        ExtId::new("scratch.opaque")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(matches!(
            ext_dtype.storage_dtype(),
            DType::Primitive(PType::I64, _)
        ));
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        storage_value.as_primitive().cast::<i64>()
    }
}

fn report<T: std::fmt::Debug>(name: &str, f: impl FnOnce() -> VortexResult<T>) {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => println!("[{name}] OK: {v:?}"),
        Ok(Err(e)) => println!(
            "[{name}] ERR: {}",
            e.to_string().lines().next().unwrap_or_default()
        ),
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            println!(
                "[{name}] PANIC: {}",
                msg.lines().next().unwrap_or_default()
            )
        }
    }
}

fn ts(unit: TimeUnit, nullability: Nullability) -> ExtDTypeRef {
    Timestamp::new(unit, nullability).erased()
}

fn opaque() -> ExtDTypeRef {
    ExtDType::<Opaque>::try_new(
        EmptyMetadata,
        DType::Primitive(PType::I64, Nullability::NonNullable),
    )
    .unwrap()
    .erased()
}

fn non_zero() -> ExtDTypeRef {
    ExtDType::<NonZero>::try_new(
        EmptyMetadata,
        DType::Primitive(PType::U64, Nullability::NonNullable),
    )
    .unwrap()
    .erased()
}

/// Reads every row with `execute_scalar` (the row path).
fn rows(a: &ArrayRef) -> VortexResult<String> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut out = vec![];
    for i in 0..a.len() {
        out.push(a.execute_scalar(i, &mut ctx)?.to_string());
    }
    Ok(format!("{} [{}]", a.dtype(), out.join(", ")))
}

/// Executes the whole array through the array kernels, then reads the stored values.
fn array(a: &ArrayRef) -> VortexResult<String> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = a
        .clone()
        .execute::<RecursiveCanonical>(&mut ctx)?
        .0
        .into_array();
    rows(&canonical)
}

/// The encoding tree of an array, on one line.
fn tree(a: &ArrayRef) -> String {
    a.display_tree_encodings_only()
        .to_string()
        .lines()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
fn r1_decimal_to_f32() {
    let dec = || {
        DecimalArray::new(
            buffer![100i64, 250],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        )
        .into_array()
    };
    let f32_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);

    report("R1 array path: Decimal(10, 2) [1.00, 2.50] -> f32", || {
        array(&dec().cast(f32_dtype.clone())?)
    });
    report("R1 row path: Decimal(10, 2) [1.00, 2.50] -> f32", || {
        rows(&dec().cast(f32_dtype.clone())?)
    });
    report("R1 scalar path: Decimal(10, 2) 1.00 -> f32", || {
        let s = Scalar::decimal(
            DecimalValue::I64(100),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );
        Ok(s.cast(&f32_dtype)?.to_string())
    });
}

#[test]
fn r2_scalar_traits() {
    let ms = ts(TimeUnit::Milliseconds, Nullability::NonNullable);
    let a = Scalar::extension_ref(opaque(), Scalar::from(1i64));
    let b = Scalar::extension_ref(opaque(), Scalar::from(2i64));

    report("R2 Opaque(1).partial_cmp(Opaque(2))", || Ok(a.partial_cmp(&b)));
    report("R2 Opaque(1) == Opaque(1), and partial_cmp", || {
        Ok((a == a.clone(), a.partial_cmp(&a)))
    });

    let f64_ts = || {
        Scalar::try_new(
            DType::Extension(ms.clone()),
            Some(ScalarValue::Primitive(PValue::F64(5.0))),
        )
    };
    let i64_ts = Scalar::extension_ref(ms.clone(), Scalar::from(5i64));

    report("R2 try_new(Timestamp(ms), F64 5.0)", || {
        Ok(f64_ts()?.to_string())
    });
    report("R2 Timestamp(ms) I64 5 == Timestamp(ms) F64 5.0", || {
        Ok(i64_ts == f64_ts()?)
    });
    report("R2 Timestamp(ms) F64 5.0 == Timestamp(ms) I64 5", || {
        Ok(f64_ts()? == i64_ts)
    });
    report("R2 Timestamp(ms) I64 5 partial_cmp F64 5.0", || {
        Ok(i64_ts.partial_cmp(&f64_ts()?))
    });
}

#[test]
fn r3_float_identity() {
    report("R3 Scalar NaN == NaN", || {
        Ok(Scalar::from(f64::NAN) == Scalar::from(f64::NAN))
    });
    report("R3 Scalar -0.0 == +0.0", || {
        Ok(Scalar::from(-0.0f64) == Scalar::from(0.0f64))
    });
    report("R3 array path: [NaN, -0.0] = [NaN, +0.0]", || {
        let l = buffer![f64::NAN, -0.0].into_array();
        let r = buffer![f64::NAN, 0.0].into_array();
        array(&l.binary(r, Operator::Eq)?)
    });
    report(
        "R3 array path: FSL<f32, 2> [[NaN, -0.0], [NaN, 1.0]] = [[NaN, +0.0], [NaN, 1.0]]",
        || {
            let l = FixedSizeListArray::try_new(
                buffer![f32::NAN, -0.0, f32::NAN, 1.0].into_array(),
                2,
                Validity::NonNullable,
                2,
            )?
            .into_array();
            let r = FixedSizeListArray::try_new(
                buffer![f32::NAN, 0.0, f32::NAN, 1.0].into_array(),
                2,
                Validity::NonNullable,
                2,
            )?
            .into_array();
            array(&l.binary(r, Operator::Eq)?)
        },
    );
}

#[test]
fn r4_non_zero_check() {
    report("R4 Scalar::try_new(NonZero, 0u64)", || {
        Ok(Scalar::try_new(
            DType::Extension(non_zero()),
            Some(ScalarValue::Primitive(PValue::U64(0))),
        )?
        .to_string())
    });
}

#[test]
fn r5_constant_detection() {
    let ms = ts(TimeUnit::Milliseconds, Nullability::NonNullable);
    let ext_const = ExtensionArray::new(
        ms,
        ConstantArray::new(Scalar::from(1000i64), 3).into_array(),
    )
    .into_array();

    report("R5 Extension(Constant): tree and as_constant()", || {
        Ok((tree(&ext_const), ext_const.as_constant().map(|s| s.to_string())))
    });
    report("R5 after optimize(): tree and as_constant()", || {
        let optimized = ext_const.optimize()?;
        Ok((
            tree(&optimized),
            optimized.as_constant().map(|s| s.to_string()),
        ))
    });
}

#[test]
fn r6_storage_extraction() {
    let codes = || buffer![0u8, 1, 0, 1].into_array();
    let values = || buffer![10i64, 20].into_array();

    let one_step = |input: ArrayRef| -> VortexResult<String> {
        let mut ctx = SESSION.create_execution_ctx();
        let lazy = input.apply(&ext_storage(root()))?;
        let stepped = lazy.clone().execute::<ArrayRef>(&mut ctx)?;
        Ok(format!("lazy: {} || one step: {}", tree(&lazy), tree(&stepped)))
    };

    report("R6 ext_storage(Extension(Dict))", || {
        let input = ExtensionArray::new(
            opaque(),
            DictArray::try_new(codes(), values())?.into_array(),
        )
        .into_array();
        one_step(input)
    });
    report("R6 ext_storage(Dict(codes, Extension(values)))", || {
        let input = DictArray::try_new(
            codes(),
            ExtensionArray::new(opaque(), values()).into_array(),
        )?
        .into_array();
        one_step(input)
    });
    report("R6 ext_storage(Chunked[Extension, Extension])", || {
        let chunk = || ExtensionArray::new(opaque(), values()).into_array();
        let input = ChunkedArray::try_new([chunk(), chunk()], DType::Extension(opaque()))?
            .into_array();
        one_step(input)
    });
    report("R6 ext_storage(Masked(Extension))", || {
        let child = ExtensionArray::new(opaque(), buffer![10i64, 20, 10, 20].into_array())
            .into_array();
        let validity =
            Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array());
        one_step(MaskedArray::try_new(child, validity)?.into_array())
    });
    report("R6 ext_storage(Sparse with extension values)", || {
        let input = Sparse::try_new(
            buffer![1u64, 3].into_array(),
            ExtensionArray::new(opaque(), values()).into_array(),
            4,
            Scalar::extension_ref(opaque(), Scalar::from(0i64)),
        )?
        .into_array();
        one_step(input)
    });
}

#[test]
fn r7_datetime_parts_compare() {
    let ms = || ts(TimeUnit::Milliseconds, Nullability::NonNullable);
    let ns = ts(TimeUnit::Nanoseconds, Nullability::NonNullable);

    // 1.6 s and 1.4 s, so `< 1.5 s` is [false, true].
    let column = || -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let temporal = TemporalArray::new_timestamp(
            buffer![1600i64, 1400].into_array(),
            TimeUnit::Milliseconds,
            None,
        );
        Ok(DateTimeParts::try_from_temporal(temporal, &mut ctx)?.into_array())
    };
    let i64_const = || ConstantArray::new(Scalar::from(1500i64), 2).into_array();
    let ns_const = || {
        ConstantArray::new(
            Scalar::extension_ref(ns.clone(), Scalar::from(1_500_000_000i64)),
            2,
        )
        .into_array()
    };
    let ms_const = || {
        ConstantArray::new(Scalar::extension_ref(ms(), Scalar::from(1500i64)), 2).into_array()
    };

    report("R7 column tree", || Ok(tree(&column()?)));
    report("R7 array path: DateTimeParts(ms) < i64 1500", || {
        array(&column()?.binary(i64_const(), Operator::Lt)?)
    });
    report("R7 row path: DateTimeParts(ms) < i64 1500", || {
        rows(&column()?.binary(i64_const(), Operator::Lt)?)
    });
    report("R7 array path: DateTimeParts(ms) < Timestamp(ns) 1.5 s", || {
        array(&column()?.binary(ns_const(), Operator::Lt)?)
    });
    report("R7 row path: DateTimeParts(ms) < Timestamp(ns) 1.5 s", || {
        rows(&column()?.binary(ns_const(), Operator::Lt)?)
    });
    report("R7 array path (control): DateTimeParts(ms) < Timestamp(ms) 1500", || {
        array(&column()?.binary(ms_const(), Operator::Lt)?)
    });
}

#[test]
fn r8_compressor_shapes() {
    let compressor = CascadingCompressor::new(vec![&IntDictScheme as &'static dyn Scheme]);
    let compress = |input: ArrayRef| -> VortexResult<String> {
        let mut ctx = SESSION.create_execution_ctx();
        Ok(tree(&compressor.compress(&input, &mut ctx)?))
    };

    report("R8 constant Timestamp(ms) column", || {
        compress(
            ExtensionArray::new(
                ts(TimeUnit::Milliseconds, Nullability::NonNullable),
                buffer![1000i64; 64].into_array(),
            )
            .into_array(),
        )
    });
    report("R8 low-cardinality Opaque column", || {
        let values: Vec<i64> = (0..300).map(|i| [10i64, 20, 30][i % 3]).collect();
        compress(
            ExtensionArray::new(opaque(), PrimitiveArray::from_iter(values).into_array())
                .into_array(),
        )
    });
    report("R8 partly null constant Timestamp(ms) column", || {
        let values = (0..64).map(|i| (i % 4 != 0).then_some(1000i64));
        compress(
            ExtensionArray::new(
                ts(TimeUnit::Milliseconds, Nullability::Nullable),
                PrimitiveArray::from_option_iter(values).into_array(),
            )
            .into_array(),
        )
    });
}

#[test]
fn r10_f16_backcompat() {
    let f16_dtype = DType::Primitive(PType::F16, Nullability::NonNullable);
    let one = vortex_array::dtype::half::f16::from_f32(1.0);
    let bits_form = || {
        Scalar::try_new(
            f16_dtype.clone(),
            Some(ScalarValue::Primitive(PValue::U64(u64::from(one.to_bits())))),
        )
    };
    let native_form = Scalar::primitive(one, Nullability::NonNullable);

    report("R10 try_new(F16, U64 bits of 1.0)", || {
        Ok(bits_form()?.to_string())
    });
    report("R10 U64 form == F16 form", || Ok(bits_form()? == native_form));
    report("R10 F16 form == U64 form", || Ok(native_form == bits_form()?));
}

#[test]
fn r9_ext_storage_paths() {
    let input =
        ExtensionArray::new(non_zero(), buffer![0u64, 1].into_array()).into_array();

    report("R9 lazy ext_storage(NonZero [0, 1]) tree", || {
        Ok(tree(&input.clone().apply(&ext_storage(root()))?))
    });
    report("R9 array path: ext_storage(NonZero [0, 1])", || {
        array(&input.clone().apply(&ext_storage(root()))?)
    });
    report("R9 row path: ext_storage(NonZero [0, 1])", || {
        rows(&input.clone().apply(&ext_storage(root()))?)
    });
}
