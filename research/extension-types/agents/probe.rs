// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Throwaway probes for extension type behavior on `develop` at `5fc5cb8`.
//!
//! This file is not part of any crate, so nothing compiles it. To run it, copy it into
//! `vortex-array/tests/` and run:
//!
//! ```text
//! cargo test -p vortex-array --test probe -- --nocapture
//! ```
//!
//! The build needs several GB of disk space. Appendix A of `findings.md` records the output of the
//! last run. The `probes` test reads rows with `execute_scalar` (the row path). The
//! `probes_array_path` test executes whole arrays with `RecursiveCanonical` (the array path).

#![allow(clippy::all, clippy::disallowed_methods, unused)]

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::EmptyArrayData;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::extension::ExtensionSlots;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::ExtensionBuilder;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

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

/// An extension type that only accepts non-nullable storage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct NonNullableOnly;

impl ExtVTable for NonNullableOnly {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = u64;

    fn id(&self) -> ExtId {
        ExtId::new("scratch.non_nullable_only")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(
            ext_dtype.storage_dtype() == &DType::Primitive(PType::U64, Nullability::NonNullable)
        );
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        storage_value.as_primitive().cast::<u64>()
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

fn ts(unit: TimeUnit) -> ExtDTypeRef {
    Timestamp::new(unit, Nullability::NonNullable).erased()
}

fn ts_array(unit: TimeUnit, values: ArrayRef) -> ArrayRef {
    ExtensionArray::new(ts(unit), values).into_array()
}

fn display(a: &ArrayRef) -> VortexResult<String> {
    let mut ctx = array_session().create_execution_ctx();
    let mut out = vec![];
    for i in 0..a.len() {
        out.push(a.execute_scalar(i, &mut ctx)?.to_string());
    }
    Ok(format!("{} [{}]", a.dtype(), out.join(", ")))
}

/// Fully materialize through the array kernels, then read the stored values.
fn display_array(a: &ArrayRef) -> VortexResult<String> {
    let mut ctx = array_session().create_execution_ctx();
    let canonical = a
        .clone()
        .execute::<vortex_array::RecursiveCanonical>(&mut ctx)?
        .0
        .into_array();
    display(&canonical)
}

#[test]
fn probes_array_path() {
    report("B' array path: PrimitiveArray i64 -> ts(ms)", || {
        let a = buffer![1000i64, 2000].into_array();
        display_array(&a.cast(DType::Extension(ts(TimeUnit::Milliseconds)))?)
    });

    report("D1' array path: ts(ms) -> ts(ns)", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        display_array(&a.cast(DType::Extension(ts(TimeUnit::Nanoseconds)))?)
    });

    report("D2' array path: constant ts(ms) -> ts(ns)", || {
        let a = ConstantArray::new(
            Scalar::extension_ref(ts(TimeUnit::Milliseconds), Scalar::from(1000i64)),
            2,
        )
        .into_array();
        display_array(&a.cast(DType::Extension(ts(TimeUnit::Nanoseconds)))?)
    });

    report("E1' array path: ts(ms) -> i64", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        display_array(&a.cast(DType::Primitive(PType::I64, Nullability::NonNullable))?)
    });

    report("E4' array path: Opaque -> NonZero", || {
        let opaque = ExtDType::<Opaque>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .erased();
        let nz = ExtDType::<NonZero>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?
        .erased();
        let a = ExtensionArray::new(opaque, buffer![1i64, 2].into_array()).into_array();
        display_array(&a.cast(DType::Extension(nz))?)
    });

    report("E5' array path: i64 [0] -> NonZero (invalid value)", || {
        let nz = ExtDType::<NonZero>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?
        .erased();
        let a = buffer![0i64, 1].into_array();
        display_array(&a.cast(DType::Extension(nz))?)
    });

    report("F1' array path: ts(ms) array < i64 array", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        let b = buffer![1500i64, 1500].into_array();
        display_array(&a.binary(b, Operator::Lt)?)
    });

    report("F2' array path: ts(ms) array < i64 constant", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        let b = ConstantArray::new(Scalar::from(1500i64), 2).into_array();
        display_array(&a.binary(b, Operator::Lt)?)
    });

    report("F3' array path: ts(ms) array < utf8 array", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        let b = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
        display_array(&a.binary(b, Operator::Lt)?)
    });

    report("F5' array path: ts(ms) array < ts(ns) array", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        let b = ts_array(TimeUnit::Nanoseconds, buffer![1500i64, 1500].into_array());
        display_array(&a.binary(b, Operator::Lt)?)
    });
}

#[test]
fn probes() {
    let session = array_session();

    // --- Casts --------------------------------------------------------------------------------

    report("A scalar cast ts(ms)=1000 -> ts(ns)", || {
        let s = Scalar::extension_ref(ts(TimeUnit::Milliseconds), Scalar::from(1000i64));
        let c = s.cast(&DType::Extension(ts(TimeUnit::Nanoseconds)))?;
        Ok(format!("{s} => {c}"))
    });

    report("B array cast PrimitiveArray i64 -> ts(ms)", || {
        let a = buffer![1000i64, 2000].into_array();
        let c = a.cast(DType::Extension(ts(TimeUnit::Milliseconds)))?;
        display(&c)
    });

    report("C constant cast i64 -> ts(ms)", || {
        let a = ConstantArray::new(Scalar::from(1000i64), 2).into_array();
        let c = a.cast(DType::Extension(ts(TimeUnit::Milliseconds)))?;
        display(&c)
    });

    report("D1 array cast ts(ms) -> ts(ns) (non-constant)", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        let c = a.cast(DType::Extension(ts(TimeUnit::Nanoseconds)))?;
        display(&c)
    });

    report("D2 array cast ts(ms) -> ts(ns) (constant)", || {
        let a = ConstantArray::new(
            Scalar::extension_ref(ts(TimeUnit::Milliseconds), Scalar::from(1000i64)),
            2,
        )
        .into_array();
        let c = a.cast(DType::Extension(ts(TimeUnit::Nanoseconds)))?;
        display(&c)
    });

    report("E1 array cast ts(ms) -> i64", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        display(&a.cast(DType::Primitive(PType::I64, Nullability::NonNullable))?)
    });

    report("E2 array cast ts(ms) -> f64", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        display(&a.cast(DType::Primitive(PType::F64, Nullability::NonNullable))?)
    });

    report("E3 array cast ts(ms) -> u8 (narrowing)", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1i64, 2].into_array());
        display(&a.cast(DType::Primitive(PType::U8, Nullability::NonNullable))?)
    });

    report("E4 array cast Opaque -> NonZero (different ext ids)", || {
        let opaque = ExtDType::<Opaque>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .erased();
        let nz = ExtDType::<NonZero>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?
        .erased();
        let a = ExtensionArray::new(opaque, buffer![1i64, 2].into_array()).into_array();
        display(&a.cast(DType::Extension(nz))?)
    });

    // --- Comparisons --------------------------------------------------------------------------

    report("F1 compare ts(ms) array < i64 array", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        let b = buffer![1500i64, 1500].into_array();
        display(&a.binary(b, Operator::Lt)?)
    });

    report("F2 compare ts(ms) array < i64 constant", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        let b = ConstantArray::new(Scalar::from(1500i64), 2).into_array();
        display(&a.binary(b, Operator::Lt)?)
    });

    report("F3 compare ts(ms) array < utf8 array (typing)", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        let b = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
        // Building the expression succeeds or fails here; executing it is in display().
        let r = a.binary(b, Operator::Lt)?;
        display(&r)
    });

    report("F4 between ts(ms) with i64 bounds", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1000i64, 2000].into_array());
        let lo = ConstantArray::new(Scalar::from(0i64), 2).into_array();
        let hi = ConstantArray::new(Scalar::from(1500i64), 2).into_array();
        display(&a.between(
            lo,
            hi,
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?)
    });

    report("G compare Opaque < Opaque (no opt-in needed)", || {
        let dt = ExtDType::<Opaque>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .erased();
        let a = ExtensionArray::new(dt.clone(), buffer![1i64, 5].into_array()).into_array();
        let b = ExtensionArray::new(dt, buffer![3i64, 3].into_array()).into_array();
        display(&a.binary(b, Operator::Lt)?)
    });

    // --- Aggregates ---------------------------------------------------------------------------

    report("H min_max Opaque (no opt-in needed)", || {
        let dt = ExtDType::<Opaque>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .erased();
        let a = ExtensionArray::new(dt, buffer![5i64, 1, 3].into_array()).into_array();
        let mut ctx = session.create_execution_ctx();
        let r = min_max(&a, &mut ctx, NumericalAggregateOpts::default())?;
        Ok(r.map(|r| format!("min={} max={}", r.min, r.max)))
    });

    report("I is_sorted on sorted ts(ms)", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1i64, 2, 3].into_array());
        let mut ctx = session.create_execution_ctx();
        is_sorted(&a, &mut ctx)
    });

    // --- Invalid values from "zero" / "default" paths -----------------------------------------

    let nz = ExtDType::<NonZero>::try_new(
        EmptyMetadata,
        DType::Primitive(PType::U64, Nullability::NonNullable),
    )
    .unwrap()
    .erased();

    report("K1 Scalar::zero_value(NonZero)", || {
        let z = Scalar::zero_value(&DType::Extension(nz.clone()));
        Ok(format!("created {z}"))
    });

    report("K2 Scalar::try_new(NonZero, zero_value)", || {
        let z = Scalar::zero_value(&DType::Extension(nz.clone()));
        let (dtype, value) = z.into_parts();
        Ok(Scalar::try_new(dtype, value)?.to_string())
    });

    report("L1 ExtensionBuilder(NonZero).append_zeros(1).finish()", || {
        let mut b = ExtensionBuilder::new_in(nz.clone(), BufferAllocatorRef::static_ref());
        b.append_zeros(1);
        let a = b.finish();
        Ok(format!("built array of len {}", a.len()))
    });

    report("L2 ... then execute_scalar(0)", || {
        let mut b = ExtensionBuilder::new_in(nz.clone(), BufferAllocatorRef::static_ref());
        b.append_zeros(1);
        let a = b.finish();
        display(&a)
    });

    report("L3 ExtensionArray::new(NonZero, [0]) (no value validation)", || {
        let a = ExtensionArray::new(nz.clone(), buffer![0u64].into_array()).into_array();
        Ok(format!("built array of len {}", a.len()))
    });

    // --- Nullability changes ------------------------------------------------------------------

    let nn = ExtDType::<NonNullableOnly>::try_new(
        EmptyMetadata,
        DType::Primitive(PType::U64, Nullability::NonNullable),
    )
    .unwrap()
    .erased();

    report("M1 take(NonNullableOnly, nullable indices)", || {
        let a = ExtensionArray::new(nn.clone(), buffer![1u64, 2].into_array()).into_array();
        let idx = PrimitiveArray::from_option_iter([Some(0u32), None]).into_array();
        display(&a.take(idx)?)
    });

    report("M2 with_nullability(NonNullableOnly, Nullable)", || {
        Ok(nn.with_nullability(Nullability::Nullable).to_string())
    });

    // --- Misc ---------------------------------------------------------------------------------

    report("N Debug of ExtDTypeRef", || Ok(format!("{:?}", ts(TimeUnit::Milliseconds))));

    report("O try_from_parts with utf8 storage under ts(ms)", || {
        let storage = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
        let parts = ArrayParts::new(
            Extension,
            DType::Extension(ts(TimeUnit::Milliseconds)),
            2,
            EmptyArrayData,
            ExtensionSlots { storage }.into_slots(),
        );
        let a = ExtensionArray::try_from_parts(parts)?;
        Ok(format!("accepted: dtype={} storage dtype={}", a.dtype(), a.storage_array().dtype()))
    });

    report("P arithmetic ts + ts", || {
        let a = ts_array(TimeUnit::Milliseconds, buffer![1i64, 2].into_array());
        let b = ts_array(TimeUnit::Milliseconds, buffer![1i64, 2].into_array());
        display(&a.binary(b, Operator::Add)?)
    });

    report("Q Scalar::is_zero on ts(ms)=0", || {
        let s = Scalar::extension_ref(ts(TimeUnit::Milliseconds), Scalar::from(0i64));
        Ok(s.is_zero())
    });
}
