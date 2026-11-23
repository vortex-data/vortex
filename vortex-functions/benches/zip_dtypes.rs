// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "benchmark fixtures use indices that fit in the chosen widths"
)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_functions::builtins::ArrayFunctions;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<KernelSession>()
        .with::<MemorySession>();
    vortex_array::initialize(&session);
    vortex_functions::initialize(&session);
    session
});

const LEN: usize = 1_024;
const FSL_WIDTH: u32 = 4;

fn mask() -> Mask {
    Mask::from_iter((0..LEN).map(|i| i.is_multiple_of(2)))
}

#[divan::bench]
fn decimal_nonnull(bencher: Bencher) {
    run(
        bencher,
        decimal_array(0, false),
        decimal_array(1_000_000, false),
    );
}

#[divan::bench]
fn decimal_nullable(bencher: Bencher) {
    run(
        bencher,
        decimal_array(0, true),
        decimal_array(1_000_000, true),
    );
}

#[divan::bench]
fn extension_nonnull(bencher: Bencher) {
    run(
        bencher,
        extension_array(0, false),
        extension_array(1_000_000, false),
    );
}

#[divan::bench]
fn extension_nullable(bencher: Bencher) {
    run(
        bencher,
        extension_array(0, true),
        extension_array(1_000_000, true),
    );
}

#[divan::bench]
fn fixed_size_list_nonnull(bencher: Bencher) {
    run(
        bencher,
        fixed_size_list_array(0, false),
        fixed_size_list_array(1_000_000, false),
    );
}

#[divan::bench]
fn fixed_size_list_nullable(bencher: Bencher) {
    run(
        bencher,
        fixed_size_list_array(0, true),
        fixed_size_list_array(1_000_000, true),
    );
}

#[divan::bench]
fn map_nonnull(bencher: Bencher) {
    run(bencher, map_array(0, false), map_array(1_000_000, false));
}

#[divan::bench]
fn map_nullable(bencher: Bencher) {
    run(bencher, map_array(0, true), map_array(1_000_000, true));
}

#[divan::bench]
fn null(bencher: Bencher) {
    run(
        bencher,
        NullArray::new(LEN).into_array(),
        NullArray::new(LEN).into_array(),
    );
}

fn run(bencher: Bencher, if_true: ArrayRef, if_false: ArrayRef) {
    mark_optimized(&if_true, &SESSION).unwrap();
    mark_optimized(&if_false, &SESSION).unwrap();

    let mask = mask();
    bencher
        .with_inputs(|| {
            let mask = mask.clone().into_array();
            mark_optimized(&mask, &SESSION).unwrap();

            (
                if_true.clone(),
                if_false.clone(),
                mask,
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(t, f, m, ctx)| {
            m.zip(t.clone(), f.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap();
        });
}

fn validity(nullable: bool) -> Validity {
    if nullable {
        Validity::Array(BoolArray::from_iter((0..LEN).map(|i| !i.is_multiple_of(7))).into_array())
    } else {
        Validity::NonNullable
    }
}

fn decimal_array(base: i64, nullable: bool) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let mut values = BufferMut::<i64>::with_capacity(LEN);
    values.extend((0..LEN as i64).map(|i| base + i));
    DecimalArray::new(
        values.freeze(),
        DecimalDType::new(18, 2),
        validity(nullable),
        &mut ctx,
    )
    .into_array()
}

fn extension_array(base: i32, nullable: bool) -> ArrayRef {
    let mut values = BufferMut::<i32>::with_capacity(LEN);
    values.extend((0..LEN as i32).map(|i| base + i));
    let storage = PrimitiveArray::new(values.freeze(), validity(nullable)).into_array();
    let nullability = if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let ext_dtype = Date::try_new(TimeUnit::Days, nullability).unwrap().erased();
    ExtensionArray::new(ext_dtype, storage).into_array()
}

fn fixed_size_list_array(base: i64, nullable: bool) -> ArrayRef {
    let elem_len = LEN * FSL_WIDTH as usize;
    let mut elements = BufferMut::<i64>::with_capacity(elem_len);
    elements.extend((0..elem_len as i64).map(|i| base + i));
    let elements = PrimitiveArray::new(elements.freeze(), Validity::NonNullable).into_array();
    FixedSizeListArray::try_new(elements, FSL_WIDTH, validity(nullable), LEN)
        .unwrap()
        .into_array()
}

fn map_array(base: i64, nullable: bool) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let mut keys = BufferMut::<i64>::with_capacity(LEN);
    let mut values = BufferMut::<i64>::with_capacity(LEN);
    keys.extend(0..LEN as i64);
    values.extend((0..LEN as i64).map(|i| base + i));

    let entry_struct = StructArray::try_new(
        FieldNames::from(["key", "value"]),
        vec![
            PrimitiveArray::new(keys.freeze(), Validity::NonNullable).into_array(),
            PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
        ],
        LEN,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    let offsets: BufferMut<u64> = (0..LEN as u64).collect();
    let sizes: BufferMut<u64> = std::iter::repeat_n(1u64, LEN).collect();
    let entries = ListViewArray::try_new(
        entry_struct,
        offsets.freeze().into_array(),
        sizes.freeze().into_array(),
        validity(nullable),
        &mut ctx,
    )
    .unwrap();

    let map_dtype = MapDType::try_new(
        DType::Primitive(PType::I64, Nullability::NonNullable),
        DType::Primitive(PType::I64, Nullability::NonNullable),
        false,
    )
    .unwrap();
    MapArray::try_new(map_dtype, entries).unwrap().into_array()
}
