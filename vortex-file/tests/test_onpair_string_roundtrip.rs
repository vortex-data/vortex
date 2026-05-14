// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Round-trip stress tests for OnPair through the full Vortex file writer +
//! reader. Mirrors the call shape `vortex-bench/src/conversions.rs` uses and
//! the multi-column, many-chunk pattern of TPC-H tables (`supplier_0.vortex`
//! is the file from which CI surfaced
//! `Misaligned buffer cannot be used to build PrimitiveArray of u32`).

#![cfg(feature = "onpair")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::tests_outside_test_module,
    clippy::redundant_clone
)]

use std::sync::LazyLock;

use futures::StreamExt;
use futures::pin_mut;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::session::DTypeSession;
use vortex_array::optimizer::kernels::ArrayKernels;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

/// Full default Vortex session — the same set of sub-sessions
/// `vortex::VortexSession::default()` would install, plus
/// `register_default_encodings`. Built inline here because `vortex-file`
/// can't depend on the umbrella `vortex` crate (it's the other way round).
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<DTypeSession>()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<ArrayKernels>()
        .with::<AggregateFnSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    session
});

fn corpus(n: usize, offset: u64) -> Vec<String> {
    let templates: &[&str] = &[
        "https://www.example.com/products/{id}",
        "https://cdn.example.com/img/{id}.webp",
        "https://api.example.com/v2/orders/{id}",
        "https://www.example.com/users/{id}/profile",
        "INFO  request_id={id} status=200 method=GET",
        "WARN  request_id={id} status=429 method=POST",
        "ERROR request_id={id} status=500 method=PUT",
    ];
    let mut out = Vec::with_capacity(n);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64.wrapping_add(offset);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pick = (state as usize) % templates.len();
        let id = state as u32;
        out.push(templates[pick].replace("{id}", &format!("{id:08x}")));
    }
    out
}

/// Write `data` to an in-memory `Vec<u8>` using the **full default Vortex
/// compressor** (`WriteStrategyBuilder::default()` =
/// `BtrBlocksCompressor::default()` cascading through every registered
/// scheme, including OnPair), then open the resulting bytes via
/// `OpenOptions::open_buffer` and stream every chunk back.
async fn write_and_read_back(data: vortex_array::ArrayRef) -> Vec<vortex_array::ArrayRef> {
    // `write_options()` builds a `VortexWriteOptions` whose `strategy` is
    // `WriteStrategyBuilder::default().build()` — the same path `vortex-bench`
    // uses for Parquet → Vortex conversion. No custom strategy injected.
    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write Vortex file");

    // Read back from the in-memory byte buffer; no disk, no FS.
    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");

    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");
    pin_mut!(stream);

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.expect("chunk"));
    }
    chunks
}

/// Single string column, single chunk. The simplest case.
#[tokio::test]
async fn single_column_single_chunk() {
    let n = 4096usize;
    let strings = corpus(n, 0);
    let str_array = VarBinViewArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let data = StructArray::new(
        FieldNames::from(["url"]),
        vec![str_array],
        n,
        Validity::NonNullable,
    )
    .into_array();

    let chunks = write_and_read_back(data).await;
    let mut row = 0;
    for chunk in chunks {
        let strct = chunk
            .try_downcast::<vortex_array::arrays::Struct>()
            .expect("Struct");
        let url = strct.unmasked_field(0).clone();
        let mut ctx = SESSION.create_execution_ctx();
        let url = url.execute::<VarBinViewArray>(&mut ctx).expect("canon");
        url.with_iterator(|iter| {
            for b in iter {
                assert_eq!(b, Some(strings[row].as_bytes()), "row {row}");
                row += 1;
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
    }
    assert_eq!(row, n);
}

/// Many rows → many chunks via the writer's default row_block_size.
#[tokio::test]
async fn single_column_many_chunks() {
    let n = 50_000usize;
    let strings = corpus(n, 0);
    let str_array = VarBinViewArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let data = StructArray::new(
        FieldNames::from(["url"]),
        vec![str_array],
        n,
        Validity::NonNullable,
    )
    .into_array();

    let chunks = write_and_read_back(data).await;
    let mut row = 0;
    for chunk in chunks {
        let strct = chunk
            .try_downcast::<vortex_array::arrays::Struct>()
            .expect("Struct");
        let url = strct.unmasked_field(0).clone();
        let mut ctx = SESSION.create_execution_ctx();
        let url = url.execute::<VarBinViewArray>(&mut ctx).expect("canon");
        url.with_iterator(|iter| {
            for b in iter {
                assert_eq!(b, Some(strings[row].as_bytes()), "row {row}");
                row += 1;
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
    }
    assert_eq!(row, n);
}

/// TPC-H supplier-shaped table: 5 string columns + a primary key + a
/// foreign key + a decimal/integer, with the row count large enough to
/// exercise multiple chunks. This is the configuration that surfaced the
/// `Misaligned buffer` error in CI.
#[tokio::test]
async fn tpch_supplier_shape() {
    let n = 32_000usize;
    let names = corpus(n, 1);
    let addresses = corpus(n, 2);
    let phones = corpus(n, 3);
    let comments = corpus(n, 4);
    let cities = corpus(n, 5);

    let suppkey: Vec<i64> = (0..n as i64).collect();
    let nationkey: Vec<i32> = (0..n as i32).map(|i| i % 25).collect();
    let acctbal: Vec<i64> = (0..n as i64).map(|i| (i * 13) % 1_000_000).collect();

    let mk_str = |v: &[String]| -> vortex_array::ArrayRef {
        VarBinViewArray::from_iter(
            v.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array()
    };

    let data = StructArray::new(
        FieldNames::from([
            "s_suppkey",
            "s_name",
            "s_address",
            "s_nationkey",
            "s_phone",
            "s_acctbal",
            "s_comment",
            "s_city",
        ]),
        vec![
            PrimitiveArray::from_iter(suppkey.iter().copied()).into_array(),
            mk_str(&names),
            mk_str(&addresses),
            PrimitiveArray::from_iter(nationkey.iter().copied()).into_array(),
            mk_str(&phones),
            PrimitiveArray::from_iter(acctbal.iter().copied()).into_array(),
            mk_str(&comments),
            mk_str(&cities),
        ],
        n,
        Validity::NonNullable,
    )
    .into_array();

    let chunks = write_and_read_back(data).await;

    let mut row = 0;
    for chunk in chunks {
        let strct = chunk
            .try_downcast::<vortex_array::arrays::Struct>()
            .expect("Struct");
        let chunk_len = strct.as_ref().len();
        let mut ctx = SESSION.create_execution_ctx();

        let name = strct
            .unmasked_field(1)
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();
        let address = strct
            .unmasked_field(2)
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();
        let phone = strct
            .unmasked_field(4)
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();
        let comment = strct
            .unmasked_field(6)
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();
        let city = strct
            .unmasked_field(7)
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();

        for (s, want) in [
            (&name, &names),
            (&address, &addresses),
            (&phone, &phones),
            (&comment, &comments),
            (&city, &cities),
        ] {
            let base = row;
            s.with_iterator(|iter| {
                for (i, b) in iter.enumerate() {
                    assert_eq!(b, Some(want[base + i].as_bytes()), "row {}", base + i);
                }
                Ok::<_, vortex_error::VortexError>(())
            })
            .unwrap();
        }
        row += chunk_len;
    }
    assert_eq!(row, n);
}

/// Mixed-shape strings: empty, short, very long, with a fair chunk of nulls
/// — exercising the validity child + edge offsets.
#[tokio::test]
async fn nullable_and_extreme_shapes() {
    let n = 16_000usize;
    let mut strings: Vec<Option<String>> = Vec::with_capacity(n);
    for i in 0..n {
        match i % 11 {
            0 => strings.push(None),
            1 => strings.push(Some(String::new())),
            2 => strings.push(Some("a".repeat(1024))),
            3 => strings.push(Some(format!("row-{i}"))),
            _ => strings.push(Some(corpus(1, i as u64).pop().unwrap())),
        }
    }
    let str_array = VarBinViewArray::from_iter(
        strings.iter().map(|s| s.as_deref()),
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let data = StructArray::new(
        FieldNames::from(["s"]),
        vec![str_array],
        n,
        Validity::NonNullable,
    )
    .into_array();

    let chunks = write_and_read_back(data).await;
    let mut row = 0;
    for chunk in chunks {
        let strct = chunk
            .try_downcast::<vortex_array::arrays::Struct>()
            .expect("Struct");
        let mut ctx = SESSION.create_execution_ctx();
        let s = strct
            .unmasked_field(0)
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();
        s.with_iterator(|iter| {
            for b in iter {
                let want = strings[row].as_deref().map(str::as_bytes);
                assert_eq!(b, want, "row {row}");
                row += 1;
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
    }
    assert_eq!(row, n);
}
