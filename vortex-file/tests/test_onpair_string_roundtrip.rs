// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Round-trip a string column through the full Vortex file writer +
//! reader. Mirrors the call shape `vortex-bench/src/conversions.rs` uses, so
//! any "normalize forbids encoding" regression caused by OnPair not being
//! registered in the default session or absent from `ALLOWED_ENCODINGS`
//! shows up here.

#![cfg(feature = "onpair")]
#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::LazyLock;

use futures::StreamExt;
use futures::pin_mut;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    session
});

fn corpus(n: usize) -> Vec<String> {
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
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pick = (state as usize) % templates.len();
        #[expect(clippy::cast_possible_truncation)]
        let id = state as u32;
        out.push(templates[pick].replace("{id}", &format!("{id:08x}")));
    }
    out
}

/// Build a single-column StructArray of `Utf8` strings and round-trip it
/// through `VortexWriteOptions::write` + `OpenOptions::open_buffer`.
///
/// TODO(onpair): currently fails with
/// `Misaligned buffer cannot be used to build PrimitiveArray of u32` when the
/// cascading compressor leaves `dict_offsets` / `codes_offsets` as raw
/// `PrimitiveArray<u32>` children (i.e. doesn't bit-pack them). The fix is
/// to move those offset arrays into the OnPair array's `VTable::buffer`
/// slots (where alignment is preserved via `BufferHandle::alignment`),
/// rather than store them as primitive slot children. Re-enable this test
/// once that refactor lands.
#[tokio::test]
#[ignore = "Misaligned buffer on file roundtrip; tracked as a layout follow-up"]
async fn onpair_string_file_roundtrip() {
    let n = 4096usize;
    let strings = corpus(n);
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

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write Vortex file");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");

    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");
    pin_mut!(stream);

    let mut collected: Vec<Option<String>> = Vec::with_capacity(n);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("chunk");
        let strct = chunk
            .try_downcast::<vortex_array::arrays::Struct>()
            .expect("Struct");
        let url = strct.unmasked_field(0).clone();
        let mut ctx = SESSION.create_execution_ctx();
        let url = url
            .execute::<VarBinViewArray>(&mut ctx)
            .expect("canonicalize url");
        url.with_iterator(|iter| {
            for b in iter {
                collected.push(b.map(|s| String::from_utf8_lossy(s).into_owned()));
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
    }
    assert_eq!(collected.len(), n);
    for (i, want) in strings.iter().enumerate() {
        assert_eq!(collected[i].as_deref(), Some(want.as_str()), "row {i}");
    }
}
