// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures Vortex file scans that produce Arrow offset arrays.
//! The file writer uses the default compressor for strings.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::LazyLock;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BinaryArray;
use arrow_array::StringArray;
use arrow_array::types::BinaryType;
use arrow_array::types::ByteArrayType;
use arrow_array::types::Utf8Type;
use divan::Bencher;
use divan::counter::ItemsCount;
use futures::StreamExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySessionExt;
use vortex_array::stream::ArrayStreamExt;
#[expect(
    deprecated,
    reason = "the benchmark requests an explicit offset layout"
)]
use vortex_arrow::ArrowArrayExecutor;
use vortex_buffer::ByteBufferMut;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&FILE);
    divan::main();
}

const ROWS_PER_CHUNK: usize = 65_536;
const CHUNKS: usize = 16;
const ROWS: usize = ROWS_PER_CHUNK * CHUNKS;
const BENCH_EDITION: EditionId = EditionId::new("bench", 2026, 8, 0);

static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
});

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let _guard = RUNTIME.enter();
    let session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_tokio();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

fn enable_all_registered_array_encodings(session: &VortexSession) {
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: BENCH_EDITION,
            min_vortex_version: None,
        })
        .unwrap();
    let ids = session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>());
    for id in ids {
        editions
            .declare_inclusion(EditionInclusion::new(&id, BENCH_EDITION))
            .unwrap();
    }
    session.enable_edition(BENCH_EDITION).unwrap();
}

#[derive(Clone, Copy)]
enum ByteKind {
    Utf8,
    Binary,
}

impl Display for ByteKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Utf8 => "string_array",
            Self::Binary => "binary_array",
        })
    }
}

const BYTE_KINDS: &[ByteKind] = &[ByteKind::Utf8, ByteKind::Binary];

static FILE: LazyLock<VortexFile> = LazyLock::new(make_file);

fn string_chunk(chunk: usize) -> StructArray {
    let start = chunk * ROWS_PER_CHUNK;
    let values = (start..start + ROWS_PER_CHUNK)
        .map(|index| format!("https://example.com/common/path/{index:08}/shared-suffix"))
        .collect::<Vec<_>>();
    let values = VarBinViewArray::from_iter(
        values.iter().map(|value| Some(value.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    StructArray::from_fields(&[("value", values.into_array())]).unwrap()
}

fn make_file() -> VortexFile {
    let chunks = (0..CHUNKS)
        .map(|chunk| string_chunk(chunk).into_array())
        .collect::<Vec<_>>();
    let array = ChunkedArray::from_iter(chunks).into_array();
    let strategy = WriteStrategyBuilder::default()
        .with_row_block_size(ROWS_PER_CHUNK)
        .with_data_block_target_bytes(None)
        .build();
    let mut bytes = ByteBufferMut::empty();
    RUNTIME
        .block_on(
            SESSION
                .write_options()
                .with_strategy(strategy)
                .write(&mut bytes, array.to_array_stream()),
        )
        .unwrap();
    SESSION.open_options().open_buffer(bytes).unwrap()
}

#[expect(
    deprecated,
    reason = "the benchmark requests an explicit offset layout"
)]
fn to_offset_array(array: ArrayRef, kind: ByteKind) -> ArrowArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let struct_array = array.execute::<StructArray>(&mut ctx).unwrap();
    let values = struct_array
        .unmasked_field_by_name("value")
        .unwrap()
        .clone();
    let arrow = match kind {
        ByteKind::Utf8 => values.execute_arrow(Some(&Utf8Type::DATA_TYPE), &mut ctx),
        ByteKind::Binary => values.execute_arrow(Some(&BinaryType::DATA_TYPE), &mut ctx),
    }
    .unwrap();

    match kind {
        ByteKind::Utf8 => assert!(arrow.as_any().is::<StringArray>()),
        ByteKind::Binary => assert!(arrow.as_any().is::<BinaryArray>()),
    }
    arrow
}

fn read_to_offset_array(file: &VortexFile, kind: ByteKind) -> ArrowArrayRef {
    RUNTIME.block_on(async {
        let array = file
            .scan()
            .unwrap()
            .into_array_stream()
            .unwrap()
            .read_all()
            .await
            .unwrap();
        to_offset_array(array, kind)
    })
}

fn read_to_offset_batches(file: &VortexFile, kind: ByteKind) -> Vec<ArrowArrayRef> {
    RUNTIME.block_on(async {
        let mut stream = file.scan().unwrap().into_array_stream().unwrap();
        let mut arrays = Vec::new();
        let mut rows = 0;
        while let Some(array) = stream.next().await {
            let array = array.unwrap();
            rows += array.len();
            arrays.push(to_offset_array(array, kind));
        }
        assert_eq!(rows, ROWS);
        arrays
    })
}

#[divan::bench(args = BYTE_KINDS)]
fn file_to_offset_array(bencher: Bencher, kind: ByteKind) {
    let file = &*FILE;
    bencher
        .with_inputs(|| file)
        .input_counter(|_| ItemsCount::new(ROWS))
        .bench_values(|file| read_to_offset_array(file, kind));
}

#[divan::bench(args = BYTE_KINDS)]
fn file_to_offset_batches(bencher: Bencher, kind: ByteKind) {
    let file = &*FILE;
    bencher
        .with_inputs(|| file)
        .input_counter(|_| ItemsCount::new(ROWS))
        .bench_values(|file| read_to_offset_batches(file, kind));
}
