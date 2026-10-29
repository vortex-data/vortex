// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks writing a file with and without encryption.
//!
//! Random `u64` columns barely compress, so nearly every input byte is encrypted, which makes
//! the cost of encryption on the write path visible. The runtime has several worker threads, so
//! that column writers can encrypt in parallel.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::encryption::EncryptionKey;
use vortex_file::encryption::FileEncryption;
use vortex_file::encryption::KeyConfig;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const COLUMNS: usize = 8;
const ROWS: usize = 1 << 21;

static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
});

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let _guard = RUNTIME.enter();
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_tokio();
    vortex_file::register_default_encodings(&session);
    session
});

static ARRAY: LazyLock<ArrayRef> = LazyLock::new(|| {
    let mut rng = StdRng::seed_from_u64(0);
    let fields = (0..COLUMNS)
        .map(|column| {
            let values: Buffer<u64> = (0..ROWS).map(|_| rng.random()).collect();
            (format!("c{column}"), values.into_array())
        })
        .collect::<Vec<_>>();
    StructArray::from_fields(&fields).unwrap().into_array()
});

fn write(encryption: Option<FileEncryption>) -> usize {
    RUNTIME.block_on(async {
        let mut options = SESSION.write_options().disable_editions();
        if let Some(encryption) = encryption {
            options = options.with_encryption(encryption);
        }
        let mut buf = ByteBufferMut::empty();
        options
            .write(&mut buf, ARRAY.clone().to_array_stream())
            .await
            .unwrap();
        buf.len()
    })
}

#[divan::bench(sample_count = 30)]
fn plaintext(bencher: Bencher) {
    LazyLock::force(&ARRAY);
    bencher.bench(|| write(None));
}

#[divan::bench(sample_count = 30)]
fn encrypted(bencher: Bencher) {
    LazyLock::force(&ARRAY);
    let encryption = FileEncryption::new(KeyConfig::new(
        EncryptionKey::from([7; 32]),
        b"key".as_slice(),
    ));
    bencher.bench(|| write(Some(encryption.clone())));
}
