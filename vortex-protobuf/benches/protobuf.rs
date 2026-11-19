// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks of the protobuf extension type on 100,000 clickstream events, whose schema has 30
//! leaf fields in nested messages, repeated messages, repeated strings and many strings.
//!
//! - `decode` parses the wire format into a protobuf array. Its baseline parses each message into
//!   a [`DynamicMessage`], which any conversion through `prost-reflect` messages does first.
//! - `encode` serializes a protobuf array. Its baseline serializes the [`DynamicMessage`]s.
//! - `scan` reads a Vortex file in memory through the file API, and executes the result to
//!   canonical arrays: one leaf field of the protobuf column, the whole column, and the wire bytes
//!   stored as a binary column, which is how a file stores protobuf messages without the type.
//!
//! The decode and encode benchmarks count the wire bytes, so divan reports their throughput. The
//! binary prints what divan cannot measure before the benchmarks run: the size of the events in
//! each form, and the bytes that each scan reads, both the bytes of the segments that it needs and
//! the bytes that a local file reads, which also reads the gaps between nearby segments.
//!
//! All work runs on the benchmark thread, on a current-thread runtime.

mod clickstream;

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use divan::Bencher;
use futures::future::BoxFuture;
use mimalloc::MiMalloc;
use prost::Message;
use prost::encoding::encode_varint;
use prost_reflect::DynamicMessage;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSessionExt;
use vortex_edition::UNSTABLE;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_file::Footer;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_protobuf::Protobuf;
use vortex_protobuf::ProtobufMetadata;
use vortex_protobuf::decode_messages;
use vortex_protobuf::encode_messages;
use vortex_protobuf::protobuf_field;
use vortex_session::VortexSession;

use crate::clickstream::Clickstream;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// The number of events.
const EVENTS: usize = 100_000;

/// The seed of the generated events.
const SEED: u64 = 42;

/// The samples of each decode and encode benchmark. An iteration handles all events and takes
/// over 100 milliseconds, so fewer samples than divan's default of 100 give stable results.
const SAMPLES: u32 = 20;

/// The zstd level that the wire bytes are compressed with, for comparison.
const ZSTD_LEVEL: i32 = 3;

/// The name of the column that holds the events in both files.
const COLUMN: &str = "event";

/// The leaf fields that `scan::field` reads: an integer of the event, and a string of a nested
/// message.
const FIELDS: [&str; 2] = ["timestamp_ms", "geo.city"];

/// The runtime that drives file writes and scans on the calling thread.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

/// A session that writes protobuf columns with the extension layout.
static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| session().vortex_expect("create the benchmark session"));

/// The events that the benchmarks start from.
static DATA: LazyLock<Events> =
    LazyLock::new(|| Events::generate().vortex_expect("generate the events"));

/// The Vortex files of the events.
static FILES: LazyLock<Files> =
    LazyLock::new(|| Files::write(&DATA).vortex_expect("write the files"));

fn main() {
    print_report().vortex_expect("measure the sizes");
    divan::main();
}

/// Creates a session whose writer stores protobuf columns with the extension layout, which the
/// unstable edition includes.
fn session() -> VortexResult<VortexSession> {
    let session = VortexSession::empty().with_handle(RUNTIME.handle());
    for family in EDITION_FAMILIES {
        session.editions().declare_family(family)?;
    }
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    vortex_file::register_default_encodings(&session);
    vortex_protobuf::initialize(&session);
    session.enable_edition(CORE_2026_08_3)?;
    session.enable_edition(UNSTABLE)?;
    Ok(session)
}

/// The events in the forms that the benchmarks start from.
struct Events {
    /// The protobuf dtype of the events.
    dtype: ExtDType<Protobuf>,
    /// The serialized events.
    wire: Vec<Vec<u8>>,
    /// The total size of the serialized events.
    wire_bytes: usize,
    /// The events as dynamic messages.
    messages: Vec<DynamicMessage>,
    /// The events decoded into a protobuf array.
    array: ArrayRef,
}

impl Events {
    /// Generates [`EVENTS`] events.
    fn generate() -> VortexResult<Self> {
        let mut clickstream = Clickstream::new(SEED)?;
        let messages = clickstream.events(EVENTS)?;
        let wire = messages
            .iter()
            .map(Message::encode_to_vec)
            .collect::<Vec<_>>();
        let wire_bytes = wire.iter().map(Vec::len).sum();
        let metadata = ProtobufMetadata::try_new(clickstream.descriptor())?;
        let dtype = ExtDType::try_new(metadata, Nullability::NonNullable)?;
        let array = decode_messages(&dtype, wire.iter().map(Vec::as_slice))?;

        // Both directions count the wire bytes, which is only right if the encoder writes them
        // back unchanged.
        let encoded = encode_messages(&array, &mut SESSION.create_execution_ctx())?;
        vortex_ensure!(
            wire.iter()
                .enumerate()
                .all(|(row, message)| encoded.bytes_at(row).as_slice() == message.as_slice()),
            "The protobuf encoder does not write the events back unchanged"
        );

        Ok(Self {
            dtype,
            wire,
            wire_bytes,
            messages,
            array,
        })
    }

    /// The serialized events, each after its length, as a file or a stream holds them.
    fn length_delimited(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.wire_bytes + 2 * self.wire.len());
        for message in &self.wire {
            encode_varint(message.len() as u64, &mut bytes);
            bytes.extend_from_slice(message);
        }
        bytes
    }
}

/// The Vortex files of the events, each a table with one column.
struct Files {
    /// The events as a protobuf column, which the writer splits into a column per field, down to
    /// the fields of nested messages.
    protobuf: InMemoryFile,
    /// The serialized events as a binary column.
    wire: InMemoryFile,
}

impl Files {
    /// Writes both files.
    fn write(events: &Events) -> VortexResult<Self> {
        let protobuf = StructArray::from_fields(&[(COLUMN, events.array.clone())])?;
        let wire = VarBinViewArray::from_iter_bin(&events.wire).into_array();
        let wire = StructArray::from_fields(&[(COLUMN, wire)])?;
        Ok(Self {
            protobuf: InMemoryFile::write(&protobuf.into_array())?,
            wire: InMemoryFile::write(&wire.into_array())?,
        })
    }
}

/// A Vortex file in memory.
struct InMemoryFile {
    bytes: ByteBuffer,
    footer: Footer,
}

impl InMemoryFile {
    /// Writes `table` with the session's default write strategy.
    fn write(table: &ArrayRef) -> VortexResult<Self> {
        let mut buffer = ByteBufferMut::empty();
        RUNTIME.block_on(
            SESSION
                .write_options()
                .write(&mut buffer, table.to_array_stream()),
        )?;
        let bytes = buffer.freeze();
        let footer = SESSION
            .open_options()
            .open_buffer(bytes.clone())?
            .footer()
            .clone();
        Ok(Self { bytes, footer })
    }

    /// Opens the file through the asynchronous reads that a file on disk uses. The reads are
    /// coalesced as `coalesce` configures, and add the bytes that they read to `bytes_read`.
    ///
    /// The file is opened with its footer, so all reads are scan reads.
    fn open(
        &self,
        coalesce: Option<CoalesceConfig>,
        bytes_read: Arc<AtomicU64>,
    ) -> VortexResult<VortexFile> {
        let source = CountingSource {
            bytes: self.bytes.clone(),
            coalesce,
            bytes_read,
        };
        RUNTIME.block_on(
            SESSION
                .open_options()
                .with_footer(self.footer.clone())
                .open_read(source),
        )
    }

    /// The bytes that a scan of `projection` reads, with reads coalesced as `coalesce` configures.
    fn bytes_read(
        &self,
        projection: &Expression,
        coalesce: Option<CoalesceConfig>,
    ) -> VortexResult<u64> {
        let bytes_read = Arc::new(AtomicU64::new(0));
        let file = self.open(coalesce, Arc::clone(&bytes_read))?;
        scan(&file, projection.bind(file.dtype())?)?;
        Ok(bytes_read.load(Ordering::Relaxed))
    }
}

/// The bytes of a file in memory, which counts the bytes read from it.
#[derive(Clone)]
struct CountingSource {
    bytes: ByteBuffer,
    /// How to coalesce nearby reads, as a local file or an object store does.
    coalesce: Option<CoalesceConfig>,
    bytes_read: Arc<AtomicU64>,
}

impl VortexReadAt for CountingSource {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce
    }

    fn concurrency(&self) -> usize {
        self.bytes.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.bytes.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.bytes_read.fetch_add(length as u64, Ordering::Relaxed);
        self.bytes.read_at(offset, length, alignment)
    }
}

/// The column of the events.
fn column() -> Expression {
    get_item(COLUMN, root())
}

/// The leaf field at the dotted `path` of the protobuf column.
fn field(path: &str) -> Expression {
    protobuf_field(path, column())
}

/// Scans `projection` from `file` and executes each chunk to canonical arrays, returning the
/// number of rows.
fn scan(file: &VortexFile, projection: BoundExpression) -> VortexResult<usize> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rows = 0;
    for chunk in file
        .scan()?
        .with_projection(projection)
        .into_array_iter(&*RUNTIME)?
    {
        let chunk = chunk?;
        rows += chunk.len();
        chunk.execute::<RecursiveCanonical>(&mut ctx)?;
    }
    Ok(rows)
}

/// Benchmarks scans of `projection` from `file`.
fn bench_scan(bencher: Bencher, file: &InMemoryFile, projection: &Expression) {
    let file = file
        .open(None, Arc::default())
        .vortex_expect("open the file");
    let projection = projection
        .bind(file.dtype())
        .vortex_expect("bind the projection");
    bencher.bench(|| scan(&file, projection.clone()).vortex_expect("scan the file"));
}

/// Prints the sizes of the events in each form and the bytes that each scan reads.
fn print_report() -> VortexResult<()> {
    let events = &*DATA;
    let files = &*FILES;
    let delimited = events.length_delimited();
    let compressed = zstd::bulk::compress(&delimited, ZSTD_LEVEL)?;

    println!(
        "{EVENTS} events of {} bytes on average",
        events.wire_bytes / EVENTS
    );
    println!();
    println!("{:<48}{:>10}{:>10}", "Size", "MB", "vs wire");
    let sizes = [
        ("Wire bytes, length-delimited".to_owned(), delimited.len()),
        (
            format!("Wire bytes, length-delimited, zstd level {ZSTD_LEVEL}"),
            compressed.len(),
        ),
        (
            "Vortex file, binary column of wire bytes".to_owned(),
            files.wire.bytes.len(),
        ),
        (
            "Vortex file, protobuf column".to_owned(),
            files.protobuf.bytes.len(),
        ),
    ];
    for (form, bytes) in sizes {
        println!(
            "{form:<48}{:>10.2}{:>9.2}x",
            megabytes(bytes as u64),
            bytes as f64 / delimited.len() as f64
        );
    }

    println!();
    println!(
        "{:<48}{:>14}{:>16}",
        "Bytes read by a scan", "segments kB", "local file kB"
    );
    let scans = FIELDS
        .iter()
        .map(|path| (format!("Field {path}"), &files.protobuf, field(path)))
        .chain([
            ("Protobuf column".to_owned(), &files.protobuf, column()),
            (
                "Binary column of wire bytes".to_owned(),
                &files.wire,
                column(),
            ),
        ]);
    for (name, file, projection) in scans {
        let segments = file.bytes_read(&projection, None)?;
        let local = file.bytes_read(&projection, Some(CoalesceConfig::file()))?;
        println!(
            "{name:<48}{:>14.1}{:>16.1}",
            kilobytes(segments),
            kilobytes(local)
        );
    }
    println!();
    Ok(())
}

/// Converts bytes to decimal megabytes, the unit of divan's throughput.
fn megabytes(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}

/// Converts bytes to decimal kilobytes.
fn kilobytes(bytes: u64) -> f64 {
    bytes as f64 / 1e3
}

/// Decoding the wire format.
#[divan::bench_group(sample_count = SAMPLES)]
mod decode {
    use divan::Bencher;
    use divan::counter::BytesCount;
    use prost_reflect::DynamicMessage;
    use vortex_error::VortexExpect;
    use vortex_protobuf::decode_messages;

    use super::DATA;

    /// Decodes the events into a protobuf array.
    #[divan::bench]
    fn vortex(bencher: Bencher) {
        let events = &*DATA;
        bencher
            .counter(BytesCount::new(events.wire_bytes))
            .bench(|| {
                decode_messages(&events.dtype, events.wire.iter().map(Vec::as_slice))
                    .vortex_expect("decode the events")
            });
    }

    /// Parses each event into a dynamic message.
    #[divan::bench]
    fn dynamic_message(bencher: Bencher) {
        let events = &*DATA;
        let descriptor = events.dtype.metadata().message_descriptor();
        bencher
            .counter(BytesCount::new(events.wire_bytes))
            .bench(|| {
                events
                    .wire
                    .iter()
                    .map(|message| DynamicMessage::decode(descriptor.clone(), message.as_slice()))
                    .collect::<Result<Vec<_>, _>>()
                    .vortex_expect("parse the events")
            });
    }
}

/// Encoding to the wire format.
#[divan::bench_group(sample_count = SAMPLES)]
mod encode {
    use divan::Bencher;
    use divan::counter::BytesCount;
    use prost::Message;
    use vortex_array::VortexSessionExecute;
    use vortex_error::VortexExpect;
    use vortex_protobuf::encode_messages;

    use super::DATA;
    use super::SESSION;

    /// Encodes the protobuf array into a binary array.
    #[divan::bench]
    fn vortex(bencher: Bencher) {
        let events = &*DATA;
        bencher
            .counter(BytesCount::new(events.wire_bytes))
            .bench(|| {
                encode_messages(&events.array, &mut SESSION.create_execution_ctx())
                    .vortex_expect("encode the events")
            });
    }

    /// Serializes the dynamic messages into one buffer, with the offsets that a binary array
    /// needs.
    #[divan::bench]
    fn dynamic_message(bencher: Bencher) {
        let events = &*DATA;
        bencher
            .counter(BytesCount::new(events.wire_bytes))
            .bench(|| {
                let mut bytes = Vec::with_capacity(events.wire_bytes);
                let mut offsets = Vec::with_capacity(events.messages.len() + 1);
                offsets.push(0);
                for message in &events.messages {
                    message
                        .encode(&mut bytes)
                        .vortex_expect("serialize an event");
                    offsets.push(bytes.len());
                }
                (bytes, offsets)
            });
    }
}

/// Scanning a Vortex file.
mod scan {
    use divan::Bencher;

    use super::FIELDS;
    use super::FILES;
    use super::bench_scan;
    use super::column;

    /// Reads one leaf field of the protobuf column.
    #[divan::bench(args = FIELDS)]
    fn field(bencher: Bencher, path: &str) {
        bench_scan(bencher, &FILES.protobuf, &super::field(path));
    }

    /// Reads the whole protobuf column.
    #[divan::bench]
    fn protobuf_column(bencher: Bencher) {
        bench_scan(bencher, &FILES.protobuf, &column());
    }

    /// Reads the wire bytes from a binary column, which a reader still has to parse.
    #[divan::bench]
    fn wire_column(bencher: Bencher) {
        bench_scan(bencher, &FILES.wire, &column());
    }
}
