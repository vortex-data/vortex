// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic `turmoil` network simulation tests for HTTP and S3 object-store access.
//!
//! All TCP traffic and retry backoff timers run inside `turmoil`'s virtual clock and simulated
//! network — zero real sockets and zero wall-clock sleeps.

use std::collections::VecDeque;
use std::error::Error as StdError;
use std::io;
use std::io::ErrorKind;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use futures::future;
use http::HeaderValue;
use http::Method;
use http::Response;
use http::Uri;
use http::header::HOST;
use http::response::Parts;
use http_body_util::BodyExt;
use hyper::body::Body;
use hyper::body::Frame;
use hyper::body::Incoming;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use object_store::BackoffConfig;
use object_store::ClientOptions;
use object_store::ObjectStore;
use object_store::RetryConfig;
use object_store::aws::AmazonS3Builder;
use object_store::client::HttpClient;
use object_store::client::HttpConnector;
use object_store::client::HttpError;
use object_store::client::HttpErrorKind;
use object_store::client::HttpRequest;
use object_store::client::HttpResponse;
use object_store::client::HttpResponseBody;
use object_store::client::HttpService;
use object_store::http::HttpBuilder;
use object_store::path::Path as ObjectPath;
use parking_lot::Mutex;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::runtime::Builder as TokioBuilder;
use tokio::runtime::Handle as TokioHandle;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time;
use turmoil::Builder as SimBuilder;
use turmoil::Sim;
use turmoil::net::TcpListener;
use turmoil::net::TcpStream;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::VortexReadAt;
use vortex_io::VortexWrite;
use vortex_io::object_store::ObjectStoreReadAt;
use vortex_io::object_store::ObjectStoreWrite;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const OBJECT_HEADERS: &str =
    "Last-Modified: Wed, 21 Oct 2015 07:28:00 GMT\r\nETag: \"sim-etag-v1\"";
/// Matches `ObjectStoreWrite`'s internal 16 MiB part threshold in `vortex-io`.
const MULTIPART_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// `object_store::client::HttpConnector` backed by `turmoil::net::TcpStream` and `hyper` HTTP/1.1.
///
/// Captures the turmoil client host's Tokio handle so requests initiated off-host (such as
/// `ObjectStoreWrite`'s detached background abort task) still execute inside the turmoil host.
#[derive(Debug, Clone)]
struct TurmoilHttpConnector {
    runtime: TokioHandle,
    connect_timeout: Duration,
}

impl TurmoilHttpConnector {
    fn current(connect_timeout: Duration) -> Self {
        Self {
            runtime: TokioHandle::current(),
            connect_timeout,
        }
    }
}

impl HttpConnector for TurmoilHttpConnector {
    fn connect(&self, _options: &ClientOptions) -> object_store::Result<HttpClient> {
        Ok(HttpClient::new(self.clone()))
    }
}

struct ChannelBody {
    rx: mpsc::UnboundedReceiver<Result<Frame<Bytes>, HttpError>>,
    worker: JoinHandle<()>,
}

impl Drop for ChannelBody {
    fn drop(&mut self) {
        self.worker.abort();
    }
}

impl Body for ChannelBody {
    type Data = Bytes;
    type Error = HttpError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        self.rx.poll_recv(cx)
    }
}

fn map_hyper_err(err: hyper::Error) -> HttpError {
    let mut kind = if err.is_timeout() {
        HttpErrorKind::Timeout
    } else if err.is_incomplete_message() || err.is_closed() || err.is_body_write_aborted() {
        HttpErrorKind::Interrupted
    } else {
        HttpErrorKind::Request
    };

    let mut source = err.source();
    while let Some(inner) = source {
        if let Some(io_err) = inner.downcast_ref::<io::Error>() {
            kind = match io_err.kind() {
                ErrorKind::TimedOut => HttpErrorKind::Timeout,
                ErrorKind::ConnectionAborted
                | ErrorKind::ConnectionReset
                | ErrorKind::BrokenPipe
                | ErrorKind::UnexpectedEof => HttpErrorKind::Interrupted,
                _ => kind,
            };
            break;
        }
        source = inner.source();
    }
    HttpError::new(kind, err)
}

async fn send_over_turmoil(
    host: &str,
    port: u16,
    connect_timeout: Duration,
    req: HttpRequest,
) -> Result<(Parts, Incoming, JoinHandle<()>), HttpError> {
    let stream = match time::timeout(connect_timeout, TcpStream::connect((host, port))).await {
        Ok(Ok(stream)) => stream,
        Ok(Err(err)) => {
            let kind = err.kind();
            let detail = io::Error::new(kind, format!("connect {host}:{port} ({kind}): {err}"));
            return Err(HttpError::new(HttpErrorKind::Connect, detail));
        }
        Err(_) => {
            let detail = io::Error::new(
                ErrorKind::TimedOut,
                format!("connect timeout to {host}:{port}"),
            );
            return Err(HttpError::new(HttpErrorKind::Connect, detail));
        }
    };

    let (mut sender, conn) = http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|err| HttpError::new(HttpErrorKind::Connect, err))?;
    let conn_task = tokio::spawn(async move {
        drop(conn.await);
    });

    match sender.send_request(req).await {
        Ok(resp) => {
            let (parts, body) = resp.into_parts();
            Ok((parts, body, conn_task))
        }
        Err(err) => {
            conn_task.abort();
            Err(map_hyper_err(err))
        }
    }
}

#[async_trait]
impl HttpService for TurmoilHttpConnector {
    async fn call(&self, mut req: HttpRequest) -> Result<HttpResponse, HttpError> {
        let host = req
            .uri()
            .host()
            .ok_or_else(|| {
                HttpError::new(
                    HttpErrorKind::Request,
                    io::Error::new(ErrorKind::InvalidInput, "request URI missing host"),
                )
            })?
            .to_string();
        let port = req.uri().port_u16().unwrap_or(80);

        if !req.headers().contains_key(HOST) {
            let value = if port == 80 {
                host.clone()
            } else {
                format!("{host}:{port}")
            };
            req.headers_mut().insert(
                HOST,
                HeaderValue::from_str(&value)
                    .map_err(|err| HttpError::new(HttpErrorKind::Request, err))?,
            );
        }

        let target = req
            .uri()
            .path_and_query()
            .map_or("/", |pq| pq.as_str())
            .to_string();
        *req.uri_mut() = target
            .parse::<Uri>()
            .map_err(|err| HttpError::new(HttpErrorKind::Request, err))?;

        let connect_timeout = self.connect_timeout;
        let (parts_tx, parts_rx) = oneshot::channel();
        let (frame_tx, frame_rx) = mpsc::unbounded_channel();

        let worker = self.runtime.spawn(async move {
            let (parts, mut body, conn_task) =
                match send_over_turmoil(&host, port, connect_timeout, req).await {
                    Ok(res) => res,
                    Err(err) => {
                        drop(parts_tx.send(Err(err)));
                        return;
                    }
                };
            if parts_tx.send(Ok(parts)).is_err() {
                conn_task.abort();
                return;
            }
            while let Some(frame) = body.frame().await {
                let mapped = frame.map_err(map_hyper_err);
                let is_err = mapped.is_err();
                if frame_tx.send(mapped).is_err() || is_err {
                    break;
                }
            }
            drop(conn_task.await);
        });

        let parts = parts_rx.await.map_err(|_| {
            HttpError::new(
                HttpErrorKind::Interrupted,
                io::Error::new(ErrorKind::BrokenPipe, "turmoil HTTP worker terminated"),
            )
        })??;

        Ok(Response::from_parts(
            parts,
            HttpResponseBody::new(ChannelBody {
                rx: frame_rx,
                worker,
            }),
        ))
    }
}

/// Fault injected by the simulated HTTP/S3 server for a single request.
#[derive(Clone, Copy)]
enum Fault {
    Status(u16, &'static str),
    TruncateTcp { prefix_bytes: usize },
    BodyLengthMismatch { actual_len: usize },
}

#[derive(Clone)]
struct HttpReq {
    method: String,
    target: String,
    range: Option<String>,
    body_len: usize,
}

#[derive(Default)]
struct ServerState {
    object: Bytes,
    scripted_faults: Mutex<VecDeque<Fault>>,
    default_fault: Option<Fault>,
    fail_part_number: Option<usize>,
    fail_complete: bool,
    requests: Mutex<Vec<HttpReq>>,
    aborted_uploads: Mutex<Vec<String>>,
}

impl ServerState {
    fn next_fault(&self) -> Option<Fault> {
        self.scripted_faults
            .lock()
            .pop_front()
            .or(self.default_fault)
    }
}

fn attach_server(sim: &mut Sim<'_>, host: &str, state: &Arc<ServerState>) {
    let state = Arc::clone(state);
    sim.host(host, move || {
        let state = Arc::clone(&state);
        async move {
            let listener = TcpListener::bind(("0.0.0.0", 80)).await?;
            loop {
                let (stream, _) = listener.accept().await?;
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    drop(handle_connection(stream, state).await);
                });
            }
        }
    });
}

async fn read_http_request(stream: &mut TcpStream) -> io::Result<HttpReq> {
    let mut header_buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(io::Error::new(
                ErrorKind::UnexpectedEof,
                "EOF before headers",
            ));
        }
        header_buf.extend_from_slice(&chunk[..read]);
        if let Some(pos) = header_buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };

    let header_str = String::from_utf8_lossy(&header_buf[..header_end]);
    let mut lines = header_str.split("\r\n");
    let mut req_line = lines.next().unwrap_or_default().split_whitespace();
    let method = req_line
        .next()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "missing HTTP method"))?
        .to_string();
    let target = req_line
        .next()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "missing request target"))?
        .to_string();

    let (mut content_length, mut range) = (0usize, None);
    for (name, value) in lines.filter_map(|line| line.split_once(':')) {
        let trimmed = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            content_length = trimmed.parse::<usize>().unwrap_or(0);
        } else if name.eq_ignore_ascii_case("range") {
            range = Some(trimmed.to_string());
        }
    }

    let mut remaining = content_length.saturating_sub(header_buf.len() - header_end);
    let mut drain = vec![0u8; 64 * 1024];
    while remaining > 0 {
        let to_read = remaining.min(drain.len());
        let read = stream.read(&mut drain[..to_read]).await?;
        if read == 0 {
            return Err(io::Error::new(
                ErrorKind::UnexpectedEof,
                "EOF in request body",
            ));
        }
        remaining -= read;
    }

    Ok(HttpReq {
        method,
        target,
        range,
        body_len: content_length,
    })
}

fn parse_range_header(header: &str, total_len: usize) -> Option<(usize, usize)> {
    let (start_s, end_s) = header.strip_prefix("bytes=")?.split_once('-')?;
    let start = start_s.parse::<usize>().ok()?;
    let end = if end_s.is_empty() {
        total_len.checked_sub(1)?
    } else {
        end_s
            .parse::<usize>()
            .ok()?
            .min(total_len.saturating_sub(1))
    };
    (start <= end && end < total_len).then_some((start, end))
}

fn query_param<'a>(target: &'a str, key: &str) -> Option<&'a str> {
    let (_, query) = target.split_once('?')?;
    query
        .split('&')
        .find_map(|pair| match pair.split_once('=') {
            Some((k, v)) if k == key => Some(v),
            None if pair == key => Some(""),
            _ => None,
        })
}

async fn write_response(
    stream: &mut TcpStream,
    status: &str,
    extra_headers: &str,
    content_length: usize,
    body: &[u8],
) -> io::Result<()> {
    let sep = if extra_headers.is_empty() { "" } else { "\r\n" };
    let head = format!(
        "HTTP/1.1 {status}\r\n{extra_headers}{sep}Content-Length: {content_length}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes()).await?;
    if !body.is_empty() {
        stream.write_all(body).await?;
    }
    stream.flush().await
}

async fn handle_multipart(
    stream: &mut TcpStream,
    state: &ServerState,
    req: &HttpReq,
) -> io::Result<bool> {
    if req.method == Method::POST.as_str() && query_param(&req.target, "uploads").is_some() {
        let xml = b"<InitiateMultipartUploadResult><UploadId>sim-upload-1</UploadId></InitiateMultipartUploadResult>";
        write_response(
            stream,
            "200 OK",
            "Content-Type: application/xml",
            xml.len(),
            xml,
        )
        .await?;
        return Ok(true);
    }

    if req.method == Method::PUT.as_str()
        && let Some(part_str) = query_param(&req.target, "partNumber")
        && query_param(&req.target, "uploadId").is_some()
    {
        let part_num = part_str.parse::<usize>().unwrap_or(0);
        if state.fail_part_number == Some(part_num) {
            let msg = b"injected put_part failure";
            write_response(stream, "500 Internal Server Error", "", msg.len(), msg).await?;
        } else {
            let etag = format!("ETag: \"part-{part_num}\"");
            write_response(stream, "200 OK", &etag, 0, b"").await?;
        }
        return Ok(true);
    }

    if req.method == Method::POST.as_str() && query_param(&req.target, "uploadId").is_some() {
        if state.fail_complete {
            let msg = b"injected complete_multipart failure";
            write_response(stream, "500 Internal Server Error", "", msg.len(), msg).await?;
        } else {
            let xml = b"<CompleteMultipartUploadResult><ETag>\"final-etag\"</ETag></CompleteMultipartUploadResult>";
            write_response(
                stream,
                "200 OK",
                "Content-Type: application/xml",
                xml.len(),
                xml,
            )
            .await?;
        }
        return Ok(true);
    }

    if req.method == Method::DELETE.as_str()
        && let Some(upload_id) = query_param(&req.target, "uploadId")
    {
        state.aborted_uploads.lock().push(upload_id.to_string());
        write_response(stream, "204 No Content", "", 0, b"").await?;
        return Ok(true);
    }
    Ok(false)
}

async fn handle_connection(mut stream: TcpStream, state: Arc<ServerState>) -> io::Result<()> {
    let req = read_http_request(&mut stream).await?;
    state.requests.lock().push(req.clone());

    if handle_multipart(&mut stream, &state, &req).await? {
        return Ok(());
    }

    let fault = state.next_fault();
    if let Some(Fault::Status(code, reason)) = fault {
        let body = if req.method == Method::HEAD.as_str() {
            b"".as_slice()
        } else {
            reason.as_bytes()
        };
        return write_response(
            &mut stream,
            &format!("{code} {reason}"),
            "",
            body.len(),
            body,
        )
        .await;
    }

    let total_len = state.object.len();
    if req.method == Method::HEAD.as_str() {
        return write_response(&mut stream, "200 OK", OBJECT_HEADERS, total_len, b"").await;
    }

    if let Some(range_hdr) = &req.range
        && let Some((start, end)) = parse_range_header(range_hdr, total_len)
    {
        let slice = &state.object[start..=end];
        let headers = format!("Content-Range: bytes {start}-{end}/{total_len}\r\n{OBJECT_HEADERS}");
        return match fault {
            Some(Fault::TruncateTcp { prefix_bytes }) => {
                let send_len = prefix_bytes.min(slice.len());
                write_response(
                    &mut stream,
                    "206 Partial Content",
                    &headers,
                    slice.len(),
                    &slice[..send_len],
                )
                .await
            }
            Some(Fault::BodyLengthMismatch { actual_len }) => {
                let mut body = Vec::with_capacity(actual_len);
                body.extend_from_slice(&slice[..actual_len.min(slice.len())]);
                body.resize(actual_len, 0xAB);
                write_response(
                    &mut stream,
                    "206 Partial Content",
                    &headers,
                    actual_len,
                    &body,
                )
                .await
            }
            Some(Fault::Status(..)) | None => {
                write_response(
                    &mut stream,
                    "206 Partial Content",
                    &headers,
                    slice.len(),
                    slice,
                )
                .await
            }
        };
    }

    write_response(
        &mut stream,
        "200 OK",
        OBJECT_HEADERS,
        total_len,
        &state.object,
    )
    .await
}

fn retry_config(max_retries: usize) -> RetryConfig {
    // Setting `init_backoff == max_backoff` clamps `object_store`'s thread-local jitter to a fixed
    // 25 ms virtual-clock sleep per retry; `retry_timeout` is a wall-clock safety ceiling.
    RetryConfig {
        backoff: BackoffConfig {
            init_backoff: Duration::from_millis(25),
            max_backoff: Duration::from_millis(25),
            base: 2.0,
        },
        max_retries,
        retry_timeout: Duration::from_secs(30),
    }
}

fn http_store(
    url: &str,
    max_retries: usize,
    connect_timeout: Duration,
) -> VortexResult<Arc<dyn ObjectStore>> {
    Ok(Arc::new(
        HttpBuilder::new()
            .with_url(url)
            .with_retry(retry_config(max_retries))
            .with_client_options(ClientOptions::new().with_allow_http(true))
            .with_http_connector(TurmoilHttpConnector::current(connect_timeout))
            .build()?,
    ))
}

fn s3_store(endpoint: &str) -> VortexResult<Arc<dyn ObjectStore>> {
    // `max_retries = 0` keeps `ObjectStoreWrite`'s detached `smol` abort task from sleeping on
    // `Compat`'s wall-clock Tokio runtime instead of `turmoil`'s virtual clock.
    Ok(Arc::new(
        AmazonS3Builder::new()
            .with_endpoint(endpoint)
            .with_bucket_name("test-bucket")
            .with_region("us-east-1")
            .with_allow_http(true)
            .with_skip_signature(true)
            .with_retry(retry_config(0))
            .with_http_connector(TurmoilHttpConnector::current(DEFAULT_CONNECT_TIMEOUT))
            .build()?,
    ))
}

async fn read_range_err(
    url: &str,
    max_retries: usize,
    connect_timeout: Duration,
    offset: u64,
    length: usize,
) -> VortexResult<VortexError> {
    let store = http_store(url, max_retries, connect_timeout)?;
    let reader =
        ObjectStoreReadAt::new(store, ObjectPath::from("obj.bin"), TokioRuntime::current());
    match reader.read_at(offset, length, Alignment::new(1)).await {
        Err(err) => Ok(err),
        Ok(_) => Err(vortex_err!("expected ObjectStoreReadAt::read_at to fail")),
    }
}

fn find_source<'a, E: StdError + 'static>(err: &'a (dyn StdError + 'static)) -> Option<&'a E> {
    let mut current = Some(err);
    while let Some(err) = current {
        if let Some(found) = err.downcast_ref::<E>() {
            return Some(found);
        }
        current = err.source();
    }
    None
}

fn make_sim() -> Sim<'static> {
    make_sim_with_duration(Duration::from_secs(60))
}

fn make_sim_with_duration(duration: Duration) -> Sim<'static> {
    SimBuilder::new()
        .rng_seed(0x5EED_4B01)
        .simulation_duration(duration)
        .build()
}

/// Steps a simulation paired with `ObjectStoreWrite`'s detached `smol` cleanup task.
///
/// Both `ObjectStoreWrite::shutdown()` (via `cleanup_error`) and `ObjectStoreWrite::abort()` await
/// an internal `oneshot::Receiver` until the off-host `smol` task completes `upload.abort().await`,
/// so `sim.step()` cannot return `Ok(true)` before `DELETE ?uploadId=...` finishes on the server.
/// Callers should build the sim with a generous virtual duration (e.g. 24h) so idle ticks while
/// waiting on OS scheduling of `smol` cannot exhaust turmoil's virtual clock under heavy CI load;
/// the 30s wall-clock deadline still catches genuine deadlocks without any wall-clock sleeps.
fn run_sim_with_background_cleanup(sim: &mut Sim<'_>) -> turmoil::Result {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if sim.step()? {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(vortex_err!(
                "simulation timed out waiting for background multipart cleanup"
            )
            .into());
        }
        std::thread::yield_now();
    }
}

fn test_session() -> VortexResult<VortexSession> {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with_tokio();
    vortex_file::register_default_encodings(&session);
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    session.enable_edition(CORE_2026_08_3)?;
    Ok(session)
}

async fn encode_sample_vortex_file(session: &VortexSession) -> VortexResult<(Bytes, ArrayRef)> {
    let values: Buffer<i32> = (0..1024).collect();
    let expected = StructArray::from_fields(&[(
        "id",
        PrimitiveArray::new(values, Validity::NonNullable).into_array(),
    )])?
    .into_array();

    let mut out = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(Arc::new(FlatLayoutStrategy::default()))
        .write(&mut out, expected.clone().to_array_stream())
        .await?;
    Ok((Bytes::copy_from_slice(out.as_ref()), expected))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Transient HTTP 500, 503, and 429 responses recover within `RetryConfig` budget and allow a
    /// full Vortex file open and scan through `ObjectStoreReadAt`.
    #[test]
    fn test_transient_http_errors_retry_and_read_vortex_file() -> turmoil::Result {
        let mut sim = make_sim();
        let (file_bytes, expected) = {
            let session = test_session()?;
            TokioBuilder::new_current_thread()
                .build()?
                .block_on(encode_sample_vortex_file(&session))?
        };

        let state = Arc::new(ServerState {
            object: file_bytes,
            scripted_faults: Mutex::new(VecDeque::from([
                Fault::Status(500, "Internal Server Error"),
                Fault::Status(503, "Service Unavailable"),
                Fault::Status(429, "Too Many Requests"),
            ])),
            ..Default::default()
        });
        attach_server(&mut sim, "server", &state);

        sim.client("client", async move {
            let store = http_store("http://server:80", 5, DEFAULT_CONNECT_TIMEOUT)?;
            let session = test_session()?;
            let file = session
                .open_options()
                .open_object_store(&store, ObjectPath::from("data.vortex"))
                .await?;
            let chunks: Vec<ArrayRef> = file.scan()?.into_array_stream()?.try_collect().await?;
            assert_eq!(chunks.len(), 1);
            let mut ctx = session.create_execution_ctx();
            assert_arrays_eq!(&chunks[0], &expected, &mut ctx);
            Ok(())
        });

        sim.run()?;
        // 3 injected failures + the successful HEAD/GET requests for footer & data segments.
        assert!(state.requests.lock().len() >= 5);
        assert!(sim.elapsed() >= Duration::from_millis(75));
        Ok(())
    }

    /// Persistent retryable HTTP status fails cleanly once `max_retries` is exhausted, reporting
    /// both the retry count and HTTP status code.
    #[test]
    fn test_persistent_http_error_exhausts_retry_budget() -> turmoil::Result {
        let mut sim = make_sim();
        let state = Arc::new(ServerState {
            object: Bytes::from_static(&[0x42; 256]),
            default_fault: Some(Fault::Status(503, "Service Unavailable")),
            ..Default::default()
        });
        attach_server(&mut sim, "server", &state);

        sim.client("client", async move {
            let msg = read_range_err("http://server:80", 2, DEFAULT_CONNECT_TIMEOUT, 0, 64)
                .await?
                .to_string();
            assert!(
                msg.contains("after 2 retries") && msg.contains("503"),
                "unexpected error message: {msg}"
            );
            Ok(())
        });

        sim.run()?;
        assert_eq!(state.requests.lock().len(), 3);
        Ok(())
    }

    /// Mid-stream TCP drop after a partial `206` body fails cleanly with a transport error when
    /// retries are disabled (`max_retries = 0`).
    #[test]
    fn test_mid_stream_tcp_reset_fails_when_retries_disabled() -> turmoil::Result {
        let mut sim = make_sim();
        let state = Arc::new(ServerState {
            object: Bytes::from((0u8..200).collect::<Vec<_>>()),
            default_fault: Some(Fault::TruncateTcp { prefix_bytes: 40 }),
            ..Default::default()
        });
        attach_server(&mut sim, "server", &state);

        sim.client("client", async move {
            let err =
                read_range_err("http://server:80", 0, DEFAULT_CONNECT_TIMEOUT, 0, 100).await?;
            assert_eq!(
                find_source::<HttpError>(&err).map(HttpError::kind),
                Some(HttpErrorKind::Interrupted),
                "expected interrupted HTTP error, got: {err}"
            );
            Ok(())
        });

        sim.run()?;
        assert_eq!(state.requests.lock().len(), 1);
        Ok(())
    }

    /// Mid-stream TCP drop with retries enabled resumes from the offset already delivered and
    /// reconstructs the exact requested range.
    #[test]
    fn test_mid_stream_tcp_reset_resumes_range_on_retry() -> turmoil::Result {
        let mut sim = make_sim();
        let payload = Bytes::from((0u8..200).collect::<Vec<_>>());
        let expected = payload.slice(10..110);
        let state = Arc::new(ServerState {
            object: payload,
            scripted_faults: Mutex::new(VecDeque::from([Fault::TruncateTcp { prefix_bytes: 40 }])),
            ..Default::default()
        });
        attach_server(&mut sim, "server", &state);

        sim.client("client", async move {
            let store = http_store("http://server:80", 2, DEFAULT_CONNECT_TIMEOUT)?;
            let reader = ObjectStoreReadAt::new(
                store,
                ObjectPath::from("resume.bin"),
                TokioRuntime::current(),
            );
            let buffer = reader.read_at(10, 100, Alignment::new(1)).await?;
            assert_eq!(buffer.to_host().await.as_slice(), expected.as_ref());
            Ok(())
        });

        sim.run()?;
        let requests = state.requests.lock();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].range.as_deref(), Some("bytes=10-109"));
        assert_eq!(requests[1].range.as_deref(), Some("bytes=50-109"));
        Ok(())
    }

    /// Valid HTTP `206` responses whose body length disagrees with the requested byte range hit
    /// `ObjectStoreReadAt`'s explicit underflow and overflow guards.
    #[rstest]
    #[case::too_few_bytes(60, "Object store stream returned too few bytes")]
    #[case::too_many_bytes(120, "Object store stream returned too many bytes")]
    fn test_range_body_length_mismatch_rejected(
        #[case] actual_len: usize,
        #[case] expected_error: &'static str,
    ) -> turmoil::Result {
        let mut sim = make_sim();
        let state = Arc::new(ServerState {
            object: Bytes::from((0u8..200).collect::<Vec<_>>()),
            default_fault: Some(Fault::BodyLengthMismatch { actual_len }),
            ..Default::default()
        });
        attach_server(&mut sim, "server", &state);

        sim.client("client", async move {
            let msg = read_range_err("http://server:80", 0, DEFAULT_CONNECT_TIMEOUT, 0, 100)
                .await?
                .to_string();
            assert!(
                msg.contains(expected_error),
                "expected {expected_error:?} in error: {msg}"
            );
            Ok(())
        });

        sim.run()?;
        Ok(())
    }

    /// Both an active network partition (`partition`/`repair`) and a blackhole packet hold
    /// (`hold`/`release` hitting connect timeout) retry and recover once connectivity is restored.
    #[rstest]
    #[case::partition_repair(false, 80)]
    #[case::blackhole_hold_timeout(true, 300)]
    fn test_network_partition_recovers_after_repair(
        #[case] use_hold: bool,
        #[case] heal_after_ms: u64,
    ) -> turmoil::Result {
        let mut sim = make_sim();
        let payload = Bytes::from_static(b"deterministic partition recovery payload");
        let expected = payload.clone();
        let state = Arc::new(ServerState {
            object: payload,
            ..Default::default()
        });
        attach_server(&mut sim, "server", &state);

        let heal_after = Duration::from_millis(heal_after_ms);
        sim.client("healer", async move {
            time::sleep(heal_after).await;
            if use_hold {
                turmoil::release("client", "server");
            } else {
                turmoil::repair("client", "server");
            }
            Ok(())
        });

        sim.client("client", async move {
            if use_hold {
                turmoil::hold("client", "server");
            } else {
                turmoil::partition("client", "server");
            }
            // Exceeds turmoil's default 200 ms max SYN+SYN-ACK handshake latency so healthy
            // post-heal connects never spuriously time out.
            let store = http_store("http://server:80", 6, Duration::from_millis(250))?;
            let reader = ObjectStoreReadAt::new(
                store,
                ObjectPath::from("healed.bin"),
                TokioRuntime::current(),
            );
            let buffer = reader.read_at(0, expected.len(), Alignment::new(1)).await?;
            assert_eq!(buffer.to_host().await.as_slice(), expected.as_ref());
            Ok(())
        });

        sim.run()?;
        assert_eq!(state.requests.lock().len(), 1);
        assert!(sim.elapsed() >= heal_after);
        Ok(())
    }

    /// Connecting to a host with no listener propagates a connection-refused error through
    /// `ObjectStoreReadAt`.
    #[test]
    fn test_connection_refused_error_propagates() -> turmoil::Result {
        let mut sim = make_sim();
        sim.host("closed-server", || async {
            future::pending::<()>().await;
            Ok(())
        });

        sim.client("client", async move {
            let msg = read_range_err(
                "http://closed-server:80",
                1,
                Duration::from_millis(50),
                0,
                16,
            )
            .await?
            .to_string();
            assert!(
                msg.contains("connection refused"),
                "unexpected error message: {msg}"
            );
            Ok(())
        });

        sim.run()?;
        Ok(())
    }

    /// Multi-chunk `ObjectStoreWrite` crossing the 16 MiB part threshold sends
    /// `DELETE ?uploadId=...` to abort the multipart upload when either a part upload or final
    /// completion fails.
    #[rstest]
    #[case::put_part_failure(Some(2), false, "injected put_part failure")]
    #[case::complete_failure(None, true, "injected complete_multipart failure")]
    fn test_multipart_upload_failure_aborts_upload_on_server(
        #[case] fail_part_number: Option<usize>,
        #[case] fail_complete: bool,
        #[case] expected_error: &'static str,
    ) -> turmoil::Result {
        let mut sim = make_sim_with_duration(Duration::from_secs(86_400));
        let state = Arc::new(ServerState {
            fail_part_number,
            fail_complete,
            ..Default::default()
        });
        attach_server(&mut sim, "s3", &state);

        sim.client("client", async move {
            let store = s3_store("http://s3:80")?;
            let mut writer = ObjectStoreWrite::new(store, &ObjectPath::from("out.vortex")).await?;

            // Write 16 MiB + 1 KiB so part 1 is dispatched during `write_all` and part 2 on
            // `shutdown`.
            writer
                .write_all(vec![0x5Au8; MULTIPART_CHUNK_SIZE + 1024])
                .await?;

            let Err(err) = writer.shutdown().await else {
                return Err(vortex_err!("expected multipart shutdown failure").into());
            };
            let msg = err.to_string();
            assert!(
                msg.contains(expected_error),
                "expected {expected_error:?} in error: {msg}"
            );
            Ok(())
        });

        run_sim_with_background_cleanup(&mut sim)?;

        assert_eq!(*state.aborted_uploads.lock(), vec!["sim-upload-1"]);
        let mut put_parts: Vec<(usize, usize)> = state
            .requests
            .lock()
            .iter()
            .filter_map(|req| {
                (req.method == "PUT")
                    .then(|| {
                        query_param(&req.target, "partNumber")?
                            .parse::<usize>()
                            .ok()
                    })
                    .flatten()
                    .map(|part_num| (part_num, req.body_len))
            })
            .collect();
        put_parts.sort_unstable();
        assert_eq!(put_parts, vec![(1, MULTIPART_CHUNK_SIZE), (2, 1024)]);
        Ok(())
    }

    /// Calling `ObjectStoreWrite::abort()` after submitting a 16 MiB part waits for in-flight
    /// parts and sends `DELETE ?uploadId=...` to clean up the upload.
    #[test]
    fn test_multipart_explicit_abort_cleans_up_upload() -> turmoil::Result {
        let mut sim = make_sim_with_duration(Duration::from_secs(86_400));
        let state = Arc::new(ServerState::default());
        attach_server(&mut sim, "s3", &state);

        sim.client("client", async move {
            let store = s3_store("http://s3:80")?;
            let mut writer =
                ObjectStoreWrite::new(store, &ObjectPath::from("aborted.vortex")).await?;
            writer
                .write_all(vec![0x33u8; MULTIPART_CHUNK_SIZE + 512])
                .await?;
            writer.abort().await?;
            Ok(())
        });

        run_sim_with_background_cleanup(&mut sim)?;
        assert_eq!(*state.aborted_uploads.lock(), vec!["sim-upload-1"]);
        let requests = state.requests.lock();
        let put_idx = requests
            .iter()
            .position(|req| {
                req.method == "PUT"
                    && query_param(&req.target, "partNumber") == Some("1")
                    && req.body_len == MULTIPART_CHUNK_SIZE
            })
            .ok_or_else(|| vortex_err!("expected part 1 PUT before multipart abort"))?;
        let delete_idx = requests
            .iter()
            .position(|req| req.method == "DELETE" && req.target.contains("uploadId=sim-upload-1"))
            .ok_or_else(|| vortex_err!("expected DELETE abort for sim-upload-1"))?;
        assert!(
            put_idx < delete_idx,
            "expected part 1 PUT (index {put_idx}) before DELETE abort (index {delete_idx})"
        );
        Ok(())
    }
}
