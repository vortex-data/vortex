// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use bytes::Bytes;
use futures::TryStreamExt;
use futures::stream;
use http::Method;
use http::Request;
use http::Response;
use http::StatusCode;
use http::header;
use http_body_util::BodyExt;
use http_body_util::Full;
use http_body_util::Limited;
use http_body_util::StreamBody;
use http_body_util::combinators::UnsyncBoxBody;
use hyper::body::Frame;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use super::Profile;

type Body = UnsyncBoxBody<Bytes, Infallible>;

#[derive(Debug, Default)]
struct Metrics {
    active: AtomicU64,
    peak: AtomicU64,
    get: AtomicU64,
    head: AtomicU64,
    list: AtomicU64,
    put: AtomicU64,
    delete: AtomicU64,
    bytes: AtomicU64,
    request_us: AtomicU64,
}

struct Guard {
    metrics: Arc<Metrics>,
    started: Instant,
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.metrics.active.fetch_sub(1, Ordering::Relaxed);
        self.metrics.request_us.fetch_add(
            u64::try_from(self.started.elapsed().as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }
}

#[derive(Debug)]
struct State {
    store: InMemory,
    profile: Profile,
    bucket: String,
    limit: u64,
    resident: tokio::sync::Mutex<u64>,
    ordinals: Mutex<BTreeMap<(String, String), u64>>,
    metrics: Arc<Metrics>,
    started: Instant,
}

/// A loopback S3 read endpoint whose objects remain in RAM for its lifetime.
///
/// The HTTP protocol allows DataFusion, DuckDB and other clients to use the same model.
/// PUT and DELETE support benchmark preparation; only reads carry learned service delays.
#[derive(Debug)]
pub struct Server {
    address: SocketAddr,
    state: Arc<State>,
    task: JoinHandle<()>,
}

impl Server {
    /// Loads a local directory into RAM and starts an HTTP endpoint on a loopback address.
    ///
    /// `max_bytes` bounds resident object data; callers reserve the remaining RAM for query work.
    pub async fn start(
        root: Option<&Path>,
        profile: Profile,
        bucket: String,
        bind: SocketAddr,
        max_bytes: u64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            bind.ip().is_loopback(),
            "The unauthenticated benchmark endpoint must bind loopback"
        );
        anyhow::ensure!(
            !bucket.is_empty() && max_bytes > 0,
            "Bucket and resident byte limit are required"
        );
        let state = Arc::new(State {
            store: InMemory::new(),
            profile,
            bucket,
            limit: max_bytes,
            resident: tokio::sync::Mutex::new(0),
            ordinals: Mutex::new(BTreeMap::new()),
            metrics: Arc::new(Metrics::default()),
            started: Instant::now(),
        });
        if let Some(root) = root {
            let root = root.canonicalize()?;
            let mut directories = vec![root.clone()];
            while let Some(directory) = directories.pop() {
                for entry in std::fs::read_dir(directory)? {
                    let entry = entry?;
                    let kind = entry.file_type()?;
                    if kind.is_dir() {
                        directories.push(entry.path());
                        continue;
                    }
                    anyhow::ensure!(
                        kind.is_file(),
                        "RAM snapshots require regular files: {}",
                        entry.path().display()
                    );
                    let path = entry.path();
                    let key = ObjectPath::from(
                        path.strip_prefix(&root)?
                            .to_str()
                            .context("Non-UTF-8 snapshot path")?,
                    );
                    let size = entry.metadata()?.len();
                    let mut resident = state.resident.lock().await;
                    anyhow::ensure!(
                        *resident <= max_bytes.saturating_sub(size) && size <= max_bytes,
                        "Dataset exceeds the RAM budget; leave working memory for the benchmark"
                    );
                    let bytes = tokio::fs::read(path).await?;
                    anyhow::ensure!(
                        bytes.len() as u64 == size,
                        "Snapshot file changed while reading"
                    );
                    state.store.put(&key, bytes.into()).await?;
                    *resident += size;
                }
            }
        }
        let listener = TcpListener::bind(bind).await?;
        let address = listener.local_addr()?;
        let shared = Arc::clone(&state);
        let task = tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    break;
                };
                // Delayed body frames must not acquire an additional Nagle/delayed-ACK stall.
                if let Err(error) = socket.set_nodelay(true) {
                    tracing::warn!(%error, "Cannot configure benchmark TCP connection");
                    continue;
                }
                let state = Arc::clone(&shared);
                tokio::spawn(async move {
                    let service = service_fn(move |request| {
                        let state = Arc::clone(&state);
                        async move {
                            let response =
                                route(Arc::clone(&state), request)
                                    .await
                                    .unwrap_or_else(|error| {
                                        failure(
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            "InternalError",
                                            &error.to_string(),
                                        )
                                    });
                            Ok::<_, Infallible>(response)
                        }
                    });
                    let _connection_result = http1::Builder::new()
                        .serve_connection(TokioIo::new(socket), service)
                        .await;
                });
            }
        });
        Ok(Self {
            address,
            state,
            task,
        })
    }

    /// The endpoint passed to S3 clients with path-style addressing and HTTP enabled.
    pub fn endpoint(&self) -> String {
        format!("http://{}", self.address)
    }

    /// Bytes currently resident as object data, excluding client buffers and query working memory.
    pub async fn resident_bytes(&self) -> u64 {
        *self.state.resident.lock().await
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn full(status: StatusCode, content_type: &'static str, data: impl Into<Bytes>) -> Response<Body> {
    let mut response = Response::new(Full::new(data.into()).boxed_unsync());
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static(content_type),
    );
    response
}

fn xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn failure(status: StatusCode, code: &str, message: &str) -> Response<Body> {
    full(
        status,
        "application/xml",
        format!(
            "<Error><Code>{}</Code><Message>{}</Message></Error>",
            xml(code),
            xml(message)
        ),
    )
}

async fn metadata_delay(state: &State, started: Instant) -> anyhow::Result<()> {
    let deadline = started
        .checked_add(Duration::from_micros(state.profile.metadata_header_us()))
        .context("Invalid metadata deadline")?;
    tokio::time::sleep_until(deadline.into()).await;
    Ok(())
}

fn object_headers(
    response: &mut Response<Body>,
    meta: &ObjectMeta,
    length: u64,
) -> anyhow::Result<()> {
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_LENGTH, length.into());
    headers.insert(
        header::ACCEPT_RANGES,
        header::HeaderValue::from_static("bytes"),
    );
    headers.insert(
        header::ETAG,
        format!(
            "\"{}\"",
            meta.e_tag.as_deref().unwrap_or("0").trim_matches('"')
        )
        .parse()?,
    );
    headers.insert(
        header::LAST_MODIFIED,
        meta.last_modified
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string()
            .parse()?,
    );
    Ok(())
}

fn range(value: Option<&header::HeaderValue>) -> anyhow::Result<Option<GetRange>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value
        .to_str()?
        .strip_prefix("bytes=")
        .context("Use a byte range")?;
    let (start, end) = value.split_once('-').context("Invalid byte range")?;
    Ok(Some(if start.is_empty() {
        GetRange::Suffix(end.parse()?)
    } else if end.is_empty() {
        GetRange::Offset(start.parse()?)
    } else {
        GetRange::Bounded(
            start.parse()?
                ..end
                    .parse::<u64>()?
                    .checked_add(1)
                    .context("Range overflow")?,
        )
    }))
}

async fn route(state: Arc<State>, request: Request<Incoming>) -> anyhow::Result<Response<Body>> {
    if request.uri().path() == "/_vortex/metrics" {
        let metrics = &state.metrics;
        return Ok(full(StatusCode::OK, "application/json", json!({
            "scope": "HTTP requests observed by the simulated endpoint", "get": metrics.get.load(Ordering::Relaxed),
            "head": metrics.head.load(Ordering::Relaxed), "list": metrics.list.load(Ordering::Relaxed),
            "put": metrics.put.load(Ordering::Relaxed), "delete": metrics.delete.load(Ordering::Relaxed),
            "bytes": metrics.bytes.load(Ordering::Relaxed), "active": metrics.active.load(Ordering::Relaxed),
            "peak_inflight": metrics.peak.load(Ordering::Relaxed), "resident_bytes": *state.resident.lock().await,
            "avg_inflight": metrics.request_us.load(Ordering::Relaxed) as f64 / (state.started.elapsed().as_secs_f64() * 1000000.0)
        }).to_string()));
    }
    let decoded = ObjectPath::from_url_path(request.uri().path())?;
    let (bucket, key) = decoded
        .as_ref()
        .split_once('/')
        .unwrap_or_else(|| (decoded.as_ref(), ""));
    if bucket != state.bucket {
        return Ok(failure(
            StatusCode::NOT_FOUND,
            "NoSuchBucket",
            "Unknown benchmark bucket",
        ));
    }
    let key = ObjectPath::from(key);
    let query = url::form_urlencoded::parse(request.uri().query().unwrap_or("").as_bytes())
        .into_owned()
        .collect::<BTreeMap<_, _>>();
    let active = state.metrics.active.fetch_add(1, Ordering::Relaxed) + 1;
    state.metrics.peak.fetch_max(active, Ordering::Relaxed);
    let guard = Guard {
        metrics: Arc::clone(&state.metrics),
        started: Instant::now(),
    };
    if key.as_ref().is_empty() && request.method() == Method::POST && query.contains_key("delete") {
        return batch_delete(&state, request).await;
    }
    if key.as_ref().is_empty()
        && (request.method() == Method::GET || request.method() == Method::HEAD)
    {
        metadata_delay(&state, guard.started).await?;
        if request.method() == Method::HEAD {
            state.metrics.head.fetch_add(1, Ordering::Relaxed);
            return Ok(full(StatusCode::OK, "application/xml", Bytes::new()));
        }
        if query.contains_key("location") {
            return Ok(full(
                StatusCode::OK,
                "application/xml",
                "<LocationConstraint>us-east-1</LocationConstraint>",
            ));
        }
        state.metrics.list.fetch_add(1, Ordering::Relaxed);
        return listing(&state, &query).await;
    }
    match *request.method() {
        Method::GET | Method::HEAD => {
            if request.method() == Method::HEAD {
                state.metrics.head.fetch_add(1, Ordering::Relaxed);
            } else {
                state.metrics.get.fetch_add(1, Ordering::Relaxed);
            }
            let meta = match state.store.head(&key).await {
                Ok(meta) => meta,
                Err(_) => {
                    return Ok(failure(
                        StatusCode::NOT_FOUND,
                        "NoSuchKey",
                        "Object does not exist",
                    ));
                }
            };
            let etag = meta.e_tag.as_deref().unwrap_or("0").trim_matches('"');
            if request
                .headers()
                .get(header::IF_MATCH)
                .is_some_and(|value| {
                    value
                        .to_str()
                        .ok()
                        .is_none_or(|value| value != "*" && value.trim_matches('"') != etag)
                })
            {
                return Ok(failure(
                    StatusCode::PRECONDITION_FAILED,
                    "PreconditionFailed",
                    "ETag differs",
                ));
            }
            if request
                .headers()
                .get(header::IF_NONE_MATCH)
                .is_some_and(|value| {
                    value
                        .to_str()
                        .is_ok_and(|value| value == "*" || value.trim_matches('"') == etag)
                })
            {
                return Ok(full(
                    StatusCode::NOT_MODIFIED,
                    "application/octet-stream",
                    Bytes::new(),
                ));
            }
            if request.method() == Method::HEAD {
                metadata_delay(&state, guard.started).await?;
                let mut response = full(StatusCode::OK, "application/octet-stream", Bytes::new());
                object_headers(&mut response, &meta, meta.size)?;
                return Ok(response);
            }
            let requested = match range(request.headers().get(header::RANGE)) {
                Ok(range) => range,
                Err(_) => {
                    return Ok(failure(
                        StatusCode::RANGE_NOT_SATISFIABLE,
                        "InvalidRange",
                        "Invalid byte range",
                    ));
                }
            };
            let actual = match requested
                .as_ref()
                .map(|range| range.as_range(meta.size))
                .transpose()
            {
                Ok(range) => range.unwrap_or(0..meta.size),
                Err(_) => {
                    return Ok(failure(
                        StatusCode::RANGE_NOT_SATISFIABLE,
                        "InvalidRange",
                        "Range outside object",
                    ));
                }
            };
            let data = state
                .store
                .get_opts(
                    &key,
                    GetOptions {
                        range: requested.clone(),
                        ..GetOptions::default()
                    },
                )
                .await?
                .bytes()
                .await?;
            let range_key = format!("{}..{}", actual.start, actual.end);
            let ordinal = {
                let mut ordinals = state.ordinals.lock();
                let ordinal = ordinals
                    .entry((key.to_string(), range_key.clone()))
                    .or_default();
                let current = *ordinal;
                *ordinal = ordinal.wrapping_add(1);
                current
            };
            let timing = state.profile.timing(
                key.as_ref(),
                &range_key,
                data.len() as u64,
                active,
                ordinal,
            )?;
            let header_ready = guard
                .started
                .checked_add(Duration::from_micros(timing.header_us))
                .context("Invalid header deadline")?;
            let first_ready = header_ready
                .checked_add(Duration::from_micros(timing.gap_us))
                .context("Invalid first-byte deadline")?;
            let end = first_ready
                .checked_add(Duration::from_micros(timing.transfer_us))
                .context("Invalid transfer deadline")?;
            tokio::time::sleep_until(header_ready.into()).await;
            let length = data.len();
            let body = stream::unfold(
                (data, 0usize, Some(guard)),
                move |(data, offset, guard)| async move {
                    if offset == data.len() {
                        return None;
                    }
                    let next = (offset + 16384).min(data.len());
                    let first = 16384.min(data.len());
                    let deadline = if length <= first {
                        end
                    } else if next == first {
                        first_ready
                    } else {
                        let micros = u128::from(timing.transfer_us) * (next - first) as u128
                            / (length - first) as u128;
                        first_ready
                            + Duration::from_micros(
                                u64::try_from(micros).unwrap_or(timing.transfer_us),
                            )
                    };
                    tokio::time::sleep_until(deadline.into()).await;
                    if let Some(guard) = &guard {
                        guard
                            .metrics
                            .bytes
                            .fetch_add((next - offset) as u64, Ordering::Relaxed);
                    }
                    Some((
                        Ok::<_, Infallible>(Frame::data(data.slice(offset..next))),
                        (data, next, guard),
                    ))
                },
            );
            let mut response = Response::new(StreamBody::new(body).boxed_unsync());
            if requested.is_some() && length > 0 {
                *response.status_mut() = StatusCode::PARTIAL_CONTENT;
                response.headers_mut().insert(
                    header::CONTENT_RANGE,
                    format!("bytes {}-{}/{}", actual.start, actual.end - 1, meta.size).parse()?,
                );
            }
            object_headers(&mut response, &meta, length as u64)?;
            Ok(response)
        }
        Method::PUT => {
            state.metrics.put.fetch_add(1, Ordering::Relaxed);
            let mut resident = state.resident.lock().await;
            let previous = state.store.head(&key).await.map_or(0, |meta| meta.size);
            let available = state
                .limit
                .saturating_sub(*resident)
                .saturating_add(previous);
            let body = match Limited::new(
                request.into_body(),
                usize::try_from(available).unwrap_or(usize::MAX),
            )
            .collect()
            .await
            {
                Ok(body) => body.to_bytes(),
                Err(_) => {
                    return Ok(failure(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "EntityTooLarge",
                        "RAM budget exceeded",
                    ));
                }
            };
            let length = body.len() as u64;
            let result = state.store.put(&key, body.into()).await?;
            *resident = resident.saturating_sub(previous).saturating_add(length);
            let mut response = full(StatusCode::OK, "application/octet-stream", Bytes::new());
            if let Some(etag) = result.e_tag {
                response
                    .headers_mut()
                    .insert(header::ETAG, format!("\"{etag}\"").parse()?);
            }
            Ok(response)
        }
        Method::DELETE => {
            state.metrics.delete.fetch_add(1, Ordering::Relaxed);
            let mut resident = state.resident.lock().await;
            let previous = state.store.head(&key).await.map_or(0, |meta| meta.size);
            if previous > 0 || state.store.head(&key).await.is_ok() {
                state.store.delete(&key).await?;
            }
            *resident = resident.saturating_sub(previous);
            Ok(full(
                StatusCode::NO_CONTENT,
                "application/xml",
                Bytes::new(),
            ))
        }
        _ => Ok(failure(
            StatusCode::NOT_IMPLEMENTED,
            "NotImplemented",
            "The benchmark endpoint supports read operations and fixture PUT/DELETE",
        )),
    }
}

async fn listing(
    state: &State,
    query: &BTreeMap<String, String>,
) -> anyhow::Result<Response<Body>> {
    let prefix = query.get("prefix").map_or("", String::as_str);
    let delimiter = query.get("delimiter").filter(|value| !value.is_empty());
    let objects = state.store.list(None).try_collect::<Vec<_>>().await?;
    let mut entries = BTreeMap::new();
    for object in objects {
        let Some(suffix) = object.location.as_ref().strip_prefix(prefix) else {
            continue;
        };
        if let Some(delimiter) = delimiter
            && let Some((first, _)) = suffix.split_once(delimiter)
        {
            let common = format!("{prefix}{first}{delimiter}");
            entries.insert(
                common.clone(),
                format!(
                    "<CommonPrefixes><Prefix>{}</Prefix></CommonPrefixes>",
                    xml(&common)
                ),
            );
            continue;
        }
        let etag = format!(
            "\"{}\"",
            object.e_tag.as_deref().unwrap_or("0").trim_matches('"')
        );
        entries.insert(object.location.to_string(), format!("<Contents><Key>{}</Key><Size>{}</Size><ETag>{}</ETag><LastModified>{}</LastModified><StorageClass>STANDARD</StorageClass></Contents>",
            xml(object.location.as_ref()), object.size, xml(&etag), object.last_modified.to_rfc3339()));
    }
    let start = query
        .get("continuation-token")
        .or_else(|| query.get("marker"))
        .map_or("", String::as_str);
    let max = query
        .get("max-keys")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000)
        .clamp(1, 1000);
    let mut after = entries.into_iter().filter(|(key, _)| key.as_str() > start);
    let page = after.by_ref().take(max).collect::<Vec<_>>();
    let truncated = after.next().is_some();
    let mut result = format!(
        "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Name>{}</Name><Prefix>{}</Prefix><MaxKeys>{max}</MaxKeys><KeyCount>{}</KeyCount><IsTruncated>{truncated}</IsTruncated>",
        xml(&state.bucket),
        xml(prefix),
        page.len()
    );
    for (_, value) in &page {
        result.push_str(value);
    }
    if truncated && let Some((key, _)) = page.last() {
        let field = if query.get("list-type").is_some_and(|value| value == "2") {
            "NextContinuationToken"
        } else {
            "NextMarker"
        };
        result.push_str(&format!("<{field}>{}</{field}>", xml(key)));
    }
    result.push_str("</ListBucketResult>");
    Ok(full(StatusCode::OK, "application/xml", result))
}

async fn batch_delete(state: &State, request: Request<Incoming>) -> anyhow::Result<Response<Body>> {
    #[derive(Deserialize)]
    struct Delete {
        #[serde(rename = "Object", default)]
        objects: Vec<DeleteObject>,
    }
    #[derive(Deserialize)]
    struct DeleteObject {
        #[serde(rename = "Key")]
        key: String,
    }
    let body = Limited::new(request.into_body(), 1024 * 1024)
        .collect()
        .await
        .map_err(anyhow::Error::from_boxed)?
        .to_bytes();
    let delete: Delete = quick_xml::de::from_reader(body.as_ref())?;
    anyhow::ensure!(
        delete.objects.len() <= 1000,
        "S3 batch delete accepts at most 1000 keys"
    );
    state.metrics.delete.fetch_add(1, Ordering::Relaxed);
    let mut resident = state.resident.lock().await;
    let mut response =
        String::from("<DeleteResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">");
    for object in delete.objects {
        let key = ObjectPath::from(object.key.as_str());
        if let Ok(meta) = state.store.head(&key).await {
            state.store.delete(&key).await?;
            *resident = resident.saturating_sub(meta.size);
        }
        response.push_str(&format!(
            "<Deleted><Key>{}</Key></Deleted>",
            xml(&object.key)
        ));
    }
    response.push_str("</DeleteResult>");
    Ok(full(StatusCode::OK, "application/xml", response))
}
