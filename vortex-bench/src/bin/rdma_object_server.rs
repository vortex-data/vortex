// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use futures::StreamExt;
use futures::TryStreamExt;
use object_store::ObjectStore;
use object_store::ObjectStoreScheme;
use object_store::aws::AmazonS3;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectStorePath;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tracing::error;
use tracing::info;
use tracing_subscriber::EnvFilter;
use url::Url;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_bench::rdma_proto::DEFAULT_RDMA_PORT;
use vortex_bench::rdma_proto::OP_LIST;
use vortex_bench::rdma_proto::OP_READ;
use vortex_bench::rdma_proto::OP_SIZE;
use vortex_bench::rdma_proto::STATUS_OK;
use vortex_bench::rdma_proto::read_string;
use vortex_bench::rdma_proto::write_error;
use vortex_bench::rdma_proto::write_string;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Demo RDMA object server: preload S3 prefix to memory and serve range reads"
)]
struct Args {
    /// S3 prefix to preload (must end with '/'), e.g. s3://bucket/path/to/prefix/
    #[arg(long)]
    source: String,
    /// TCP bind address for the server.
    #[arg(long, default_value = "0.0.0.0:9900")]
    bind: String,
    /// Number of concurrent S3 downloads during warmup.
    #[arg(long, default_value_t = 32)]
    download_concurrency: usize,
}

struct CachedObject {
    data: Arc<[u8]>,
    size: u64,
}

#[derive(Clone)]
struct CachedStore {
    objects: Arc<HashMap<String, CachedObject>>,
    keys: Arc<Vec<(String, u64)>>,
    total_bytes: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    anyhow::ensure!(
        args.source.ends_with('/'),
        "--source must be a prefix ending with '/'"
    );

    let cache = preload_s3_prefix(&args.source, args.download_concurrency.max(1)).await?;
    info!(
        objects = cache.keys.len(),
        total_bytes = cache.total_bytes,
        "warmup complete, accepting connections"
    );

    let listener = TcpListener::bind(&args.bind)
        .await
        .with_context(|| format!("failed to bind {}", args.bind))?;
    info!(bind = %args.bind, default_rdma_port = DEFAULT_RDMA_PORT, "server listening");

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        let cache = cache.clone();
        tokio::spawn(async move {
            if let Err(err) = handle_client(stream, cache).await {
                error!(%peer_addr, error = %err, "connection failed");
            }
        });
    }
}

async fn preload_s3_prefix(source: &str, download_concurrency: usize) -> Result<CachedStore> {
    let url = Url::parse(source)?;
    let (scheme, prefix) = ObjectStoreScheme::parse(&url).map_err(object_store::Error::from)?;
    anyhow::ensure!(
        scheme == ObjectStoreScheme::AmazonS3,
        "only s3:// is supported"
    );

    let store = Arc::new(AmazonS3Builder::from_env().with_url(source).build()?) as Arc<AmazonS3>;
    let store_dyn: Arc<dyn ObjectStore> = store.clone();

    let mut objects = store_dyn
        .list(Some(&prefix))
        .try_collect::<Vec<_>>()
        .await
        .with_context(|| format!("failed to list prefix {source}"))?;
    objects.sort_by(|a, b| a.location.cmp(&b.location));
    anyhow::ensure!(!objects.is_empty(), "no objects found under {source}");
    info!(count = objects.len(), "found objects in prefix");

    let loaded = futures::stream::iter(objects.into_iter())
        .map(|meta| {
            let store_dyn = store_dyn.clone();
            async move {
                let bytes = download_object(&store_dyn, &meta.location, meta.size).await?;
                Ok::<_, anyhow::Error>((meta.location.to_string(), bytes))
            }
        })
        .buffer_unordered(download_concurrency)
        .try_collect::<Vec<_>>()
        .await?;

    let mut map = HashMap::with_capacity(loaded.len());
    let mut keys = Vec::with_capacity(loaded.len());
    let mut total_bytes = 0u64;
    for (key, data) in loaded {
        let size = u64::try_from(data.len()).context("cached object exceeds u64 length")?;
        total_bytes = total_bytes.saturating_add(size);
        keys.push((key.clone(), size));
        map.insert(key, CachedObject { data, size });
    }
    keys.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(CachedStore {
        objects: Arc::new(map),
        keys: Arc::new(keys),
        total_bytes,
    })
}

async fn download_object(
    store: &Arc<dyn ObjectStore>,
    path: &ObjectStorePath,
    expected_size: u64,
) -> Result<Arc<[u8]>> {
    let expected_len = usize::try_from(expected_size).context("object size exceeds usize")?;
    let response = store.get(path).await?;
    let mut stream = response.into_stream();
    let mut out = Vec::with_capacity(expected_len);
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk?);
    }
    anyhow::ensure!(
        out.len() == expected_len,
        "downloaded {} bytes for {}, expected {}",
        out.len(),
        path,
        expected_size
    );
    Ok(Arc::from(out.into_boxed_slice()))
}

async fn handle_client(mut stream: TcpStream, cache: CachedStore) -> Result<()> {
    loop {
        let op = match stream.read_u8().await {
            Ok(op) => op,
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(err) => return Err(err.into()),
        };

        match op {
            OP_LIST => {
                let prefix = read_string(&mut stream).await?;
                stream.write_u8(STATUS_OK).await?;
                let matching: Vec<_> = cache
                    .keys
                    .iter()
                    .filter(|(key, _)| key.starts_with(prefix.as_str()))
                    .collect();
                stream
                    .write_u32_le(u32::try_from(matching.len()).context("too many list results")?)
                    .await?;
                for (key, size) in matching {
                    write_string(&mut stream, key).await?;
                    stream.write_u64_le(*size).await?;
                }
            }
            OP_SIZE => {
                let key = read_string(&mut stream).await?;
                if let Some(obj) = cache.objects.get(key.as_str()) {
                    stream.write_u8(STATUS_OK).await?;
                    stream.write_u64_le(obj.size).await?;
                } else {
                    write_error(&mut stream, &format!("object not found: {key}")).await?;
                }
            }
            OP_READ => {
                let key = read_string(&mut stream).await?;
                let offset = stream.read_u64_le().await?;
                let length_u32 = stream.read_u32_le().await?;
                let length = usize::try_from(length_u32).context("request length exceeds usize")?;
                let Some(obj) = cache.objects.get(key.as_str()) else {
                    write_error(&mut stream, &format!("object not found: {key}")).await?;
                    continue;
                };
                let start = usize::try_from(offset).context("offset exceeds usize")?;
                let end = start.saturating_add(length);
                if end > obj.data.len() {
                    write_error(
                        &mut stream,
                        &format!(
                            "range {}..{} out of bounds for object {} size {}",
                            start,
                            end,
                            key,
                            obj.data.len()
                        ),
                    )
                    .await?;
                    continue;
                }

                stream.write_u8(STATUS_OK).await?;
                stream.write_u32_le(length_u32).await?;
                stream.write_all(&obj.data[start..end]).await?;
            }
            other => {
                write_error(&mut stream, &format!("unknown opcode: {other}")).await?;
            }
        }
    }
}
