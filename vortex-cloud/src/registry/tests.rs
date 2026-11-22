// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::path::Path;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex_session::VortexSession;

use super::Registry;

/// A registry whose S3 configuration comes from a fixed map rather than the process environment,
/// so these tests neither read nor mutate global state.
fn registry() -> Registry {
    Registry::with_vars([("AWS_REGION".to_string(), "us-east-3".to_string())])
}

/// A percent-encoded segment (as HuggingFace dataset URLs use for `refs/convert/parquet`) decodes
/// to more path parts than the raw URL has segments. Both the build-and-cache branch and the
/// cached-store branch must decode it rather than underflowing the segment-count subtraction.
#[test]
fn test_resolve_percent_encoded_path() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let url = Url::parse(
        "https://example.com/datasets/org/name/resolve/refs%2Fconvert%2Fparquet/dir/file.vortex",
    )?;
    let expected = Path::from("datasets/org/name/resolve/refs/convert/parquet/dir/file.vortex");

    // First resolution parses and registers the store; it must decode the path, not panic.
    let (_store, path) = registry.resolve(&url)?;
    assert_eq!(path, expected);

    // Second resolution takes the cached-store branch and must agree.
    let (_store, path) = registry.resolve(&url)?;
    assert_eq!(path, expected);
    Ok(())
}

#[test]
#[expect(clippy::use_debug)]
fn test_resolve_url() -> Result<(), Box<dyn std::error::Error>> {
    let (store, _) = registry().resolve(&Url::parse("s3://my-bucket/test")?)?;

    // NOTE(aduffy): object_store doesn't let us downcast stores, the only way to verify
    //  that a configuration was added was to validate that it ends up in the Debug
    //  output :/
    let mut debug_str = String::new();
    write!(&mut debug_str, "{store:?}")?;

    assert!(debug_str.contains("us-east-3"));
    Ok(())
}

/// Two resolves of the same URL must return the same `Arc` (the cached client) and the same
/// path, and two different keys must get distinct stores. This pins the symmetry the registry
/// promises: cache hit returns the same client and the same key.
#[test]
fn test_resolve_url_caches_per_key() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let first = Url::parse("s3://my-bucket/first/second")?;
    let second = Url::parse("s3://my-bucket/first/second")?;

    let (store_a, path_a) = registry.resolve(&first)?;
    let (store_b, path_b) = registry.resolve(&second)?;
    assert!(Arc::ptr_eq(&store_a, &store_b));
    assert_eq!(path_a, path_b);

    // A different bucket gets its own client.
    let other = Url::parse("s3://other-bucket/first/second")?;
    let (store_c, _) = registry.resolve(&other)?;
    assert!(!Arc::ptr_eq(&store_a, &store_c));
    Ok(())
}

/// Two objects in the same bucket must share a cached client. This is the regression that
/// the bespoke "walk every segment then return the whole key" path caused: by walking all
/// segments it stored the store at full depth, so the next resolve saw the whole key
/// already consumed and returned an empty path. Pinning the path here guards against that.
#[test]
fn test_resolve_url_shared_client_same_bucket() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let a = Url::parse("s3://my-bucket/path/to/data.vortex")?;
    let b = Url::parse("s3://my-bucket/other/data.vortex")?;

    let (store_a, path_a) = registry.resolve(&a)?;
    let (store_b, path_b) = registry.resolve(&b)?;
    assert!(Arc::ptr_eq(&store_a, &store_b));
    assert_eq!(path_a.as_ref(), "path/to/data.vortex");
    assert_eq!(path_b.as_ref(), "other/data.vortex");
    Ok(())
}

/// Pinning the `entry_at` helper at depth > 0 (the path-prefix case) protects the shared
/// `entry_at` / `cache_and_resolve` helpers from regressing prefix registration.
#[test]
fn test_register_at_prefix_shares_store() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let prefix = Url::parse("s3://my-bucket/prefix/")?;
    let (inner, _) = registry.resolve(&prefix)?;
    let cached: Arc<dyn ObjectStore> = Arc::clone(&inner);
    let replaced = registry.register(prefix.clone(), Arc::clone(&cached));
    // First registration at this prefix replaces nothing.
    assert!(replaced.is_none());

    // A resolve at a deeper key still walks through the registered prefix's store.
    let deeper = Url::parse("s3://my-bucket/prefix/inner/key")?;
    let (store_deeper, path_deeper) = registry.resolve(&deeper)?;
    assert!(Arc::ptr_eq(&store_deeper, &cached));
    assert_eq!(path_deeper.as_ref(), "inner/key");
    Ok(())
}

/// A `file://` URL with its path cleared must resolve to a local store. `vortex-duckdb` resolves
/// exactly this shape — it clears the path off the glob's base URL before asking for a
/// filesystem — so this pins the most common local-file case.
#[test]
fn test_resolve_file_url_with_empty_path() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();

    let mut base = Url::parse("file:///data/part-0.vortex")?;
    base.set_path("");
    let (store, path) = registry.resolve(&base)?;
    assert!(format!("{store:?}").contains("LocalFileSystem"));
    assert_eq!(path.as_ref(), "");

    // The same store is reused for a second glob rooted at the same authority.
    let (store_again, _) = registry.resolve(&base)?;
    assert!(Arc::ptr_eq(&store, &store_again));
    Ok(())
}

/// An explicitly registered store must win over the build-and-cache fallback, including for
/// schemes `object_store` does not recognize natively. `memory://` stands in for any such scheme
/// so this holds regardless of which service features are enabled.
#[test]
fn test_registered_store_wins_over_build() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let registered: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
    registry.register(Url::parse("memory://bucket/")?, Arc::clone(&registered));

    let (store, path) = registry.resolve(&Url::parse("memory://bucket/a/b.vortex")?)?;
    assert!(Arc::ptr_eq(&store, &registered));
    assert_eq!(path.as_ref(), "a/b.vortex");
    Ok(())
}

/// An `hf://` URL must resolve to a store rooted at the repository revision, leaving the
/// in-repository path as the object key. The repository and revision occupy URL path segments, so
/// this is the one scheme whose store is not mounted at the URL authority — getting the depth wrong
/// would send the repository name to the Hub as part of the file path.
#[cfg(feature = "hf")]
#[rstest::rstest]
#[case("hf://datasets/org/name/data/train.vortex", "data/train.vortex")]
#[case(
    "hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/train.vortex",
    "data/train.vortex"
)]
#[case("hf://org/name/model.vortex", "model.vortex")]
fn test_hf_scheme_mounts_at_the_repository(
    #[case] url: &str,
    #[case] expected: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let url = Url::parse(url)?;

    // First resolution builds and caches the store; the second takes the cached-store branch.
    // Both must report the same key, since the cached branch recomputes it from the mount depth.
    let (_store, path) = registry.resolve(&url)?;
    assert_eq!(path, Path::from(expected));
    let (_store, path) = registry.resolve(&url)?;
    assert_eq!(path, Path::from(expected));
    Ok(())
}

/// Two revisions of one repository are different stores, since the revision is part of the prefix
/// the store is rooted at.
#[cfg(feature = "hf")]
#[test]
fn test_hf_revisions_do_not_share_a_store() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();

    let (main, _) = registry.resolve(&Url::parse("hf://datasets/org/name/train.vortex")?)?;
    let (tagged, _) = registry.resolve(&Url::parse("hf://datasets/org/name@v2/train.vortex")?)?;

    assert!(!Arc::ptr_eq(&main, &tagged));
    Ok(())
}

/// The OpenDAL-backed schemes must resolve through the registry rather than falling through to
/// `parse_url_opts`, which does not recognize them.
///
/// COS and OSS need an endpoint that is not derivable from the URL, so a bare
/// `scheme://bucket/key.vortex` URL fails the build; that failure must come from the OpenDAL
/// builder ("missing required OpenDAL store configuration"), not from `object_store`'s
/// unrecognized-scheme path. GooseFS is different: its `master_addr` is taken from the URL
/// authority, so the same URL builds a store successfully — which itself proves it reached the
/// OpenDAL builder, since `parse_url_opts` would have rejected `goosefs://`.
#[cfg(any(feature = "cos", feature = "oss", feature = "goosefs"))]
#[test]
fn test_opendal_schemes_reach_the_opendal_builder() -> Result<(), Box<dyn std::error::Error>> {
    for scheme in crate::opendal::SUPPORTED_SCHEMES {
        let url = Url::parse(&format!("{scheme}://bucket/key.vortex"))?;
        match registry().resolve(&url) {
            // GooseFS derives `master_addr` from the URL authority, so the store builds
            // successfully. `parse_url_opts` does not recognize `goosefs://`, so an `Ok`
            // here proves the OpenDAL builder ran.
            Ok(_) => {}
            Err(err) => {
                let message = err.to_string();
                assert!(
                    message.contains("OpenDAL"),
                    "{scheme} did not reach the OpenDAL builder: {message}"
                );
            }
        }
    }
    Ok(())
}

/// The key the registry reports must be the object's *literal* key.
///
/// `~` is unreserved, so it travels through a URL verbatim; characters a URL has to escape decode
/// back to their literal form.
#[rstest::rstest]
#[case("repro~tilde~key/data.vortex", "repro~tilde~key/data.vortex")]
#[case("tablets/~~all~0/data.vortex", "tablets/~~all~0/data.vortex")]
#[case("dir/a[1].vortex", "dir/a[1].vortex")]
#[case("dir/a%23b.vortex", "dir/a#b.vortex")]
#[case("dir/a%20b.vortex", "dir/a b.vortex")]
#[case("dir/a%2520b.vortex", "dir/a%20b.vortex")]
fn test_resolve_reports_literal_key(
    #[case] url_path: &str,
    #[case] expected: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    registry.register(
        Url::parse("memory://bucket/")?,
        Arc::new(object_store::memory::InMemory::new()),
    );

    // Both the registered-store branch and the recomputed cached branch must agree.
    let url = Url::parse(&format!("memory://bucket/{url_path}"))?;
    let (_store, path) = registry.resolve(&url)?;
    assert_eq!(path.as_ref(), expected);
    let (_store, path) = registry.resolve(&url)?;
    assert_eq!(path.as_ref(), expected);
    Ok(())
}

#[rstest::rstest]
#[case::tilde("repro~tilde~key/data.vortex")]
#[case::double_tilde("tablets/~~all~0/data.vortex")]
#[case::brackets("dir/a[1].vortex")]
#[case::braces("dir/a{x}.vortex")]
#[case::caret("dir/a^b.vortex")]
#[case::pipe("dir/a|b.vortex")]
#[case::star("dir/a*b.vortex")]
#[case::space("dir/a b.vortex")]
#[case::plain("plain/data.vortex")]
#[tokio::test]
async fn test_resolved_key_reaches_the_object(
    #[case] key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use object_store::ObjectStoreExt;
    use vortex_file::OpenOptionsSessionExt;

    let session = VortexSession::empty()
        .with::<vortex_layout::session::LayoutSession>()
        .with::<vortex_io::session::RuntimeSession>();

    // Seed the store with the literal key, the way any other client would.
    let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
    let location = Path::parse(key)?;
    assert_eq!(
        location.as_ref(),
        key,
        "the store must hold the literal key"
    );
    store.put(&location, vec![0u8; 1024].into()).await?;

    let registry = registry();
    registry.register(Url::parse("memory://bucket/")?, Arc::clone(&store));
    let (store, path) = registry.resolve(&Url::parse(&format!("memory://bucket/{key}"))?)?;
    assert_eq!(
        path.as_ref(),
        key,
        "the registry must report the literal key"
    );

    let Err(err) = session.open_options().open_object_store(&store, path).await else {
        panic!("the payload is not a Vortex file, so opening it must fail")
    };
    assert!(
        err.to_string()
            .contains("Malformed file, invalid magic bytes, got [0, 0, 0, 0]"),
        "opening {key:?} must read the stored payload and reject its footer: {err}"
    );
    Ok(())
}

#[test]
fn test_unwrapped_s3_multipart_abort_and_drop_succeed_outside_idle_tokio_runtime()
-> Result<(), Box<dyn std::error::Error>> {
    use std::io::Read;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use object_store::ClientOptions;
    use object_store::RetryConfig;
    use object_store::aws::AmazonS3Builder;
    use vortex_io::VortexWrite;
    use vortex_io::object_store::ObjectStoreWrite;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::current::CurrentThreadRuntime;

    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let post_count = Arc::new(AtomicUsize::new(0));
    let put_count = Arc::new(AtomicUsize::new(0));
    let delete_count = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let posts_server = Arc::clone(&post_count);
    let puts_server = Arc::clone(&put_count);
    let deletes_server = Arc::clone(&delete_count);
    let stop_server = Arc::clone(&stop);

    // Keep-alive HTTP fixture: each accepted connection is served on its own thread and stays open
    // across POST (initiate), PUT (upload part), and DELETE (abort) requests so hyper's connection
    // pool reuses connections opened during initiation and multipart part uploads.
    let server_thread = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        drop(listener.set_nonblocking(true));
        let mut conn_threads = Vec::new();
        while !stop_server.load(Ordering::SeqCst)
            && deletes_server.load(Ordering::SeqCst) < 2
            && std::time::Instant::now() < deadline
        {
            match listener.accept() {
                Ok((mut conn, _)) => {
                    let posts_conn = Arc::clone(&posts_server);
                    let puts_conn = Arc::clone(&puts_server);
                    let deletes_conn = Arc::clone(&deletes_server);
                    let stop_conn = Arc::clone(&stop_server);
                    conn_threads.push(std::thread::spawn(move || {
                        drop(conn.set_read_timeout(Some(Duration::from_millis(50))));
                        let mut pending = Vec::new();
                        let mut chunk = [0u8; 64 * 1024];
                        while !stop_conn.load(Ordering::SeqCst)
                            && deletes_conn.load(Ordering::SeqCst) < 2
                            && std::time::Instant::now() < deadline
                        {
                            // Read until we have full HTTP headers (`\r\n\r\n`).
                            let header_end = loop {
                                if let Some(pos) =
                                    pending.windows(4).position(|window| window == b"\r\n\r\n")
                                {
                                    break Some(pos + 4);
                                }
                                if stop_conn.load(Ordering::SeqCst)
                                    || deletes_conn.load(Ordering::SeqCst) >= 2
                                    || std::time::Instant::now() >= deadline
                                {
                                    break None;
                                }
                                match conn.read(&mut chunk) {
                                    Ok(0) => break None,
                                    Ok(n) => pending.extend_from_slice(&chunk[..n]),
                                    Err(err)
                                        if err.kind() == std::io::ErrorKind::WouldBlock
                                            || err.kind() == std::io::ErrorKind::TimedOut => {}
                                    Err(_) => break None,
                                }
                            };
                            let Some(header_end) = header_end else {
                                break;
                            };

                            let headers = String::from_utf8_lossy(&pending[..header_end]).into_owned();
                            let first_line = headers.lines().next().unwrap_or("");
                            let content_len: usize = headers
                                .lines()
                                .find_map(|line| {
                                    let (name, value) = line.split_once(':')?;
                                    name.eq_ignore_ascii_case("content-length")
                                        .then(|| value.trim().parse::<usize>().ok())
                                        .flatten()
                                })
                                .unwrap_or(0);

                            // Drain the exact request body so keep-alive stays synchronized.
                            let mut remaining = content_len;
                            let already_buffered = pending.len() - header_end;
                            if already_buffered >= remaining {
                                pending.drain(..header_end + remaining);
                                remaining = 0;
                            } else {
                                pending.clear();
                                remaining -= already_buffered;
                            }
                            while remaining > 0 {
                                let to_read = remaining.min(chunk.len());
                                match conn.read(&mut chunk[..to_read]) {
                                    Ok(0) => break,
                                    Ok(n) => remaining -= n,
                                    Err(err)
                                        if err.kind() == std::io::ErrorKind::WouldBlock
                                            || err.kind() == std::io::ErrorKind::TimedOut =>
                                    {
                                        if std::time::Instant::now() >= deadline {
                                            break;
                                        }
                                    }
                                    Err(_) => break,
                                }
                            }
                            if remaining > 0 {
                                break;
                            }

                            if first_line.starts_with("POST ") {
                                let id = posts_conn.fetch_add(1, Ordering::SeqCst) + 1;
                                let body = format!(
                                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
                                     <InitiateMultipartUploadResult>\
                                     <Bucket>test-bucket</Bucket>\
                                     <Key>file-{id}.vortex</Key>\
                                     <UploadId>upload-{id}</UploadId>\
                                     </InitiateMultipartUploadResult>"
                                );
                                let resp = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/xml\r\nConnection: keep-alive\r\n\r\n{}",
                                    body.len(),
                                    body
                                );
                                if conn.write_all(resp.as_bytes()).is_err() {
                                    break;
                                }
                            } else if first_line.starts_with("PUT ") {
                                let part_id = puts_conn.fetch_add(1, Ordering::SeqCst) + 1;
                                let resp = format!(
                                    "HTTP/1.1 200 OK\r\nETag: \"etag-{part_id}\"\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n"
                                );
                                if conn.write_all(resp.as_bytes()).is_err() {
                                    break;
                                }
                            } else if first_line.starts_with("DELETE ") {
                                deletes_conn.fetch_add(1, Ordering::SeqCst);
                                let resp = "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
                                if conn.write_all(resp.as_bytes()).is_err() {
                                    break;
                                }
                            } else {
                                let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                                drop(conn.write_all(resp.as_bytes()));
                                break;
                            }
                        }
                    }));
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
        for handle in conn_threads {
            drop(handle.join());
        }
    });

    let build_s3 = || -> Result<Arc<dyn ObjectStore>, object_store::Error> {
        Ok(Arc::new(
            AmazonS3Builder::new()
                .with_endpoint(format!("http://{addr}"))
                .with_bucket_name("test-bucket")
                .with_region("us-east-1")
                .with_access_key_id("test")
                .with_secret_access_key("test")
                .with_virtual_hosted_style_request(false)
                .with_retry(RetryConfig {
                    max_retries: 0,
                    ..Default::default()
                })
                .with_client_options(
                    ClientOptions::new()
                        .with_allow_http(true)
                        .with_timeout(Duration::from_millis(1500)),
                )
                .build()?,
        ))
    };
    let s3_abort = build_s3()?;
    let s3_drop = build_s3()?;

    let tokio_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    // 32 MiB + 1 byte forces `flush()` to upload two 16 MiB parts concurrently per writer, opening
    // multiple persistent connections in each client's pool.
    let two_parts_plus_one = 32 * 1024 * 1024 + 1;
    let (mut writer_for_abort, writer_for_drop) = tokio_rt.block_on(async {
        let mut w1 = ObjectStoreWrite::new(s3_abort, &Path::from("s3-abort.vortex")).await?;
        w1.write_all(vec![1u8; two_parts_plus_one]).await?;
        w1.flush().await?;
        let mut w2 = ObjectStoreWrite::new(s3_drop, &Path::from("s3-drop.vortex")).await?;
        w2.write_all(vec![2u8; two_parts_plus_one]).await?;
        w2.flush().await?;
        Ok::<_, Box<dyn std::error::Error>>((w1, w2))
    })?;
    assert_eq!(post_count.load(Ordering::SeqCst), 2);
    assert_eq!(put_count.load(Ordering::SeqCst), 4);

    // Move both unwrapped S3 writers onto Vortex's CurrentThreadRuntime while the originating
    // Tokio current-thread runtime stays alive and idle. With retries disabled and keep-alive
    // connections held open, both DELETE aborts must still reach the server immediately.
    let current_rt = CurrentThreadRuntime::new();
    let deletes_client = Arc::clone(&delete_count);
    current_rt.block_on(async move {
        writer_for_abort.abort().await?;
        drop(writer_for_drop);
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while deletes_client.load(Ordering::SeqCst) < 2 {
            assert!(
                std::time::Instant::now() < deadline,
                "expected 2 S3 DELETE aborts, got {}",
                deletes_client.load(Ordering::SeqCst)
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;

    stop.store(true, Ordering::SeqCst);
    server_thread
        .join()
        .map_err(|_| "HTTP fixture thread panicked")?;
    assert_eq!(delete_count.load(Ordering::SeqCst), 2);
    drop(tokio_rt);
    Ok(())
}

#[test]
fn test_unwrapped_s3_completion_connection_reused_for_abort_outside_idle_tokio_runtime()
-> Result<(), Box<dyn std::error::Error>> {
    use std::io::Read;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use object_store::ClientOptions;
    use object_store::ObjectStore;
    use object_store::RetryConfig;
    use object_store::aws::AmazonS3Builder;
    use object_store::path::Path;
    use vortex_io::VortexWrite;
    use vortex_io::object_store::ObjectStoreWrite;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::current::CurrentThreadRuntime;

    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let init_count = Arc::new(AtomicUsize::new(0));
    let complete_count = Arc::new(AtomicUsize::new(0));
    let delete_count = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let inits_server = Arc::clone(&init_count);
    let completes_server = Arc::clone(&complete_count);
    let deletes_server = Arc::clone(&delete_count);
    let stop_server = Arc::clone(&stop);

    // Server closes the first connection after acknowledging writer 1's PUT part so writer 1's
    // `shutdown()`/`complete()` must open a second keep-alive connection. Writer 2 then reuses
    // that completion connection for its initiation and DELETE abort while the caller's Tokio
    // runtime stays idle.
    let server_thread = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        drop(listener.set_nonblocking(true));
        let mut conn_threads = Vec::new();
        while !stop_server.load(Ordering::SeqCst)
            && deletes_server.load(Ordering::SeqCst) < 1
            && std::time::Instant::now() < deadline
        {
            match listener.accept() {
                Ok((mut conn, _)) => {
                    let inits_conn = Arc::clone(&inits_server);
                    let completes_conn = Arc::clone(&completes_server);
                    let deletes_conn = Arc::clone(&deletes_server);
                    let stop_conn = Arc::clone(&stop_server);
                    conn_threads.push(std::thread::spawn(move || {
                        drop(conn.set_read_timeout(Some(Duration::from_millis(50))));
                        let mut pending = Vec::new();
                        let mut chunk = [0u8; 4096];
                        while !stop_conn.load(Ordering::SeqCst)
                            && deletes_conn.load(Ordering::SeqCst) < 1
                            && std::time::Instant::now() < deadline
                        {
                            let header_end = loop {
                                if let Some(pos) =
                                    pending.windows(4).position(|window| window == b"\r\n\r\n")
                                {
                                    break Some(pos + 4);
                                }
                                if stop_conn.load(Ordering::SeqCst)
                                    || deletes_conn.load(Ordering::SeqCst) >= 1
                                    || std::time::Instant::now() >= deadline
                                {
                                    break None;
                                }
                                match conn.read(&mut chunk) {
                                    Ok(0) => break None,
                                    Ok(n) => pending.extend_from_slice(&chunk[..n]),
                                    Err(err)
                                        if err.kind() == std::io::ErrorKind::WouldBlock
                                            || err.kind() == std::io::ErrorKind::TimedOut => {}
                                    Err(_) => break None,
                                }
                            };
                            let Some(header_end) = header_end else {
                                break;
                            };

                            let headers =
                                String::from_utf8_lossy(&pending[..header_end]).into_owned();
                            let first_line = headers.lines().next().unwrap_or("");
                            let content_len: usize = headers
                                .lines()
                                .find_map(|line| {
                                    let (name, value) = line.split_once(':')?;
                                    name.eq_ignore_ascii_case("content-length")
                                        .then(|| value.trim().parse::<usize>().ok())
                                        .flatten()
                                })
                                .unwrap_or(0);

                            let mut remaining = content_len;
                            let already_buffered = pending.len() - header_end;
                            if already_buffered >= remaining {
                                pending.drain(..header_end + remaining);
                                remaining = 0;
                            } else {
                                pending.clear();
                                remaining -= already_buffered;
                            }
                            while remaining > 0 {
                                let to_read = remaining.min(chunk.len());
                                match conn.read(&mut chunk[..to_read]) {
                                    Ok(0) => break,
                                    Ok(n) => remaining -= n,
                                    Err(err)
                                        if err.kind() == std::io::ErrorKind::WouldBlock
                                            || err.kind() == std::io::ErrorKind::TimedOut =>
                                    {
                                        if std::time::Instant::now() >= deadline {
                                            break;
                                        }
                                    }
                                    Err(_) => break,
                                }
                            }
                            if remaining > 0 {
                                break;
                            }

                            if first_line.starts_with("POST ") && first_line.contains("uploads") {
                                let id = inits_conn.fetch_add(1, Ordering::SeqCst) + 1;
                                let body = format!(
                                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
                                     <InitiateMultipartUploadResult>\
                                     <Bucket>test-bucket</Bucket>\
                                     <Key>file-{id}.vortex</Key>\
                                     <UploadId>upload-{id}</UploadId>\
                                     </InitiateMultipartUploadResult>"
                                );
                                let resp = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/xml\r\nConnection: keep-alive\r\n\r\n{}",
                                    body.len(),
                                    body
                                );
                                if conn.write_all(resp.as_bytes()).is_err() {
                                    break;
                                }
                            } else if first_line.starts_with("PUT ") {
                                // Close the connection after acknowledging writer 1's part so
                                // `complete()` must open a fresh persistent connection.
                                let resp = "HTTP/1.1 200 OK\r\nETag: \"etag-1\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                                drop(conn.write_all(resp.as_bytes()));
                                break;
                            } else if first_line.starts_with("POST ") {
                                completes_conn.fetch_add(1, Ordering::SeqCst);
                                let body = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
                                            <CompleteMultipartUploadResult>\
                                            <Location>http://test-bucket/file-1.vortex</Location>\
                                            <Bucket>test-bucket</Bucket>\
                                            <Key>file-1.vortex</Key>\
                                            <ETag>\"final-etag-1\"</ETag>\
                                            </CompleteMultipartUploadResult>";
                                let resp = format!(
                                    "HTTP/1.1 200 OK\r\nETag: \"final-etag-1\"\r\nContent-Length: {}\r\nContent-Type: application/xml\r\nConnection: keep-alive\r\n\r\n{}",
                                    body.len(),
                                    body
                                );
                                if conn.write_all(resp.as_bytes()).is_err() {
                                    break;
                                }
                            } else if first_line.starts_with("DELETE ") {
                                deletes_conn.fetch_add(1, Ordering::SeqCst);
                                let resp = "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
                                if conn.write_all(resp.as_bytes()).is_err() {
                                    break;
                                }
                            } else {
                                let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                                drop(conn.write_all(resp.as_bytes()));
                                break;
                            }
                        }
                    }));
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
        for handle in conn_threads {
            drop(handle.join());
        }
    });

    let s3: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::new()
            .with_endpoint(format!("http://{addr}"))
            .with_bucket_name("test-bucket")
            .with_region("us-east-1")
            .with_access_key_id("test")
            .with_secret_access_key("test")
            .with_virtual_hosted_style_request(false)
            .with_retry(RetryConfig {
                max_retries: 0,
                ..Default::default()
            })
            .with_client_options(
                ClientOptions::new()
                    .with_allow_http(true)
                    .with_timeout(Duration::from_millis(1500)),
            )
            .build()?,
    );

    let tokio_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut writer2 = tokio_rt.block_on(async {
        let mut writer1 =
            ObjectStoreWrite::new(Arc::clone(&s3), &Path::from("file-1.vortex")).await?;
        writer1.write_all(vec![1u8; 128]).await?;
        writer1.shutdown().await?;
        assert!(writer1.put_result().is_some());

        let mut writer2 =
            ObjectStoreWrite::new(Arc::clone(&s3), &Path::from("file-2.vortex")).await?;
        writer2.write_all(vec![2u8; 64]).await?;
        Ok::<_, Box<dyn std::error::Error>>(writer2)
    })?;
    assert_eq!(init_count.load(Ordering::SeqCst), 2);
    assert_eq!(complete_count.load(Ordering::SeqCst), 1);

    let current_rt = CurrentThreadRuntime::new();
    current_rt.block_on(async move {
        writer2.abort().await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;

    stop.store(true, Ordering::SeqCst);
    server_thread
        .join()
        .map_err(|_| "HTTP fixture thread panicked")?;
    assert_eq!(delete_count.load(Ordering::SeqCst), 1);
    drop(tokio_rt);
    Ok(())
}
