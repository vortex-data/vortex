// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(feature = "registry")]

use std::io::Read;
use std::io::Write;
use std::net::SocketAddr;
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use object_store::ClientOptions;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;
use vortex_io::VortexWrite;
use vortex_io::object_store::ObjectStoreWrite;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;

/// Keep-alive HTTP fixture implementing the S3 multipart upload endpoints (`POST ?uploads`,
/// `PUT ?partNumber=...&uploadId=...`, `POST ?uploadId=...`, `DELETE ?uploadId=...`).
///
/// Each accepted connection runs on its own thread and stays open across requests so hyper's
/// connection pool reuses persistent connections across multipart operations.
struct MockS3Server {
    addr: SocketAddr,
    init_count: Arc<AtomicUsize>,
    put_count: Arc<AtomicUsize>,
    complete_count: Arc<AtomicUsize>,
    delete_count: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    server_thread: Option<std::thread::JoinHandle<()>>,
}

impl MockS3Server {
    fn start(expected_deletes: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let init_count = Arc::new(AtomicUsize::new(0));
        let put_count = Arc::new(AtomicUsize::new(0));
        let complete_count = Arc::new(AtomicUsize::new(0));
        let delete_count = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let inits_server = Arc::clone(&init_count);
        let puts_server = Arc::clone(&put_count);
        let completes_server = Arc::clone(&complete_count);
        let deletes_server = Arc::clone(&delete_count);
        let stop_server = Arc::clone(&stop);

        let server_thread = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(15);
            drop(listener.set_nonblocking(true));
            let mut conn_threads = Vec::new();
            while !stop_server.load(Ordering::SeqCst)
                && (expected_deletes == 0
                    || deletes_server.load(Ordering::SeqCst) < expected_deletes)
                && Instant::now() < deadline
            {
                match listener.accept() {
                    Ok((mut conn, _)) => {
                        let inits_conn = Arc::clone(&inits_server);
                        let puts_conn = Arc::clone(&puts_server);
                        let completes_conn = Arc::clone(&completes_server);
                        let deletes_conn = Arc::clone(&deletes_server);
                        let stop_conn = Arc::clone(&stop_server);
                        conn_threads.push(std::thread::spawn(move || {
                            drop(conn.set_read_timeout(Some(Duration::from_millis(50))));
                            let mut pending = Vec::new();
                            let mut chunk = [0u8; 64 * 1024];
                            while !stop_conn.load(Ordering::SeqCst)
                                && (expected_deletes == 0
                                    || deletes_conn.load(Ordering::SeqCst) < expected_deletes)
                                && Instant::now() < deadline
                            {
                                let header_end = loop {
                                    if let Some(pos) =
                                        pending.windows(4).position(|window| window == b"\r\n\r\n")
                                    {
                                        break Some(pos + 4);
                                    }
                                    if stop_conn.load(Ordering::SeqCst)
                                        || (expected_deletes > 0
                                            && deletes_conn.load(Ordering::SeqCst)
                                                >= expected_deletes)
                                        || Instant::now() >= deadline
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
                                            if Instant::now() >= deadline {
                                                break;
                                            }
                                        }
                                        Err(_) => break,
                                    }
                                }
                                if remaining > 0 {
                                    break;
                                }

                                if first_line.starts_with("POST ") && first_line.contains("uploads")
                                {
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
                                    let part_id = puts_conn.fetch_add(1, Ordering::SeqCst) + 1;
                                    let resp = format!(
                                        "HTTP/1.1 200 OK\r\nETag: \"etag-{part_id}\"\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n"
                                    );
                                    if conn.write_all(resp.as_bytes()).is_err() {
                                        break;
                                    }
                                } else if first_line.starts_with("POST ") {
                                    let id = completes_conn.fetch_add(1, Ordering::SeqCst) + 1;
                                    let body = format!(
                                        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
                                         <CompleteMultipartUploadResult>\
                                         <Location>http://127.0.0.1/test-bucket/file-{id}.vortex</Location>\
                                         <Bucket>test-bucket</Bucket>\
                                         <Key>file-{id}.vortex</Key>\
                                         <ETag>\"final-etag-{id}\"</ETag>\
                                         </CompleteMultipartUploadResult>"
                                    );
                                    let resp = format!(
                                        "HTTP/1.1 200 OK\r\nETag: \"final-etag-{id}\"\r\nContent-Length: {}\r\nContent-Type: application/xml\r\nConnection: keep-alive\r\n\r\n{}",
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

        Ok(Self {
            addr,
            init_count,
            put_count,
            complete_count,
            delete_count,
            stop,
            server_thread: Some(server_thread),
        })
    }

    fn build_s3(&self) -> Result<Arc<dyn ObjectStore>, object_store::Error> {
        Ok(Arc::new(
            AmazonS3Builder::new()
                .with_endpoint(format!("http://{}", self.addr))
                .with_bucket_name("test-bucket")
                .with_region("us-east-1")
                .with_access_key_id("test")
                .with_secret_access_key("test")
                .with_virtual_hosted_style_request(false)
                .with_client_options(
                    ClientOptions::new()
                        .with_allow_http(true)
                        .with_timeout(Duration::from_millis(2000)),
                )
                .build()?,
        ))
    }

    fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.server_thread.take() {
            handle.join().map_err(|_| "HTTP fixture thread panicked")?;
        }
        Ok(())
    }
}

impl Drop for MockS3Server {
    fn drop(&mut self) {
        drop(self.shutdown());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_multipart_abort_and_drop_succeed_after_tokio_runtime_shutdown()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut server = MockS3Server::start(2)?;
        let s3_abort = server.build_s3()?;
        let s3_drop = server.build_s3()?;

        let tokio_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
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
        assert_eq!(server.init_count.load(Ordering::SeqCst), 2);
        assert_eq!(server.put_count.load(Ordering::SeqCst), 4);

        // Shut down the originating Tokio runtime before aborting/dropping on a non-Tokio runtime.
        drop(tokio_rt);

        let current_rt = CurrentThreadRuntime::new();
        let deletes_client = Arc::clone(&server.delete_count);
        current_rt.block_on(async move {
            writer_for_abort.abort().await?;
            drop(writer_for_drop);
            let deadline = Instant::now() + Duration::from_secs(5);
            while deletes_client.load(Ordering::SeqCst) < 2 {
                assert!(
                    Instant::now() < deadline,
                    "expected 2 S3 DELETE aborts, got {}",
                    deletes_client.load(Ordering::SeqCst)
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok::<_, Box<dyn std::error::Error>>(())
        })?;

        server.shutdown()?;
        assert_eq!(server.delete_count.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn test_s3_completion_and_subsequent_abort_after_tokio_runtime_shutdown()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut server = MockS3Server::start(1)?;
        let s3 = server.build_s3()?;

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
        assert_eq!(server.init_count.load(Ordering::SeqCst), 2);
        assert_eq!(server.complete_count.load(Ordering::SeqCst), 1);

        drop(tokio_rt);

        let current_rt = CurrentThreadRuntime::new();
        current_rt.block_on(async move {
            writer2.abort().await?;
            Ok::<_, Box<dyn std::error::Error>>(())
        })?;

        server.shutdown()?;
        assert_eq!(server.delete_count.load(Ordering::SeqCst), 1);
        Ok(())
    }
}
