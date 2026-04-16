// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fs::File;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::Duration;

use anyhow::Context;
use anyhow::Error;
use anyhow::Result;
use bzip2::read::BzDecoder;
use futures::StreamExt;
use futures::stream;
use indicatif::MultiProgress;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use parking_lot::Mutex;
use reqwest::Client;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;
use tracing::info;
use tracing::warn;

use crate::utils::file::idempotent;
use crate::utils::file::idempotent_async;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Default concurrency limit for bulk downloads. Keeps us polite to the upstream while
/// still saturating a typical 10 Gb link on a parquet-per-shard benchmark.
pub const DEFAULT_DOWNLOAD_CONCURRENCY: usize = 16;

/// Anything that can be described as a `(target_path, url)` pair accepted by
/// [`download_many`].
pub trait IntoDownload {
    fn into_download(self) -> (PathBuf, String);
}

impl<P, S> IntoDownload for (P, S)
where
    P: Into<PathBuf>,
    S: Into<String>,
{
    fn into_download(self) -> (PathBuf, String) {
        (self.0.into(), self.1.into())
    }
}

/// Idempotently download a single URL to `fname`.
///
/// Uses the shared HTTP client, a 3-attempt exponential backoff retry loop with jitter,
/// and an [`indicatif::ProgressBar`]. If `fname` already exists, the download is
/// skipped.
#[tracing::instrument(skip_all, fields(url = %data_url.as_ref(), path = %fname.display()))]
pub async fn download_data(fname: PathBuf, data_url: impl AsRef<str>) -> Result<PathBuf> {
    download_one(fname, data_url.as_ref(), None).await
}

/// Idempotently download many `(path, url)` pairs with bounded parallelism.
///
/// This is the preferred way to fetch multi-shard datasets (ClickBench partitioned,
/// vector dataset train shards, Public BI tables, etc.) because it:
///
/// - caps in-flight HTTP requests at `max_concurrency` so we don't overwhelm the
///   upstream or our own network stack,
/// - reuses the shared HTTP client across every shard,
/// - renders a top-of-block `N/total` bar plus a fixed number of reusable slot bars via
///   a shared [`MultiProgress`]: the terminal block size stays constant for the entire
///   run, so nothing "jumps" as shards cycle,
/// - keeps the worker pool continuously full via `buffer_unordered`: as soon as any
///   shard finishes, the next queued shard reuses the freed slot,
/// - short-circuits on the first error (the remaining in-flight downloads are dropped
///   when the returned future is dropped),
/// - returns the resolved on-disk paths in completion order (not submission order).
///
/// Pass `0` as `max_concurrency` to use [`DEFAULT_DOWNLOAD_CONCURRENCY`].
#[tracing::instrument(skip_all, fields(count = tracing::field::Empty, max_concurrency))]
pub async fn download_many<I>(downloads: I, max_concurrency: usize) -> Result<Vec<PathBuf>>
where
    I: IntoIterator,
    I::Item: IntoDownload,
{
    let downloads: Vec<(PathBuf, String)> = downloads
        .into_iter()
        .map(IntoDownload::into_download)
        .collect();
    tracing::Span::current().record("count", downloads.len());

    if downloads.is_empty() {
        return Ok(Vec::new());
    }

    let concurrency = if max_concurrency == 0 {
        DEFAULT_DOWNLOAD_CONCURRENCY
    } else {
        max_concurrency
    };
    let num_slots = downloads.len().min(concurrency);

    let batch = BatchProgress::new(downloads.len() as u64, num_slots, num_slots);

    let results: Vec<Result<PathBuf>> = stream::iter(downloads)
        .map(|(path, url)| {
            let batch = batch.clone();
            async move {
                let result = download_one(path, &url, Some(&batch)).await;
                if result.is_ok() {
                    batch.advance();
                }
                result
            }
        })
        .buffer_unordered(num_slots)
        .collect()
        .await;

    batch.finish();

    results.into_iter().collect()
}

/// Idempotently decompress a bzip2 file into `output_path`, streaming the decompressed bytes
/// straight to disk so memory stays bounded.
///
/// This is used for the public BI dataset.
#[tracing::instrument(skip_all, fields(input = %input_path.display(), output = %output_path.display()))]
pub fn decompress_bz2(input_path: PathBuf, output_path: PathBuf) -> Result<PathBuf> {
    idempotent(&output_path, |path| {
        info!(
            "Decompressing bzip from {} to {}",
            input_path.display(),
            output_path.display()
        );
        let input_file = File::open(&input_path)
            .with_context(|| format!("Failed to open input file: {:?}", input_path))?;
        let mut decoder = BzDecoder::new(input_file);

        let mut output_file = File::create(path)
            .with_context(|| format!("Failed to create output file: {:?}", path))?;
        io::copy(&mut decoder, &mut output_file).context("Failed to decompress bzip2 stream")?;
        Ok(output_path.clone())
    })
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shared HTTP client
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Shared HTTP client used by every dataset download.
///
/// Reusing a single client gives us connection pooling, DNS caching, and consistent
/// timeouts across all callers. Each benchmark used to build its own
/// [`reqwest::Client`] on every download, which both wasted TLS handshakes and made it
/// hard to reason about total in-flight concurrency.
static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .read_timeout(Duration::from_secs(60))
        .timeout(Duration::from_secs(60 * 15))
        .build()
        .expect("failed to build shared benchmark HTTP client")
});

////////////////////////////////////////////////////////////////////////////////////////////////////
// Progress-bar templates
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for a slot that currently holds no download. `{msg}` with an empty message
/// renders as a blank line while still reserving the terminal row.
const IDLE_TEMPLATE: &str = "{msg}";

/// Template for a slot that has acquired a download but has not yet received response
/// headers. Shows the filename plus an animated spinner so the user can see we are
/// still making forward progress during TLS + request + first-byte latency.
const CONNECTING_TEMPLATE: &str = "{prefix:>28!} {spinner} connecting...";

/// Template for an active download when the response advertised a `Content-Length`.
const KNOWN_SIZE_TEMPLATE: &str =
    "{prefix:>28!} [{bar:30.cyan/blue}] {bytes:>9}/{total_bytes:>9} ({bytes_per_sec})";

/// Template for an active download when the response size is unknown.
const UNKNOWN_SIZE_TEMPLATE: &str = "{prefix:>28!} {spinner} {bytes} ({bytes_per_sec})";

/// Template for the top-of-block `N/total` summary bar rendered by [`download_many`].
const SHARDS_TEMPLATE: &str = "[{elapsed_precise}] shards  [{bar:30.green/white}] {pos}/{len}";

/// How often slot spinners redraw. Fast enough to feel alive; slow enough that stderr
/// writes sneaking past `MultiProgress` do not constantly fight for cursor position.
const SLOT_TICK: Duration = Duration::from_millis(80);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Batch download internals
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Shared rendering state for a batched download.
///
/// Layout is built once at construction: the shards bar is registered first (top row),
/// then `num_slots` slot bars are registered below it. None of these are ever added to
/// or removed from the [`MultiProgress`] again. The block size on the terminal is
/// exactly `num_slots + 1` rows for the entire run of [`download_many`].
///
/// Per-download lifecycle reuses a single slot bar by swapping its style between an
/// idle placeholder and the active progress-bar / spinner variants. A
/// [`tokio::sync::Semaphore`] gates how many slots can be in use at any instant, which
/// lets a future controller adjust concurrency at runtime via
/// [`BatchProgress::set_max_in_flight`] without ever touching the MP layout.
#[derive(Clone)]
struct BatchProgress {
    inner: Arc<BatchInner>,
}

struct BatchInner {
    shards_bar: ProgressBar,
    free: Mutex<Vec<ProgressBar>>,
    in_flight: Arc<Semaphore>,
    current_in_flight: AtomicUsize,
    num_slots: usize,
    // The MP is kept alive alongside the Arc so bars stay registered and rendered.
    // Once the last BatchProgress clone drops, the MP drops and clears the block.
    _mp: MultiProgress,
}

impl BatchProgress {
    fn new(total: u64, num_slots: usize, initial_in_flight: usize) -> Self {
        let initial_in_flight = initial_in_flight.min(num_slots);
        let mp = MultiProgress::new();

        let shards_bar = mp.add(ProgressBar::new(total));
        shards_bar
            .set_style(ProgressStyle::with_template(SHARDS_TEMPLATE).expect("valid template"));

        let idle_style = ProgressStyle::with_template(IDLE_TEMPLATE).expect("valid template");
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            let bar = mp.add(ProgressBar::new(0));
            bar.set_style(idle_style.clone());
            bar.set_message("");
            bar.enable_steady_tick(SLOT_TICK);
            slots.push(bar);
        }

        let inner = BatchInner {
            shards_bar,
            free: Mutex::new(slots),
            in_flight: Arc::new(Semaphore::new(initial_in_flight)),
            current_in_flight: AtomicUsize::new(initial_in_flight),
            num_slots,
            _mp: mp,
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Wait for an in-flight permit, then claim an idle slot and switch it to the
    /// [`CONNECTING_TEMPLATE`] style. The returned guard releases both the permit and
    /// the slot on drop.
    async fn acquire(&self, prefix: &str) -> SlotGuard {
        let permit = Arc::clone(&self.inner.in_flight)
            .acquire_owned()
            .await
            .expect("batch semaphore is never closed while a download is in flight");
        let bar = self
            .inner
            .free
            .lock()
            .pop()
            .expect("slot free list invariant broken: permits outnumber pre-allocated slots");
        bar.set_style(ProgressStyle::with_template(CONNECTING_TEMPLATE).expect("valid template"));
        bar.set_prefix(prefix.to_owned());
        bar.set_message("");
        bar.set_length(0);
        bar.reset();
        SlotGuard {
            bar,
            owner: Arc::clone(&self.inner),
            _permit: permit,
        }
    }

    fn advance(&self) {
        self.inner.shards_bar.inc(1);
    }

    fn finish(&self) {
        self.inner.shards_bar.finish_and_clear();
    }

    /// Adjust how many downloads may run concurrently. Clamped to the pre-allocated
    /// slot count. Raising the limit returns immediately; lowering spawns a background
    /// task that acquires and forgets the delta so the limit takes effect as active
    /// downloads complete naturally, never interrupting an in-flight transfer.
    ///
    /// The mechanism is in place but no policy currently calls it. A future adaptive
    /// controller (error-rate backoff, throughput watchdog, explicit CLI flag) can drop
    /// in without any further changes to this module.
    #[allow(dead_code)]
    pub(crate) fn set_max_in_flight(&self, target: usize) {
        let target = target.min(self.inner.num_slots);
        let prev = self
            .inner
            .current_in_flight
            .swap(target, AtomicOrdering::Relaxed);
        match target.cmp(&prev) {
            Ordering::Greater => {
                self.inner.in_flight.add_permits(target - prev);
            }
            Ordering::Less => {
                let delta = u32::try_from(prev - target).expect("delta fits in u32");
                let sem = Arc::clone(&self.inner.in_flight);
                tokio::spawn(async move {
                    if let Ok(permit) = sem.acquire_many_owned(delta).await {
                        permit.forget();
                    }
                });
            }
            Ordering::Equal => {}
        }
    }
}

/// RAII handle for a borrowed slot bar. Drives the bar through its active lifecycle and
/// resets it back to the idle placeholder on drop.
struct SlotGuard {
    bar: ProgressBar,
    owner: Arc<BatchInner>,
    _permit: OwnedSemaphorePermit,
}

impl SlotGuard {
    fn activate_known(&self, total: u64) {
        self.bar
            .set_style(ProgressStyle::with_template(KNOWN_SIZE_TEMPLATE).expect("valid template"));
        self.bar.set_length(total);
        self.bar.reset();
    }

    fn activate_unknown(&self) {
        self.bar.set_style(
            ProgressStyle::with_template(UNKNOWN_SIZE_TEMPLATE).expect("valid template"),
        );
        self.bar.set_length(0);
        self.bar.reset();
    }

    fn inc(&self, n: u64) {
        self.bar.inc(n);
    }

    fn reset_for_retry(&self) {
        self.bar
            .set_style(ProgressStyle::with_template(CONNECTING_TEMPLATE).expect("valid template"));
        self.bar.set_length(0);
        self.bar.reset();
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        let bar = self.bar.clone();
        bar.set_style(ProgressStyle::with_template(IDLE_TEMPLATE).expect("valid template"));
        bar.set_prefix("");
        bar.set_message("");
        bar.set_length(0);
        bar.reset();
        self.owner.free.lock().push(bar);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Core download implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Core download implementation shared by [`download_data`] and [`download_many`].
///
/// When `batch` is `Some`, the download reuses one of the pre-allocated slot bars from
/// the batch's [`MultiProgress`]. When `batch` is `None` the download renders its own
/// standalone bar.
async fn download_one(fname: PathBuf, url: &str, batch: Option<&BatchProgress>) -> Result<PathBuf> {
    let display_name = fname
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<download>")
        .to_owned();
    idempotent_async(&fname, async move |tmp_path| {
        retry_get(&HTTP_CLIENT, url, &tmp_path, &display_name, batch).await
    })
    .await
}

/// Perform an HTTP GET into `tmp_path`, retrying up to three times with exponential
/// backoff and a small jitter to avoid lockstep retries across concurrent shards. A
/// partial temp file from an exhausted retry loop is removed before returning the final
/// error.
///
/// When `batch` is `Some`, a pre-allocated slot bar is reused across retries. When
/// `batch` is `None`, a standalone [`ProgressBar`] is created and cleared at the end.
async fn retry_get(
    client: &Client,
    url: &str,
    tmp_path: &Path,
    display_name: &str,
    batch: Option<&BatchProgress>,
) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 3;
    let mut last_err: Option<Error> = None;

    let slot = match batch {
        Some(b) => Some(b.acquire(display_name).await),
        None => None,
    };
    let standalone = slot.is_none().then(|| new_standalone_bar(display_name));

    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            reset_progress_for_retry(slot.as_ref(), standalone.as_ref());
        }
        let outcome: Result<()> = async {
            let mut file = TokioFile::create(tmp_path)
                .await
                .context("Failed to create file")?;
            let response = client
                .get(url)
                .send()
                .await
                .context("Failed to send HTTP request")?
                .error_for_status()
                .context("HTTP request returned error status")?;

            activate_progress(
                slot.as_ref(),
                standalone.as_ref(),
                response.content_length(),
            );

            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                AsyncWriteExt::write_all(&mut file, &chunk)
                    .await
                    .context("Failed to write to file")?;
                advance_progress(slot.as_ref(), standalone.as_ref(), chunk.len() as u64);
            }

            AsyncWriteExt::flush(&mut file).await?;
            Ok(())
        }
        .await;

        match outcome {
            Ok(()) => {
                if let Some(bar) = standalone.as_ref() {
                    bar.finish_and_clear();
                }
                // `slot` drops here, resetting its bar to idle.
                return Ok(());
            }
            Err(e) => last_err = Some(e),
        }

        if attempt + 1 < MAX_ATTEMPTS {
            let jitter = Duration::from_millis(rand::random::<u64>() % 500);
            let backoff = Duration::from_secs(1u64 << attempt) + jitter;
            warn!(
                "download attempt {} failed; retrying in {:?}",
                attempt + 1,
                backoff
            );
            tokio::time::sleep(backoff).await;
        }
    }

    if let Some(bar) = standalone.as_ref() {
        bar.finish_and_clear();
    }

    if let Err(err) = std::fs::remove_file(tmp_path) {
        warn!(
            "failed to remove leftover temp download {}: {}",
            tmp_path.display(),
            err
        );
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("retry_get exhausted with no recorded error")))
}

fn new_standalone_bar(display_name: &str) -> ProgressBar {
    let bar = ProgressBar::new(0);
    bar.set_style(ProgressStyle::with_template(CONNECTING_TEMPLATE).expect("valid template"));
    bar.set_prefix(display_name.to_owned());
    bar.enable_steady_tick(SLOT_TICK);
    bar
}

fn reset_progress_for_retry(slot: Option<&SlotGuard>, standalone: Option<&ProgressBar>) {
    if let Some(slot) = slot {
        slot.reset_for_retry();
    } else if let Some(bar) = standalone {
        bar.set_style(ProgressStyle::with_template(CONNECTING_TEMPLATE).expect("valid template"));
        bar.set_length(0);
        bar.reset();
    }
}

fn activate_progress(
    slot: Option<&SlotGuard>,
    standalone: Option<&ProgressBar>,
    content_length: Option<u64>,
) {
    match (slot, standalone) {
        (Some(slot), _) => match content_length {
            Some(total) => slot.activate_known(total),
            None => slot.activate_unknown(),
        },
        (None, Some(bar)) => match content_length {
            Some(total) => {
                bar.set_style(
                    ProgressStyle::with_template(KNOWN_SIZE_TEMPLATE).expect("valid template"),
                );
                bar.set_length(total);
                bar.reset();
            }
            None => {
                bar.set_style(
                    ProgressStyle::with_template(UNKNOWN_SIZE_TEMPLATE).expect("valid template"),
                );
                bar.set_length(0);
                bar.reset();
            }
        },
        (None, None) => {}
    }
}

fn advance_progress(slot: Option<&SlotGuard>, standalone: Option<&ProgressBar>, n: u64) {
    if let Some(slot) = slot {
        slot.inc(n);
    } else if let Some(bar) = standalone {
        bar.inc(n);
    }
}
