// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fs::File;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::Duration;
use std::time::Instant;

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

/// Idempotently download many `(path, url)` pairs with adaptive parallelism.
///
/// This is the preferred way to fetch multi-shard datasets (ClickBench partitioned,
/// vector dataset train shards, Public BI tables, etc.) because it:
///
/// - skips all downloads immediately if `dir/.success` already exists,
/// - starts at `INITIAL_IN_FLIGHT` concurrent downloads and ramps up to
///   `MAX_IN_FLIGHT` as clean completions come in (TCP-style slow-start), then
///   halves on retries to back off from upstream rate limits,
/// - reuses the shared HTTP client across every shard,
/// - renders a top-of-block `N/total` bar plus a fixed number of reusable slot bars via
///   a shared [`MultiProgress`]: the terminal block size stays constant for the entire
///   run, so nothing "jumps" as shards cycle,
/// - short-circuits on the first error (the remaining in-flight downloads are dropped
///   when the returned future is dropped),
/// - writes `dir/.success` on completion so subsequent runs skip the whole batch.
#[tracing::instrument(skip_all, fields(count = tracing::field::Empty))]
pub async fn download_many<I>(dir: &Path, downloads: I) -> Result<Vec<PathBuf>>
where
    I: IntoIterator,
    I::Item: IntoDownload,
{
    if dir.join(".success").exists() {
        info!("skipping {}: already complete", dir.display());
        return Ok(Vec::new());
    }

    let downloads: Vec<(PathBuf, String)> = downloads
        .into_iter()
        .map(IntoDownload::into_download)
        .collect();
    tracing::Span::current().record("count", downloads.len());

    if downloads.is_empty() {
        return Ok(Vec::new());
    }

    let num_slots = downloads.len().min(MAX_IN_FLIGHT);
    let initial_in_flight = INITIAL_IN_FLIGHT.min(num_slots);
    let batch = BatchProgress::new(downloads.len() as u64, num_slots, initial_in_flight);

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

    let paths: Result<Vec<PathBuf>> = results.into_iter().collect();
    if paths.is_ok() {
        std::fs::write(dir.join(".success"), "").context("writing .success marker")?;
    }
    paths
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

/// Template for the top-of-block summary bar rendered by [`download_many`]. `{pos}/{len}`
/// tracks completed-of-total; `{msg}` is updated on every slot acquire / release to show
/// how many downloads are currently in flight.
const SHARDS_TEMPLATE: &str =
    "[{elapsed_precise}] shards  [{bar:30.green/white}] {pos}/{len}  {msg}";

/// How often slot spinners redraw. Fast enough to feel alive; slow enough that stderr
/// writes sneaking past `MultiProgress` do not constantly fight for cursor position.
const SLOT_TICK: Duration = Duration::from_millis(80);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Dynamic concurrency controller
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Number of in-flight downloads to start at. Matches TCP-style slow-start: start small so
/// we don't hammer the upstream on the very first connection, and double from there.
const INITIAL_IN_FLIGHT: usize = 4;

/// Upper bound on the number of concurrent downloads the controller can ramp up to, and
/// the number of slot rows pre-allocated in the [`MultiProgress`] block. Chosen large
/// enough that the retry-based controller is the effective ceiling, not this constant.
/// The trade-off is that on large batches the MP block will exceed a typical local
/// terminal height — indicatif handles this by drawing the most recent rows plus the
/// top shards bar — but on CI there is no TTY so the visual overflow does not apply.
const MAX_IN_FLIGHT: usize = 256;

/// Never let the controller drive concurrency below this floor on a flaky network.
/// A value of `1` means the fallback is serial downloads.
const MIN_IN_FLIGHT: usize = 1;

/// Minimum time between successive halves, in milliseconds. Coalesces simultaneous
/// retries from one upstream hiccup into a single reaction, preventing over-halving.
const HALVE_COOLDOWN_MS: u64 = 1000;

/// Decide the next in-flight limit after a clean (no-retry) download completes.
///
/// Returns `Some(new_limit)` if the limit should change, or `None` if it is already at
/// the cap or the computed move would be a no-op.
fn decide_on_success(current: usize, in_slow_start: bool) -> Option<usize> {
    if current >= MAX_IN_FLIGHT {
        return None;
    }
    let new = if in_slow_start {
        current.saturating_mul(2)
    } else {
        current.saturating_add(1)
    }
    .min(MAX_IN_FLIGHT);
    (new != current).then_some(new)
}

/// Decide the next in-flight limit after a failed download attempt.
///
/// Returns `Some(new_limit)` if the limit should be halved now. Returns `None` if the
/// halve is debounced (another halve fired within [`HALVE_COOLDOWN_MS`]) or we are
/// already at the [`MIN_IN_FLIGHT`] floor.
fn decide_on_retry(current: usize, now_ms: u64, last_halve_ms: u64) -> Option<usize> {
    if now_ms.saturating_sub(last_halve_ms) < HALVE_COOLDOWN_MS {
        return None;
    }
    if current <= MIN_IN_FLIGHT {
        return None;
    }
    let new = (current / 2).max(MIN_IN_FLIGHT);
    (new != current).then_some(new)
}

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
    /// Current concurrency limit — the source of truth read by the controller and
    /// written via [`BatchProgress::set_max_in_flight`].
    current_in_flight: AtomicUsize,
    num_slots: usize,
    /// Controller state: are we still in slow-start (double on success) or have we
    /// dropped into additive-increase (`+=1` on success) after the first retry?
    in_slow_start: AtomicBool,
    /// Millis since [`BatchInner::created_at`] of the most recent halve event, used to
    /// debounce bursts of retries from a single upstream hiccup.
    last_halve_at_ms: AtomicU64,
    created_at: Instant,
    // The MP is kept alive alongside the Arc so bars stay registered and rendered.
    // Once the last BatchProgress clone drops, the MP drops and clears the block.
    _mp: MultiProgress,
}

impl BatchInner {
    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.created_at.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// Refresh the shards bar message to reflect the current in-flight count. Called on
    /// every slot acquire / release and from `set_max_in_flight`.
    fn refresh_shards_message(&self) {
        let active = self.num_slots - self.free.lock().len();
        let limit = self.current_in_flight.load(AtomicOrdering::Relaxed);
        self.shards_bar
            .set_message(format!("({active} active, limit {limit})"));
    }
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
            in_slow_start: AtomicBool::new(true),
            last_halve_at_ms: AtomicU64::new(0),
            created_at: Instant::now(),
            _mp: mp,
        };
        inner.refresh_shards_message();
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
        self.inner.refresh_shards_message();
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

    /// Called when a download completed on its first attempt (no retries). Drives the
    /// slow-start / additive-increase side of AIMD.
    fn report_clean_success(&self) {
        let current = self.inner.current_in_flight.load(AtomicOrdering::Relaxed);
        let in_slow_start = self.inner.in_slow_start.load(AtomicOrdering::Relaxed);
        if let Some(new) = decide_on_success(current, in_slow_start) {
            self.set_max_in_flight(new);
        }
    }

    /// Called when a download attempt failed. Drives the halving side of AIMD, with an
    /// internal cooldown so a burst of simultaneous retries from one upstream hiccup
    /// halves the limit at most once.
    fn report_retry(&self) {
        let now_ms = self.inner.elapsed_ms();
        let current = self.inner.current_in_flight.load(AtomicOrdering::Relaxed);
        let last_halve_ms = self.inner.last_halve_at_ms.load(AtomicOrdering::Relaxed);
        if let Some(new) = decide_on_retry(current, now_ms, last_halve_ms) {
            self.inner
                .in_slow_start
                .store(false, AtomicOrdering::Relaxed);
            self.inner
                .last_halve_at_ms
                .store(now_ms, AtomicOrdering::Relaxed);
            self.set_max_in_flight(new);
        }
    }

    /// Adjust how many downloads may run concurrently. Clamped to the pre-allocated
    /// slot count. Raising the limit returns immediately; lowering spawns a background
    /// task that acquires and forgets the delta so the limit takes effect as active
    /// downloads complete naturally, never interrupting an in-flight transfer.
    fn set_max_in_flight(&self, target: usize) {
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
        self.inner.refresh_shards_message();
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
        self.owner.refresh_shards_message();
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
    let progress = DownloadProgress::new(batch, display_name).await;
    let mut last_err: Option<Error> = None;

    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            progress.reset_for_retry();
        }
        match single_attempt(client, url, tmp_path, &progress).await {
            Ok(()) => {
                if attempt == 0
                    && let Some(b) = batch
                {
                    b.report_clean_success();
                }
                progress.finalize();
                return Ok(());
            }
            Err(e) => {
                if let Some(b) = batch {
                    b.report_retry();
                }
                last_err = Some(e);
            }
        }
        if attempt + 1 < MAX_ATTEMPTS {
            sleep_with_jitter(attempt).await;
        }
    }

    progress.finalize();
    cleanup_partial_temp(tmp_path);
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("retry_get exhausted with no recorded error")))
}

/// Perform one download attempt end to end: create the temp file, issue the GET, stream
/// bytes to disk while advancing the progress bar. Returns on the first error so the
/// retry loop in [`retry_get`] can decide whether to try again.
async fn single_attempt(
    client: &Client,
    url: &str,
    tmp_path: &Path,
    progress: &DownloadProgress,
) -> Result<()> {
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

    progress.activate(response.content_length());

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .context("Failed to write to file")?;
        progress.inc(chunk.len() as u64);
    }

    AsyncWriteExt::flush(&mut file).await?;
    Ok(())
}

/// Sleep `2^attempt` seconds plus 0-500 ms of jitter before the next retry.
async fn sleep_with_jitter(attempt: u32) {
    let jitter = Duration::from_millis(rand::random::<u64>() % 500);
    let backoff = Duration::from_secs(1u64 << attempt) + jitter;
    warn!(
        "download attempt {} failed; retrying in {:?}",
        attempt + 1,
        backoff
    );
    tokio::time::sleep(backoff).await;
}

/// Best-effort removal of a partial temp file left behind when every retry attempt
/// failed. The UUID-named temp lives under `target/`; leaking it would be mostly
/// harmless but adds up over many CI runs.
fn cleanup_partial_temp(tmp_path: &Path) {
    if let Err(err) = std::fs::remove_file(tmp_path) {
        warn!(
            "failed to remove leftover temp download {}: {}",
            tmp_path.display(),
            err
        );
    }
}

/// Unified progress handle for a single download. Hides the split between pooled slot
/// bars (batched path) and one-off standalone bars (single-download path) so
/// [`retry_get`] does not have to branch on every update.
enum DownloadProgress {
    Slot(SlotGuard),
    Standalone(ProgressBar),
}

impl DownloadProgress {
    async fn new(batch: Option<&BatchProgress>, display_name: &str) -> Self {
        match batch {
            Some(b) => Self::Slot(b.acquire(display_name).await),
            None => Self::Standalone(new_standalone_bar(display_name)),
        }
    }

    fn reset_for_retry(&self) {
        match self {
            Self::Slot(s) => s.reset_for_retry(),
            Self::Standalone(bar) => {
                bar.set_style(
                    ProgressStyle::with_template(CONNECTING_TEMPLATE).expect("valid template"),
                );
                bar.set_length(0);
                bar.reset();
            }
        }
    }

    fn activate(&self, content_length: Option<u64>) {
        match (self, content_length) {
            (Self::Slot(s), Some(total)) => s.activate_known(total),
            (Self::Slot(s), None) => s.activate_unknown(),
            (Self::Standalone(bar), Some(total)) => {
                bar.set_style(
                    ProgressStyle::with_template(KNOWN_SIZE_TEMPLATE).expect("valid template"),
                );
                bar.set_length(total);
                bar.reset();
            }
            (Self::Standalone(bar), None) => {
                bar.set_style(
                    ProgressStyle::with_template(UNKNOWN_SIZE_TEMPLATE).expect("valid template"),
                );
                bar.set_length(0);
                bar.reset();
            }
        }
    }

    fn inc(&self, n: u64) {
        match self {
            Self::Slot(s) => s.inc(n),
            Self::Standalone(bar) => bar.inc(n),
        }
    }

    /// Tear down any visible state. Standalone bars are explicitly cleared here;
    /// slot bars clean themselves up when their [`SlotGuard`] drops.
    fn finalize(&self) {
        if let Self::Standalone(bar) = self {
            bar.finish_and_clear();
        }
    }
}

fn new_standalone_bar(display_name: &str) -> ProgressBar {
    let bar = ProgressBar::new(0);
    bar.set_style(ProgressStyle::with_template(CONNECTING_TEMPLATE).expect("valid template"));
    bar.set_prefix(display_name.to_owned());
    bar.enable_steady_tick(SLOT_TICK);
    bar
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    const COOLDOWN_MS: u64 = HALVE_COOLDOWN_MS;

    #[test]
    fn ramp_up_doubles_in_slow_start() {
        // Start at INITIAL (4). Each clean success in slow-start doubles until MAX.
        let mut cur = INITIAL_IN_FLIGHT;
        let expected = [8, 16, 32, 64, 128, 256];
        for want in expected {
            let next = decide_on_success(cur, true).expect("should ramp");
            assert_eq!(next, want);
            cur = next;
        }
        // At MAX, further successes are no-ops.
        assert_eq!(cur, MAX_IN_FLIGHT);
        assert_eq!(decide_on_success(cur, true), None);
        assert_eq!(decide_on_success(cur, false), None);
    }

    #[test]
    fn additive_increase_after_slow_start_exits() {
        // Once out of slow-start, successes add 1 instead of doubling.
        assert_eq!(decide_on_success(16, false), Some(17));
        assert_eq!(decide_on_success(17, false), Some(18));
    }

    #[test]
    fn retry_halves() {
        // At 64, a single retry (past the cooldown) halves to 32.
        assert_eq!(decide_on_retry(64, COOLDOWN_MS + 1, 0), Some(32));
        assert_eq!(decide_on_retry(32, COOLDOWN_MS + 1, 0), Some(16));
        assert_eq!(decide_on_retry(2, COOLDOWN_MS + 1, 0), Some(1));
    }

    #[test]
    fn halve_is_debounced() {
        // Three retries at t=100, t=200, t=500 (all within the 1 s cooldown after the
        // first halve at t=100) only produce one halve.
        let last_halve = 100;
        assert_eq!(decide_on_retry(64, 200, last_halve), None);
        assert_eq!(decide_on_retry(64, 500, last_halve), None);
        // A retry past the cooldown halves again.
        assert_eq!(
            decide_on_retry(64, last_halve + COOLDOWN_MS + 1, last_halve),
            Some(32)
        );
    }

    #[test]
    fn halve_respects_min_floor() {
        // At MIN (1), retries are no-ops — we never go below 1.
        assert_eq!(decide_on_retry(MIN_IN_FLIGHT, COOLDOWN_MS + 1, 0), None);
        // At 2, halving to 1 is the last step.
        assert_eq!(decide_on_retry(2, COOLDOWN_MS + 1, 0), Some(1));
    }

    #[test]
    fn ramp_up_respects_max_cap() {
        // Even from a large `current`, we never exceed MAX.
        assert_eq!(
            decide_on_success(MAX_IN_FLIGHT - 1, true),
            Some(MAX_IN_FLIGHT)
        );
        assert_eq!(decide_on_success(MAX_IN_FLIGHT, true), None);
        // Additive at the cap is also a no-op.
        assert_eq!(decide_on_success(MAX_IN_FLIGHT, false), None);
    }
}
