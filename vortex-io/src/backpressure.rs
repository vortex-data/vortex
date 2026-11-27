// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

/// The compression ratio to use before a write measures one.
///
/// A ratio of one assumes that the data does not compress. Thus, a size estimate made before the
/// first measurement does not underestimate the file.
const DEFAULT_COMPRESSION_RATIO: f64 = 1.0;

/// The byte accounting of one write.
///
/// Each byte that a write accepted is in one of three states:
///
/// 1. Pending: not compressed yet. This is the input queue, the data that a strategy coalesces
///    before compression, and the chunks in compression. See [`Self::pending_bytes`].
/// 2. Buffered: compressed, but not given to the output yet. See [`Self::buffered_bytes`].
/// 3. Written: given to the output. See [`Self::written_bytes`].
///
/// [`Self::estimated_size`] is the written bytes, plus the buffered bytes, plus the estimated
/// compressed size of the pending bytes. Pending bytes count in a tree of scopes, made with
/// [`Self::child`], because data compresses at a different ratio in each column and form. Each
/// scope has its own pending count and its own ratio. A scope without a sample uses the ratio of
/// its nearest ancestor with a sample, then the ratio over all the samples, then one.
///
/// Clones share the same counts, so a handle can be captured before the write begins and polled
/// while it runs.
#[derive(Clone, Debug)]
pub struct Backpressure {
    write: Arc<WriteState>,
    scope: Arc<Scope>,
}

/// The counts of one write.
#[derive(Debug)]
struct WriteState {
    written: AtomicU64,
    buffered: AtomicU64,
    scopes: Mutex<Vec<Arc<Scope>>>,
    overall: Ratio,
    /// The uncompressed bytes of the chunks that were written as constants, which take no space.
    constant: AtomicU64,
}

/// The pending bytes and the compression ratio of the data in one form.
#[derive(Debug, Default)]
struct Scope {
    pending: AtomicU64,
    ratio: Ratio,
    parent: Option<Arc<Scope>>,
}

/// The compressed bytes out and the uncompressed bytes in, over all the recorded samples.
#[derive(Debug, Default)]
struct Ratio {
    uncompressed: AtomicU64,
    compressed: AtomicU64,
}

impl Ratio {
    fn record(&self, uncompressed: u64, compressed: u64) {
        self.uncompressed.fetch_add(uncompressed, Ordering::Relaxed);
        self.compressed.fetch_add(compressed, Ordering::Relaxed);
    }

    fn measured(&self, extra_uncompressed: u64) -> Option<f64> {
        let uncompressed = self.uncompressed.load(Ordering::Relaxed);
        if uncompressed == 0 {
            return None;
        }

        let compressed = self.compressed.load(Ordering::Relaxed);
        Some(compressed as f64 / (uncompressed + extra_uncompressed) as f64)
    }
}

impl Scope {
    fn ratio(&self) -> Option<f64> {
        self.ratio
            .measured(0)
            .or_else(|| self.parent.as_ref().and_then(|parent| parent.ratio()))
    }
}

impl Default for Backpressure {
    fn default() -> Self {
        Self::new()
    }
}

impl Backpressure {
    /// Creates the accounting of a write.
    pub fn new() -> Self {
        let scope = Arc::new(Scope::default());

        Self {
            write: Arc::new(WriteState {
                written: AtomicU64::new(0),
                buffered: AtomicU64::new(0),
                scopes: Mutex::new(vec![Arc::clone(&scope)]),
                overall: Ratio::default(),
                constant: AtomicU64::new(0),
            }),
            scope,
        }
    }

    /// Creates a scope for data in a new form, which inherits the ratio of this scope until it
    /// has a sample of its own.
    pub fn child(&self) -> Self {
        let scope = Arc::new(Scope {
            parent: Some(Arc::clone(&self.scope)),
            ..Scope::default()
        });

        let mut scopes = self.write.scopes.lock();
        // Remove the scopes that no handle or reservation uses.
        scopes.retain(|scope| Arc::strong_count(scope) > 1);
        scopes.push(Arc::clone(&scope));
        drop(scopes);

        Self {
            write: Arc::clone(&self.write),
            scope,
        }
    }

    /// Returns the bytes that the write gave to the output.
    pub fn written_bytes(&self) -> u64 {
        self.write.written.load(Ordering::Relaxed)
    }

    /// Returns the compressed bytes that the write holds and did not give to the output yet.
    pub fn buffered_bytes(&self) -> u64 {
        self.write.buffered.load(Ordering::Relaxed)
    }

    /// Returns the bytes that the write accepted but did not compress yet, in all the scopes.
    pub fn pending_bytes(&self) -> u64 {
        self.write
            .scopes
            .lock()
            .iter()
            .map(|scope| scope.pending.load(Ordering::Relaxed))
            .sum()
    }

    /// Returns the compressed bytes divided by the uncompressed bytes over all the samples of the
    /// write, or one if there is no sample.
    ///
    /// The constant bytes are included only after the first compressed sample. Thus a write that
    /// starts with constant columns does not estimate its other data at zero bytes.
    pub fn compression_ratio(&self) -> f64 {
        self.write
            .overall
            .measured(self.write.constant.load(Ordering::Relaxed))
            .unwrap_or(DEFAULT_COMPRESSION_RATIO)
    }

    /// Returns the estimated compressed size of the pending bytes in all the scopes.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the ratios are not negative, and the cast saturates at u64::MAX"
    )]
    pub fn estimated_pending_bytes(&self) -> u64 {
        let fallback = self.compression_ratio();
        let estimate: f64 = self
            .write
            .scopes
            .lock()
            .iter()
            .map(|scope| {
                let pending = scope.pending.load(Ordering::Relaxed);
                if pending == 0 {
                    return 0.0;
                }
                pending as f64 * scope.ratio().unwrap_or(fallback)
            })
            .sum();
        estimate.round() as u64
    }

    /// Returns the estimated size of the output if the write finished now.
    ///
    /// The estimate does not include the footer of a file, which the write makes when it
    /// finishes. It can go down a small amount when a measured compression ratio changes.
    pub fn estimated_size(&self) -> u64 {
        self.written_bytes()
            .saturating_add(self.buffered_bytes())
            .saturating_add(self.estimated_pending_bytes())
    }

    /// Records that the write gave `bytes` to the output.
    pub fn record_written(&self, bytes: u64) {
        self.write.written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Records that `uncompressed` bytes of the data in this scope compressed to `compressed`
    /// bytes.
    pub fn record_ratio(&self, uncompressed: u64, compressed: u64) {
        self.scope.ratio.record(uncompressed, compressed);
        self.write.overall.record(uncompressed, compressed);
    }

    /// Records that `uncompressed` bytes of a chunk were written as a constant, which takes no
    /// space in the output.
    pub fn record_constant(&self, uncompressed: u64) {
        self.write
            .constant
            .fetch_add(uncompressed, Ordering::Relaxed);
    }

    /// Counts `bytes` as pending in this scope until the returned reservation is dropped.
    pub fn reserve_pending(&self, bytes: u64) -> BackpressureReservation {
        self.scope.pending.fetch_add(bytes, Ordering::Relaxed);
        self.reservation(State::Pending, bytes)
    }

    /// Counts `bytes` as buffered until the returned reservation is dropped.
    pub fn reserve_buffered(&self, bytes: u64) -> BackpressureReservation {
        self.write.buffered.fetch_add(bytes, Ordering::Relaxed);
        self.reservation(State::Buffered, bytes)
    }

    fn reservation(&self, state: State, bytes: u64) -> BackpressureReservation {
        BackpressureReservation {
            backpressure: self.clone(),
            state,
            bytes,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum State {
    Pending,
    Buffered,
}

/// A claim on the counts of a [`Backpressure`], released when dropped.
#[derive(Debug)]
pub struct BackpressureReservation {
    backpressure: Backpressure,
    state: State,
    bytes: u64,
}

impl BackpressureReservation {
    /// Returns the number of bytes held by this reservation.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for BackpressureReservation {
    fn drop(&mut self) {
        match self.state {
            State::Pending => {
                let pending = &self.backpressure.scope.pending;
                pending.fetch_sub(self.bytes, Ordering::Relaxed);
            }
            State::Buffered => {
                let buffered = &self.backpressure.write.buffered;
                buffered.fetch_sub(self.bytes, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_state_has_its_own_count() {
        let backpressure = Backpressure::new();
        let observer = backpressure.clone();

        let pending = backpressure.reserve_pending(16);
        let buffered = backpressure.reserve_buffered(32);
        backpressure.record_written(64);
        assert_eq!(pending.bytes(), 16);
        assert_eq!(observer.pending_bytes(), 16);
        assert_eq!(observer.buffered_bytes(), 32);
        assert_eq!(observer.written_bytes(), 64);
        // Without a sample, the pending bytes count in full.
        assert_eq!(observer.estimated_size(), 16 + 32 + 64);

        drop((pending, buffered));
        assert_eq!(observer.estimated_size(), 64);
    }

    #[test]
    fn scopes_use_their_own_ratio_or_the_nearest_sample() {
        let root = Backpressure::new();
        let values = root.child();
        let codes = values.child();
        let other = root.child();

        values.record_ratio(100, 10);
        let _values_bytes = values.reserve_pending(1000);
        let _codes_bytes = codes.reserve_pending(1000);
        let _other_bytes = other.reserve_pending(1000);

        // The codes inherit the ratio of the values, and the other scope uses the ratio over all
        // the samples.
        assert_eq!(root.estimated_pending_bytes(), 300);

        codes.record_ratio(100, 50);
        // The ratio over all the samples is 60 / 200.
        assert_eq!(root.estimated_pending_bytes(), 100 + 500 + 300);
        assert_eq!(root.compression_ratio(), 0.3);
        assert_eq!(root.pending_bytes(), 3000);
    }

    #[test]
    fn constant_bytes_count_only_after_a_compressed_sample() {
        let root = Backpressure::new();
        let _bytes = root.reserve_pending(1000);

        root.record_constant(300);
        assert_eq!(root.compression_ratio(), 1.0);
        assert_eq!(root.estimated_pending_bytes(), 1000);

        root.child().record_ratio(100, 40);
        assert_eq!(root.compression_ratio(), 0.1);
        assert_eq!(root.estimated_pending_bytes(), 100);
    }

    #[test]
    fn unused_scopes_are_removed() {
        let root = Backpressure::new();
        let child = root.child();
        let reservation = child.reserve_pending(10);
        drop(child);

        // The bytes of a dropped scope still count until the reservation is released.
        let _other = root.child();
        assert_eq!(root.pending_bytes(), 10);

        drop(reservation);
        let _another = root.child();
        assert_eq!(root.pending_bytes(), 0);
        assert_eq!(root.write.scopes.lock().len(), 3);
    }
}
