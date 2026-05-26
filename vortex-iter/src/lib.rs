//! Batch-first iterators that autovectorize across data layouts and operations.
//!
//! The unit of work is a fixed-width lane block `[T; N]` rather than a single
//! element. A source ([`batches`]) reinterprets a slice as `[T; N]` blocks with
//! [`as_chunks`](slice::as_chunks) — zero copy — and yields each block *by value*.
//! Because `[T; N]` is [`Copy`] and `N` is a compile-time constant, reading a
//! block lowers to a single vector load into a SIMD register (no `memcpy`), and
//! the combinators keep that block shape so the inner loop over `N` lanes stays
//! a tight, countable, branch-free loop that LLVM turns into SIMD.
//!
//! # Tail contract
//!
//! Iteration has two phases: call [`next_batch`](BatchIter::next_batch) until it
//! returns `None` to drain the full `[T; N]` blocks, then call
//! [`next_tail`](BatchIter::next_tail) until it returns `None` to drain the
//! `< N` trailing elements. The provided terminals follow this order, so the
//! scalar tail is handled automatically.

/// An iterator that yields fixed-width lane blocks of size `N` by value.
///
/// See the [crate] docs for the two-phase tail contract between
/// [`next_batch`](Self::next_batch) and [`next_tail`](Self::next_tail).
pub trait BatchIter<const N: usize>: Sized {
    /// The per-lane element type.
    type Lane: Copy;

    /// Returns the next full block of `N` lanes, or `None` once only the
    /// trailing `< N` elements remain.
    fn next_batch(&mut self) -> Option<[Self::Lane; N]>;

    /// Returns the next trailing element. Only valid once
    /// [`next_batch`](Self::next_batch) has returned `None`.
    fn next_tail(&mut self) -> Option<Self::Lane>;

    /// Transforms each lane with `f`, elementwise. Autovectorizes via
    /// [`core::array::from_fn`]; the same `f` is applied to the scalar tail.
    fn map<U, F>(self, f: F) -> Map<Self, F>
    where
        U: Copy,
        F: FnMut(Self::Lane) -> U,
    {
        Map { inner: self, f }
    }

    /// Transforms a whole block with `f`. Hand-tuned escape hatch for block
    /// kernels. The partial tail is padded to `N` with [`Default::default`],
    /// passed through `f`, and the valid prefix kept — correct for *elementwise*
    /// block kernels only. Cross-lane kernels must use the raw
    /// [`next_batch`](Self::next_batch)/[`next_tail`](Self::next_tail) API.
    fn map_batch<U, F>(self, f: F) -> MapBatch<Self, F, U, N>
    where
        U: Copy,
        Self::Lane: Default,
        F: FnMut([Self::Lane; N]) -> [U; N],
    {
        MapBatch {
            inner: self,
            f,
            tail: MapBatchTail::Pending,
        }
    }

    /// Lane-parallel reduction: keeps `N` running accumulators, combines blocks
    /// lane-wise, horizontal-reduces once at the end, then folds the tail.
    ///
    /// `identity` must be the neutral element of `f`. For floating-point sums
    /// this changes summation order versus a sequential fold.
    fn reduce_lanes<F>(mut self, identity: Self::Lane, mut f: F) -> Self::Lane
    where
        F: FnMut(Self::Lane, Self::Lane) -> Self::Lane,
    {
        let mut acc = [identity; N];
        while let Some(b) = self.next_batch() {
            for i in 0..N {
                acc[i] = f(acc[i], b[i]);
            }
        }
        let mut s = identity;
        for lane in acc {
            s = f(s, lane);
        }
        while let Some(x) = self.next_tail() {
            s = f(s, x);
        }
        s
    }

    /// Writes every lane into `out`, which must have the same length as the
    /// logical sequence.
    fn write_to(mut self, out: &mut [Self::Lane]) {
        let (chunks, tail) = out.as_chunks_mut::<N>();
        let mut i = 0;
        while let Some(b) = self.next_batch() {
            chunks[i] = b;
            i += 1;
        }
        let mut t = 0;
        while let Some(x) = self.next_tail() {
            tail[t] = x;
            t += 1;
        }
    }

    /// Applies `f` to every lane, blocks first then tail.
    fn for_each<F>(mut self, mut f: F)
    where
        F: FnMut(Self::Lane),
    {
        while let Some(b) = self.next_batch() {
            for x in b {
                f(x);
            }
        }
        while let Some(x) = self.next_tail() {
            f(x);
        }
    }
}

/// Zero-copy view of `slice` as `[T; N]` blocks plus an embedded scalar tail.
pub struct Batches<'a, T, const N: usize> {
    chunks: &'a [[T; N]],
    tail: &'a [T],
    cpos: usize,
    tpos: usize,
}

/// Creates a [`Batches`] iterator over `slice`, splitting it into `[T; N]`
/// blocks and a trailing remainder via [`as_chunks`](slice::as_chunks).
pub fn batches<T: Copy, const N: usize>(slice: &[T]) -> Batches<'_, T, N> {
    let (chunks, tail) = slice.as_chunks::<N>();
    Batches {
        chunks,
        tail,
        cpos: 0,
        tpos: 0,
    }
}

impl<'a, T: Copy, const N: usize> BatchIter<N> for Batches<'a, T, N> {
    type Lane = T;

    #[inline]
    fn next_batch(&mut self) -> Option<[T; N]> {
        let b = *self.chunks.get(self.cpos)?;
        self.cpos += 1;
        Some(b)
    }

    #[inline]
    fn next_tail(&mut self) -> Option<T> {
        let x = *self.tail.get(self.tpos)?;
        self.tpos += 1;
        Some(x)
    }
}

/// Adapter returned by [`BatchIter::map`].
pub struct Map<I, F> {
    inner: I,
    f: F,
}

impl<I, F, U, const N: usize> BatchIter<N> for Map<I, F>
where
    I: BatchIter<N>,
    U: Copy,
    F: FnMut(I::Lane) -> U,
{
    type Lane = U;

    #[inline]
    fn next_batch(&mut self) -> Option<[U; N]> {
        let b = self.inner.next_batch()?;
        Some(core::array::from_fn(|i| (self.f)(b[i])))
    }

    #[inline]
    fn next_tail(&mut self) -> Option<U> {
        self.inner.next_tail().map(|x| (self.f)(x))
    }
}

enum MapBatchTail<U, const N: usize> {
    Pending,
    Ready([U; N], usize, usize),
}

/// Adapter returned by [`BatchIter::map_batch`].
pub struct MapBatch<I, F, U, const N: usize> {
    inner: I,
    f: F,
    tail: MapBatchTail<U, N>,
}

impl<I, F, U, const N: usize> BatchIter<N> for MapBatch<I, F, U, N>
where
    I: BatchIter<N>,
    I::Lane: Default,
    U: Copy,
    F: FnMut([I::Lane; N]) -> [U; N],
{
    type Lane = U;

    #[inline]
    fn next_batch(&mut self) -> Option<[U; N]> {
        let b = self.inner.next_batch()?;
        Some((self.f)(b))
    }

    fn next_tail(&mut self) -> Option<U> {
        if matches!(self.tail, MapBatchTail::Pending) {
            let mut src = [I::Lane::default(); N];
            let mut t = 0;
            while let Some(x) = self.inner.next_tail() {
                src[t] = x;
                t += 1;
            }
            self.tail = MapBatchTail::Ready((self.f)(src), t, 0);
        }
        match &mut self.tail {
            MapBatchTail::Ready(buf, len, pos) if *pos < *len => {
                let v = buf[*pos];
                *pos += 1;
                Some(v)
            }
            _ => None,
        }
    }
}

/// Planar zip of two slices into aligned `([A; N], [B; N])` block pairs.
///
/// Block pairs are kept *planar* (the `A` lanes in one block, the `B` lanes in
/// another) rather than interleaved, so binary kernels load each side straight
/// into its own vector register. A single shared trip count drives both sides,
/// which is what lets the binary loop widen to the full vector width.
///
/// When the two slices differ in length, full blocks and tails are each
/// truncated to the shorter side.
pub struct ZipBatches<'a, A, B, const N: usize> {
    a_chunks: &'a [[A; N]],
    b_chunks: &'a [[B; N]],
    a_tail: &'a [A],
    b_tail: &'a [B],
    n: usize,
    tn: usize,
    pos: usize,
}

/// Creates a [`ZipBatches`] over `lhs` and `rhs`.
pub fn zip<'a, A: Copy, B: Copy, const N: usize>(
    lhs: &'a [A],
    rhs: &'a [B],
) -> ZipBatches<'a, A, B, N> {
    let (a_chunks, a_tail) = lhs.as_chunks::<N>();
    let (b_chunks, b_tail) = rhs.as_chunks::<N>();
    let full_blocks = a_chunks.len().min(b_chunks.len());
    let tail_lanes = a_tail.len().min(b_tail.len());
    ZipBatches {
        a_chunks,
        b_chunks,
        a_tail,
        b_tail,
        n: full_blocks,
        tn: tail_lanes,
        pos: 0,
    }
}

impl<'a, A: Copy, B: Copy, const N: usize> ZipBatches<'a, A, B, N> {
    /// Returns the next planar block pair, or `None` when full blocks are
    /// exhausted.
    #[inline]
    pub fn next_pair(&mut self) -> Option<([A; N], [B; N])> {
        if self.pos >= self.n {
            return None;
        }
        let pair = (self.a_chunks[self.pos], self.b_chunks[self.pos]);
        self.pos += 1;
        Some(pair)
    }

    /// Lane-parallel binary reduction (e.g. dot product). `lane_op` folds one
    /// `A` and one `B` lane into the accumulator; pass [`f32::mul_add`]-style
    /// fused ops to emit FMA. `combine` performs the final horizontal reduce.
    pub fn fold_lanes<Acc, F, G>(mut self, init: Acc, mut lane_op: F, mut combine: G) -> Acc
    where
        Acc: Copy,
        F: FnMut(Acc, A, B) -> Acc,
        G: FnMut(Acc, Acc) -> Acc,
    {
        let mut acc = [init; N];
        while let Some((xs, ys)) = self.next_pair() {
            for i in 0..N {
                acc[i] = lane_op(acc[i], xs[i], ys[i]);
            }
        }
        let mut total = init;
        for lane in acc {
            total = combine(total, lane);
        }
        for k in 0..self.tn {
            total = lane_op(total, self.a_tail[k], self.b_tail[k]);
        }
        total
    }

    /// Elementwise binary map of the two sides into `out` (length must match the
    /// shorter side).
    pub fn map_to<U, F>(mut self, out: &mut [U], mut f: F)
    where
        U: Copy,
        F: FnMut(A, B) -> U,
    {
        let (out_chunks, out_tail) = out.as_chunks_mut::<N>();
        let mut i = 0;
        while let Some((xs, ys)) = self.next_pair() {
            out_chunks[i] = core::array::from_fn(|j| f(xs[j], ys[j]));
            i += 1;
        }
        for k in 0..self.tn {
            out_tail[k] = f(self.a_tail[k], self.b_tail[k]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_with_tail() {
        let data: Vec<u32> = (0..100).collect();
        let got = batches::<u32, 8>(&data).reduce_lanes(0, |a, b| a.wrapping_add(b));
        assert_eq!(got, data.iter().sum::<u32>());
    }

    #[test]
    fn map_write_with_tail() {
        let data: Vec<u32> = (0..100).collect();
        let mut out = vec![0u32; data.len()];
        batches::<u32, 8>(&data)
            .map(|x| x.wrapping_mul(3))
            .write_to(&mut out);
        let expected: Vec<u32> = data.iter().map(|x| x.wrapping_mul(3)).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn map_batch_pads_tail() {
        let data: Vec<u32> = (0..100).collect();
        let mut out = vec![0u32; data.len()];
        batches::<u32, 8>(&data)
            .map_batch(|b: [u32; 8]| core::array::from_fn(|i| b[i].wrapping_add(1)))
            .write_to(&mut out);
        let expected: Vec<u32> = data.iter().map(|x| x.wrapping_add(1)).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn dot_with_tail() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..100).map(|x| (x as f32) * 0.5).collect();
        let got = zip::<f32, f32, 8>(&a, &b).fold_lanes(
            0.0f32,
            |acc, x, y| x.mul_add(y, acc),
            |x, y| x + y,
        );
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!(
            (got - expected).abs() < 1e-1,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn zip_map_to_with_tail() {
        let a: Vec<i32> = (0..100).collect();
        let b: Vec<i32> = (0..100).map(|x| x * 2).collect();
        let mut out = vec![0i32; 100];
        zip::<i32, i32, 8>(&a, &b).map_to(&mut out, |x, y| x.wrapping_add(y));
        let expected: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(out, expected);
    }
}
