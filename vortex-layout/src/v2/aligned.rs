// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`AlignedArrayStream`] — a k-way zip of [`ArrayStream`]s that
//! emits row-aligned slices.
//!
//! Each step polls all child streams, finds the smallest currently-
//! available row count, slices every child to that length, and emits
//! a `Vec<ArrayRef>` of equal-length arrays. Larger remaining slices
//! are kept buffered for the next step.
//!
//! The motivating use case: `StructPlan.execute(range)` calls
//! `child.execute(range)` on each field. Different fields can return
//! arrays at different chunk granularities (the writer's byte-based
//! coalescing produces one big chunk for narrow numeric columns and
//! several smaller chunks for wide string columns). A naive lockstep
//! zip would mismatch lengths chunk-for-chunk; this stream realigns.
//!
//! Optionally, callers can ask for a minimum batch size — the stream
//! will accumulate (concat) per-child arrays until each child has at
//! least `min_rows` queued, or hits EOF, before emitting. This trades
//! a copy for fewer, larger emitted batches.

use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::StreamExt as _;
use vortex_array::ArrayRef;
use vortex_array::IntoArray as _;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// K-way row-aligned zip of [`SendableArrayStream`]s. See module docs.
pub struct AlignedArrayStream {
    children: Vec<ChildState>,
    /// Optional minimum batch size hint — see [`AlignedArrayStream::with_min_rows`].
    min_rows: Option<usize>,
}

struct ChildState {
    /// Upstream source. Set to `None` when EOF is reached.
    stream: Option<SendableArrayStream>,
    /// Output dtype of the child stream — needed when concatenating
    /// multiple buffered batches.
    dtype: DType,
    /// Currently-buffered head array. `None` while we haven't pulled
    /// anything yet (or after the last slice exhausted it).
    head: Option<ArrayRef>,
}

impl AlignedArrayStream {
    /// Construct a new aligned k-way zip. Each child's dtype is
    /// captured up-front (used when concatenating multiple buffered
    /// batches per the `min_rows` hint). The number of outputs per
    /// step equals `children.len()`.
    pub fn new(children: Vec<SendableArrayStream>) -> Self {
        let children = children
            .into_iter()
            .map(|s| {
                let dtype = s.dtype().clone();
                ChildState {
                    stream: Some(s),
                    dtype,
                    head: None,
                }
            })
            .collect();
        Self {
            children,
            min_rows: None,
        }
    }

    /// Try to emit at least `min_rows` rows per step (concatenating
    /// per-child as needed). The stream still emits short final
    /// batches when one or more children hit EOF.
    pub fn with_min_rows(mut self, min_rows: usize) -> Self {
        self.min_rows = Some(min_rows.max(1));
        self
    }

    /// Number of children being zipped.
    pub fn arity(&self) -> usize {
        self.children.len()
    }
}

impl Stream for AlignedArrayStream {
    type Item = VortexResult<Vec<ArrayRef>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let target = this.min_rows.unwrap_or(1);

        // Step 1: refill each child's head until it has at least
        // `target` rows OR upstream is EOF. Polling is sequential per
        // child; if any poll is Pending we return Pending.
        for child in this.children.iter_mut() {
            loop {
                let buffered = child.head.as_ref().map_or(0, |a| a.len());
                if buffered >= target {
                    break;
                }
                let Some(stream) = child.stream.as_mut() else {
                    break; // EOF; can't add more.
                };
                match stream.poll_next_unpin(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(None) => {
                        child.stream = None;
                        break;
                    }
                    Poll::Ready(Some(Err(err))) => {
                        return Poll::Ready(Some(Err(err)));
                    }
                    Poll::Ready(Some(Ok(arr))) => {
                        if arr.is_empty() {
                            // Skip empty arrays; they'd corrupt the
                            // bookkeeping below without contributing.
                            continue;
                        }
                        child.head = Some(match child.head.take() {
                            None => arr,
                            Some(prev) => {
                                let chunked = match ChunkedArray::try_new(
                                    vec![prev, arr],
                                    child.dtype.clone(),
                                ) {
                                    Ok(c) => c,
                                    Err(err) => {
                                        return Poll::Ready(Some(Err(err)));
                                    }
                                };
                                chunked.into_array()
                            }
                        });
                    }
                }
            }
        }

        // Step 2: decide how much to emit.
        // Emit length is the minimum of every child's currently
        // buffered head. Children whose head is `None` after refill
        // are EOF with nothing left — that means the aligned stream
        // is done.
        let mut emit_len = usize::MAX;
        for child in &this.children {
            match child.head.as_ref() {
                Some(arr) => emit_len = emit_len.min(arr.len()),
                None => return Poll::Ready(None),
            }
        }
        if emit_len == 0 {
            return Poll::Ready(None);
        }

        // Step 3: take `emit_len` rows from each child's head, slicing
        // and updating the buffer.
        let mut output = Vec::with_capacity(this.children.len());
        for child in this.children.iter_mut() {
            let head = child
                .head
                .take()
                .ok_or_else(|| vortex_err!("AlignedArrayStream: head missing after refill"));
            let head = match head {
                Ok(h) => h,
                Err(e) => return Poll::Ready(Some(Err(e))),
            };
            if head.len() == emit_len {
                output.push(head);
            } else {
                let front = match head.slice(0..emit_len) {
                    Ok(a) => a,
                    Err(e) => return Poll::Ready(Some(Err(e))),
                };
                let rest = match head.slice(emit_len..head.len()) {
                    Ok(a) => a,
                    Err(e) => return Poll::Ready(Some(Err(e))),
                };
                child.head = Some(rest);
                output.push(front);
            }
        }

        Poll::Ready(Some(Ok(output)))
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use futures::stream;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;

    use super::AlignedArrayStream;

    fn primitive_dtype() -> DType {
        DType::Primitive(PType::I32, NonNullable)
    }

    fn stream_of(parts: Vec<Vec<i32>>) -> SendableArrayStream {
        let dtype = primitive_dtype();
        let arrays: Vec<_> = parts
            .into_iter()
            .map(|part| Ok(PrimitiveArray::from_iter(part).into_array()))
            .collect();
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, stream::iter(arrays)))
    }

    #[test]
    fn aligns_misaligned_children() -> VortexResult<()> {
        block_on(|_| async move {
            // Field A: one big chunk of 10 rows.
            // Field B: two chunks of 4 + 6 rows.
            let left = stream_of(vec![vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
            let right = stream_of(vec![vec![11, 12, 13, 14], vec![15, 16, 17, 18, 19, 20]]);

            let mut s = AlignedArrayStream::new(vec![left, right]);
            let mut steps: Vec<(usize, usize)> = Vec::new();
            while let Some(item) = s.next().await {
                let pair = item?;
                steps.push((pair[0].len(), pair[1].len()));
            }
            // Step 1: B's first chunk is 4 rows → emit 4 from A and 4 from B.
            // Step 2: A has 6 left, B has 6 left → emit 6 each.
            assert_eq!(steps, vec![(4, 4), (6, 6)]);
            Ok::<_, vortex_error::VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn ends_when_any_child_eof_with_no_remainder() -> VortexResult<()> {
        block_on(|_| async move {
            let left = stream_of(vec![vec![1, 2, 3]]);
            let right = stream_of(vec![vec![10, 11, 12, 13, 14]]);
            let mut s = AlignedArrayStream::new(vec![left, right]);
            let mut steps = Vec::new();
            while let Some(item) = s.next().await {
                let pair = item?;
                steps.push((pair[0].len(), pair[1].len()));
            }
            // A only has 3 rows; we emit 3-row aligned slices then stop
            // (B has 2 leftover rows but no matching A — caller must
            // own the precondition that children agree on total rows).
            assert_eq!(steps, vec![(3, 3)]);
            Ok::<_, vortex_error::VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn min_rows_concats_until_threshold() -> VortexResult<()> {
        block_on(|_| async move {
            // Each child emits 4 small chunks of 3 rows = 12 rows total.
            let left = stream_of(vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
                vec![7, 8, 9],
                vec![10, 11, 12],
            ]);
            let right = stream_of(vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
                vec![7, 8, 9],
                vec![10, 11, 12],
            ]);

            let mut s = AlignedArrayStream::new(vec![left, right]).with_min_rows(8);
            let mut steps = Vec::new();
            while let Some(item) = s.next().await {
                let pair = item?;
                steps.push((pair[0].len(), pair[1].len()));
            }
            // First emit batches up to >=8 rows: 3+3+3 = 9. Emits 9.
            // Remaining 3 rows go in the second emit (only 3 left, less than
            // min_rows but also EOF, so we emit short).
            assert_eq!(steps, vec![(9, 9), (3, 3)]);
            Ok::<_, vortex_error::VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn skips_empty_inner_arrays() -> VortexResult<()> {
        block_on(|_| async move {
            let left = stream_of(vec![vec![], vec![1, 2, 3], vec![]]);
            let right = stream_of(vec![vec![10, 20, 30]]);
            let mut s = AlignedArrayStream::new(vec![left, right]);
            let mut steps = Vec::new();
            while let Some(item) = s.next().await {
                let pair = item?;
                steps.push((pair[0].len(), pair[1].len()));
            }
            assert_eq!(steps, vec![(3, 3)]);
            Ok::<_, vortex_error::VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn arity_matches_children() {
        let s = AlignedArrayStream::new(vec![
            stream_of(vec![vec![1]]),
            stream_of(vec![vec![1]]),
            stream_of(vec![vec![1]]),
        ]);
        assert_eq!(s.arity(), 3);
    }
}
