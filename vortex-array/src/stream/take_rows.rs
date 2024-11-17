use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::{ready, Stream};
use pin_project::pin_project;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::Scalar;

use crate::compute::unary::subtract_scalar;
use crate::compute::{search_sorted, slice, take, SearchSortedSide, TakeOptions};
use crate::stats::{ArrayStatistics, Stat};
use crate::stream::ArrayStream;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};

#[pin_project]
pub struct TakeRows<R: ArrayStream> {
    #[pin]
    reader: R,
    indices: ArrayData,
    row_offset: usize,
}

impl<R: ArrayStream> TakeRows<R> {
    pub fn try_new(reader: R, indices: ArrayData) -> VortexResult<Self> {
        if !indices.is_empty() {
            if !indices.statistics().compute_is_sorted().unwrap_or(false) {
                vortex_bail!("Indices must be sorted to take from IPC stream")
            }

            if indices
                .statistics()
                .compute_null_count()
                .map(|nc| nc > 0)
                .unwrap_or(true)
            {
                vortex_bail!("Indices must not contain nulls")
            }

            if !indices.dtype().is_int() {
                vortex_bail!("Indices must be integers")
            }

            if indices.dtype().is_signed_int()
                && indices
                    .statistics()
                    .compute_as_cast::<i64>(Stat::Min)
                    .map(|min| min < 0)
                    .unwrap_or(true)
            {
                vortex_bail!("Indices must be positive")
            }
        }

        Ok(Self {
            reader,
            indices,
            row_offset: 0,
        })
    }
}

impl<R: ArrayStream> Stream for TakeRows<R> {
    type Item = VortexResult<ArrayData>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if this.indices.is_empty() {
            return Poll::Ready(None);
        }

        while let Some(batch) = ready!(this.reader.as_mut().poll_next(cx)?) {
            let curr_offset = *this.row_offset;
            let left = search_sorted(this.indices, curr_offset, SearchSortedSide::Left)?.to_index();
            let right = search_sorted(
                this.indices,
                curr_offset + batch.len(),
                SearchSortedSide::Left,
            )?
            .to_index();

            *this.row_offset += batch.len();

            if left == right {
                continue;
            }

            // TODO(ngates): this is probably too heavy to run on the event loop. We should spawn
            //  onto a worker pool.
            let indices_for_batch = slice(this.indices, left, right)?.into_primitive()?;
            let shifted_arr = match_each_integer_ptype!(indices_for_batch.ptype(), |$T| {
                subtract_scalar(&indices_for_batch.into_array(), &Scalar::from(curr_offset as $T))?
            });
            return Poll::Ready(
                take(&batch, &shifted_arr, TakeOptions::default())
                    .map(Some)
                    .transpose(),
            );
        }

        Poll::Ready(None)
    }
}
