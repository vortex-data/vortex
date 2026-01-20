// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::try_join_all;
use futures::pin_mut;
use itertools::Itertools;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::list_from_list_view;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::Handle;

use crate::IntoLayout as _;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

trait ToU64 {
    fn to_u64(self) -> u64;
}

impl ToU64 for u8 {
    fn to_u64(self) -> u64 {
        u64::from(self)
    }
}

impl ToU64 for u16 {
    fn to_u64(self) -> u64 {
        u64::from(self)
    }
}

impl ToU64 for u32 {
    fn to_u64(self) -> u64 {
        u64::from(self)
    }
}

impl ToU64 for u64 {
    fn to_u64(self) -> u64 {
        self
    }
}

/// A write strategy that performs component shredding for list types.
///
/// - Variable-size lists are written as:
///   - optional validity (is_valid: bool)
///   - offsets (u64, length = rows + 1)
///   - elements (concatenated)
/// - Fixed-size lists are written as:
///   - optional validity (is_valid: bool)
///   - elements (concatenated)
#[derive(Clone)]
pub struct ListStrategy {
    validity: Arc<dyn LayoutStrategy>,
    offsets: Arc<dyn LayoutStrategy>,
    elements: Arc<dyn LayoutStrategy>,
}

impl ListStrategy {
    pub fn new(
        validity: Arc<dyn LayoutStrategy>,
        offsets: Arc<dyn LayoutStrategy>,
        elements: Arc<dyn LayoutStrategy>,
    ) -> Self {
        Self {
            validity,
            offsets,
            elements,
        }
    }
}

#[async_trait]
impl LayoutStrategy for ListStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        let is_nullable = dtype.is_nullable();
        let offsets_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);

        let (stream_count, column_dtypes): (usize, Vec<DType>) = match &dtype {
            DType::List(element_dtype, _) => {
                let mut dtypes = Vec::new();
                if is_nullable {
                    dtypes.push(DType::Bool(Nullability::NonNullable));
                }
                dtypes.push(offsets_dtype.clone());
                dtypes.push((**element_dtype).clone());
                (dtypes.len(), dtypes)
            }
            DType::FixedSizeList(element_dtype, ..) => {
                let mut dtypes = Vec::new();
                if is_nullable {
                    dtypes.push(DType::Bool(Nullability::NonNullable));
                }
                dtypes.push((**element_dtype).clone());
                (dtypes.len(), dtypes)
            }
            _ => {
                vortex_bail!("ListStrategy expected list dtype, got {}", dtype);
            }
        };

        let (column_streams_tx, column_streams_rx): (Vec<_>, Vec<_>) =
            (0..stream_count).map(|_| kanal::bounded_async(1)).unzip();

        let total_rows = Arc::new(AtomicU64::new(0));

        // Spawn a task to fan out chunk components to their respective transposed streams.
        {
            let total_rows = total_rows.clone();
            let dtype = dtype.clone();
            handle
                .spawn(async move {
                    let mut base_elements: u64 = 0;
                    let mut first_offsets = true;

                    pin_mut!(stream);
                    while let Some(result) = stream.next().await {
                        match result {
                            Ok((sequence_id, chunk)) => {
                                total_rows.fetch_add(chunk.len() as u64, Ordering::SeqCst);

                                let mut sequence_pointer = sequence_id.descend();

                                // validity (optional)
                                if is_nullable {
                                    let validity = chunk.validity_mask().into_array();
                                    let _ = column_streams_tx[0]
                                        .send(Ok((sequence_pointer.advance(), validity)))
                                        .await;
                                }

                                match &dtype {
                                    DType::List(..) => {
                                        let list_view = chunk.to_listview();
                                        let list = match list_from_list_view(list_view) {
                                            Ok(list) => list,
                                            Err(e) => {
                                                let e: Arc<VortexError> = Arc::new(e);
                                                for tx in column_streams_tx.iter() {
                                                    let _ = tx
                                                        .send(Err(VortexError::from(e.clone())))
                                                        .await;
                                                }
                                                break;
                                            }
                                        };

                                        // Build global u64 offsets, dropping the leading 0 for all but the first chunk.
                                        let offsets = list.offsets().to_primitive();
                                        let offsets_slice_u64: VortexResult<Vec<u64>> =
                                            match offsets.ptype() {
                                                ptype if ptype.is_unsigned_int() => vortex_dtype::match_each_unsigned_integer_ptype!(ptype, |T| {
                                                    Ok(offsets
                                                        .as_slice::<T>()
                                                        .iter()
                                                        .map(|&v| v.to_u64())
                                                        .collect())
                                                }),
                                                ptype if ptype.is_signed_int() => {
                                                    vortex_dtype::match_each_signed_integer_ptype!(
                                                        ptype,
                                                        |T| {
                                                        offsets
                                                            .as_slice::<T>()
                                                            .iter()
                                                            .map(|&v| {
                                                                u64::try_from(v).map_err(|_| {
                                                                    vortex_err!(
                                                                        "List offsets must be convertible to u64"
                                                                    )
                                                                })
                                                            })
                                                            .collect()
                                                        }
                                                    )
                                                }
                                                other => Err(vortex_err!(
                                                    "List offsets must be an integer type, got {other}"
                                                )),
                                            };
                                        let offsets_slice_u64 = match offsets_slice_u64 {
                                            Ok(v) => v,
                                            Err(e) => {
                                                let e: Arc<VortexError> = Arc::new(e);
                                                for tx in column_streams_tx.iter() {
                                                    let _ = tx
                                                        .send(Err(VortexError::from(e.clone())))
                                                        .await;
                                                }
                                                break;
                                            }
                                        };

                                        let mut adjusted: Vec<u64> = Vec::with_capacity(
                                            offsets_slice_u64
                                                .len()
                                                .saturating_sub((!first_offsets) as usize),
                                        );
                                        for (i, v) in offsets_slice_u64.into_iter().enumerate() {
                                            if !first_offsets && i == 0 {
                                                continue;
                                            }
                                            adjusted.push(v + base_elements);
                                        }

                                        let offsets_arr =
                                            vortex_array::arrays::PrimitiveArray::from_iter(
                                                adjusted,
                                            )
                                            .into_array();

                                        // offsets index depends on nullable validity child
                                        let offsets_idx = if is_nullable { 1 } else { 0 };
                                        let elements_idx = offsets_idx + 1;

                                        let _ = column_streams_tx[offsets_idx]
                                            .send(Ok((sequence_pointer.advance(), offsets_arr)))
                                            .await;
                                        let _ = column_streams_tx[elements_idx]
                                            .send(Ok((
                                                sequence_pointer.advance(),
                                                list.elements().clone(),
                                            )))
                                            .await;

                                        base_elements += list.elements().len() as u64;
                                        first_offsets = false;
                                    }
                                    DType::FixedSizeList(..) => {
                                        let list = chunk.to_fixed_size_list();

                                        let elements_idx = if is_nullable { 1 } else { 0 };
                                        let _ = column_streams_tx[elements_idx]
                                            .send(Ok((
                                                sequence_pointer.advance(),
                                                list.elements().clone(),
                                            )))
                                            .await;
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            Err(e) => {
                                let e: Arc<VortexError> = Arc::new(e);
                                for tx in column_streams_tx.iter() {
                                    let _ = tx.send(Err(VortexError::from(e.clone()))).await;
                                }
                                break;
                            }
                        }
                    }
                })
                .detach();
        }

        let layout_futures: Vec<_> = column_dtypes
            .into_iter()
            .zip_eq(column_streams_rx)
            .enumerate()
            .map(|(index, (dtype, recv))| {
                let column_stream =
                    SequentialStreamAdapter::new(dtype.clone(), recv.into_stream().boxed())
                        .sendable();
                let child_eof = eof.split_off();
                handle.spawn_nested(|h| {
                    let validity = self.validity.clone();
                    let offsets = self.offsets.clone();
                    let elements = self.elements.clone();
                    let ctx = ctx.clone();
                    let segment_sink = segment_sink.clone();
                    async move {
                        if is_nullable && index == 0 {
                            validity
                                .write_stream(ctx, segment_sink, column_stream, child_eof, h)
                                .await
                        } else if matches!(dtype, DType::Primitive(PType::U64, _)) {
                            offsets
                                .write_stream(ctx, segment_sink, column_stream, child_eof, h)
                                .await
                        } else {
                            elements
                                .write_stream(ctx, segment_sink, column_stream, child_eof, h)
                                .await
                        }
                    }
                })
            })
            .collect();

        let children = try_join_all(layout_futures).await?;

        let row_count = total_rows.load(Ordering::SeqCst);

        // Basic invariant: for variable-size lists, offsets must have row_count + 1 entries.
        if matches!(dtype, DType::List(..)) {
            let offsets_layout = if is_nullable {
                &children[1]
            } else {
                &children[0]
            };
            vortex_ensure!(
                offsets_layout.row_count() == row_count + 1,
                "ListLayout offsets row_count {} does not match list row_count + 1 ({})",
                offsets_layout.row_count(),
                row_count + 1
            );
        }

        Ok(ListLayout::new(row_count, dtype, children).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.elements.buffered_bytes()
    }
}
