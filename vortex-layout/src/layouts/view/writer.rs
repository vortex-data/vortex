//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(aduffy): implement the writer

use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use vortex_array::arrays::VarBinViewVTable;
use vortex_array::{Array, ArrayContext};
use vortex_error::vortex_bail;

use crate::segments::SequenceWriter;
use crate::{LayoutStrategy, SendableLayoutFuture, SendableSequentialStream};

pub struct ViewStrategy {
    fallback: Arc<dyn LayoutStrategy>,
}

impl LayoutStrategy for ViewStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        mut stream: SendableSequentialStream,
    ) -> SendableLayoutFuture {
        Box::pin(async move {
            let Some(chunk) = stream.next().await else {
                vortex_bail!("view layout needs a single chunk");
            };
            let (sequence_id, chunk) = chunk?;

            let row_count = chunk.len() as u64;

            // If the chunk is a VarBinView, serialize using our specialized layout.
            if let Some(view_array) = chunk.as_opt::<VarBinViewVTable>() {
                // If there is a validity layout, we assign it a child sequence ID.
                let validity_ptr = sequence_id.descend();
                validity_ptr.downgrade();

                // Serialize a child array containing the validity
                let views = view_array.views().clone();
                let buffers = view_array.buffers().iter().cloned().collect_vec();

                // Write a child layout for the validity.

                // Write all of the buffers as different segments
                sequence_writer.put()
            } else {
                // Use the fallback layout for non-view chunks.
                return self
                    .fallback
                    .write_stream(ctx, sequence_writer, stream)
                    .await;
            }
        })
    }
}
