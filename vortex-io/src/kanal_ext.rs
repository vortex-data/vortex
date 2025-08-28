// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_stream::stream;
use futures::Stream;
use kanal::AsyncReceiver;

pub trait KanalExt<T> {
    fn into_stream(self) -> impl Stream<Item = T>;
}

impl<T> KanalExt<T> for AsyncReceiver<T> {
    fn into_stream(self) -> impl Stream<Item = T> {
        stream! {
            // The Err case indicates the sender / channel has been closed so we terminate
            // the stream.
            while let Ok(next) = self.recv().await {
                yield next
            }
        }
    }
}
