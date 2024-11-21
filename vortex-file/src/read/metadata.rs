use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use vortex_io::VortexReadAt;

use super::LayoutReader;

pub struct MetadataReader<R: VortexReadAt> {
    input: R,
    dispatcher: Arc<IoDispatcher>,
    root_layout: Box<dyn LayoutReader>,
    state: State,
}

enum State {}

impl<R: VortexReadAt> Future for MetadataReader<R> {
    type Output = VortexResult<Vec<ArrayData>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        todo!()
    }
}
