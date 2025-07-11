mod intersection;
mod repartition;

use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
pub use intersection::*;
pub use repartition::*;
use vortex_error::VortexResult;
use vortex_mask::Mask;

pub type MaskStream = BoxStream<'static, VortexResult<Mask>>;

pub trait MaskStreamExt: Stream<Item = VortexResult<Mask>> {
    fn repartition(self, target_size: usize) -> RepartitionMaskStream<'static>
    where
        Self: Sized + Send + 'static,
    {
        RepartitionMaskStream::new(self.boxed(), target_size)
    }
}

impl<S: Stream<Item = VortexResult<Mask>> + ?Sized> MaskStreamExt for S {}
