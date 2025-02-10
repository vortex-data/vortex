use std::marker::PhantomData;
use std::sync::Arc;

use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::stream;
use vortex_array::stream::ArrayStream;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_io::VortexReadAt;
use vortex_layout::scan::{ScanDriver, ScanDriverV2};
use vortex_layout::segments::AsyncSegmentReader;

use crate::segments::SegmentCache;
use crate::{FileLayout, FileType, VortexFile};

pub struct FileV2<R>(PhantomData<R>);

impl<R: VortexReadAt> FileType for FileV2<R> {
    type Options = ();
    type Read = R;
    type ScanDriver = Self;

    fn scan_driver(
        _read: Self::Read,
        _options: Self::Options,
        _file_layout: FileLayout,
        _segment_cache: Arc<dyn SegmentCache>,
    ) -> Self::ScanDriver {
        FileV2(PhantomData)
    }
}

impl<R: 'static> ScanDriver for FileV2<R> {
    type Options = ();

    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
        todo!()
    }

    fn drive_stream(
        self,
        stream: impl Stream<Item = BoxFuture<'static, VortexResult<()>>> + Send + 'static,
    ) -> impl Stream<Item = VortexResult<()>> + 'static {
        stream::empty()
    }
}

impl<R> ScanDriverV2 for FileV2<R> {}

#[cfg(test)]
mod test {
    use vortex_array::stream::ArrayStreamExt;
    use vortex_error::VortexResult;
    use vortex_expr::Identity;
    use vortex_io::TokioFile;

    use super::*;
    use crate::VortexOpenOptions;

    #[tokio::test]
    async fn test_old_scan() -> VortexResult<()> {
        let file = TokioFile::open("../bench-vortex/data/clickbench/vortex/hits_0.vortex")?;
        let vxf = VortexOpenOptions::file(file).open().await?;

        let array = vxf.scan().into_array().await?;
        println!("{}", array.tree_display());

        Ok(())
    }

    #[tokio::test]
    async fn test_scan() -> VortexResult<()> {
        let file = TokioFile::open("../bench-vortex/data/clickbench/vortex/hits_0.vortex")?;
        let vxf = VortexOpenOptions::file2(file).open().await?;

        let array = vxf
            .scan()
            .build()?
            .into_stream_v2::<FileV2<TokioFile>>()
            .into_array()
            .await?;
        println!("{}", array.tree_display());

        Ok(())
    }
}
