use crate::file::ReadSource;

impl ReadSource for tokio::fs::File {
    fn uri(&self) -> &std::sync::Arc<str> {
        todo!()
    }

    fn coalesce_window(&self) -> Option<super::CoalesceWindow> {
        todo!()
    }

    fn size(&self) -> futures::future::BoxFuture<'static, vortex_error::VortexResult<u64>> {
        todo!()
    }

    fn drive_send(
        self: std::sync::Arc<Self>,
        requests: futures::stream::BoxStream<'static, super::IoRequest>,
    ) -> futures::future::BoxFuture<'static, ()> {
        todo!()
    }
}
