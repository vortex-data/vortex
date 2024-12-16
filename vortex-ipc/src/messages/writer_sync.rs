pub struct SyncMessageWriter<W> {
    write: W,
}

impl<W> SyncMessageWriter<W> {
    pub fn new(write: W) -> Self {
        Self { write }
    }
}
