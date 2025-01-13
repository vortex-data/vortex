/// The [`IoMode`] describes how the IO-bound tasks are executed.
pub enum IoMode {
    /// Blocks on the I/O tasks on the execution thread. This may be useful when the I/O source
    /// is trivially cheap, such as reading zero-copy from an in-memory buffer.
    Blocking,
    /// Polls the I/O tasks inline on the execution thread.
    /// [`vortex_array::stream::ArrayStream`].
    Inline,
}
