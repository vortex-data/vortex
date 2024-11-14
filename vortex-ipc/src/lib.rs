pub mod messages;
pub mod stream_reader;
pub mod stream_writer;

/// All messages in Vortex are aligned to start at a multiple of 64 bytes.
///
/// This is a multiple of the native alignment for all PTypes,
/// thus all buffers allocated with this alignment are naturally aligned
/// for any data we may put inside of it.
pub const ALIGNMENT: usize = 64;
