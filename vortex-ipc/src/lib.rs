//! Vortex IPC messages and associated readers and writers.
//!
//! Vortex provides an IPC messaging format to exchange array data over a streaming
//! interface. The format emits message headers in FlatBuffer format, along with their
//! data buffers.
//!
//! This crate provides both in-memory message representations for holding IPC messages
//! before/after serialization, as well as streaming readers and writers that sit on top
//! of any type implementing `VortexRead` or `VortexWrite` respectively.

pub mod messages;
pub mod stream_reader;
pub mod stream_writer;

/// All messages in Vortex are aligned to start at a multiple of 64 bytes.
///
/// This is a multiple of the native alignment for all PTypes,
/// thus all buffers allocated with this alignment are naturally aligned
/// for any data we may put inside of it.
pub const ALIGNMENT: usize = 64;
