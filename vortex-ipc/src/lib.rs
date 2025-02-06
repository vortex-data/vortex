//! Vortex IPC messages and associated readers and writers.
//!
//! Vortex provides an IPC messaging format to exchange array data over a streaming
//! interface. The format emits message headers in FlatBuffer format, along with their
//! data buffers.
//!
//! This crate provides both in-memory message representations for holding IPC messages
//! before/after serialization, and streaming readers and writers that sit on top
//! of any type implementing `VortexRead` or `VortexWrite` respectively.

pub mod iterator;
pub mod messages;
pub mod stream;
