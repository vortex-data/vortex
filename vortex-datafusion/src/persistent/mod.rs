//! Persistent implementation of a Vortex table provider.
mod cache;
mod config;
mod execution;
mod format;
mod opener;
mod sink;

pub use format::{VortexFormat, VortexFormatFactory, VortexFormatOptions};
