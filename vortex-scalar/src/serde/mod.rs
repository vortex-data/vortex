#[cfg(feature = "flatbuffers")]
mod flatbuffers;

#[cfg(feature = "proto")]
mod proto;

#[allow(clippy::module_inception)]
mod serde;
