//! Implementation of Dictionary encoding.
//!
//! Expose a [DictArray] which is zero-copy equivalent to Arrow's
//! [DictionaryArray](https://docs.rs/arrow/latest/arrow/array/struct.DictionaryArray.html).
pub use array::*;
pub use compress::*;
pub use primitive_builder::*;
pub use varbin_builder::*;
pub use varbinview_builder::*;

mod array;
mod compress;
mod compute;
mod primitive_builder;
mod stats;
mod varbin_builder;
mod varbinview_builder;
mod variants;
