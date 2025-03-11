//! An array that uses the [Fast Static Symbol Table][fsst] compression scheme
//! to compress string arrays.
//!
//! FSST arrays can generally compress string data up to 2x through the use of
//! string tables. The string table is static for an entire array, and occupies
//! up to 2048 bytes of buffer space. Thus, FSST is only worth reaching for when
//! dealing with larger arrays of potentially hundreds of kilobytes or more.
//!
//! [fsst]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf

mod array;
mod canonical;
mod compress;
mod compute;
mod serde;

pub use array::*;
pub use compress::*;
