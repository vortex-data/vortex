// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads through [t4](::t4), a cache of an object store in memory and on local drives.

mod read_at;

pub use read_at::*;
// t4's cache and its opened files, re-exported so that the cache a `T4ReadAt` reads through is
// built with the t4 Vortex is built against.
pub use t4::CachedFile;
pub use t4::T4Caching;
pub use t4::T4CachingBuilder;
