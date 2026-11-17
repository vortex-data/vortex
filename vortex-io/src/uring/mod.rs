// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experimental: one io_uring shared by every file it opens, behind [`Openable`] and
//! [`VortexReadAt`].
//!
//! This is a proof of concept for the plan-level IO protocol, not a tuned backend. It checks that
//! a ring needs nothing from the protocol beyond the two existing traits:
//!
//! - [`UringOpenable`] opens a path with `IORING_OP_OPENAT` on a shared [`UringRing`]. It needs
//!   no [`open_service`](Openable::open_service) override: the default
//!   [`ReadAtIoService`](crate::request::ReadAtIoService) already creates a read's future when
//!   the request is submitted, which is all a ring needs to batch.
//! - [`UringReadAt`] holds the opened descriptor and the same ring, so the reads of every file
//!   opened through one ring share its submission queue.
//!
//! Operations are queued when their future is *created* and submitted when any operation of the
//! ring is first *polled*, so everything queued in between, across files and across opens and
//! reads, goes to the kernel in one `io_uring_enter`. Reads the page cache holds complete inside
//! that call and are reaped by the polling thread. A reaper thread waits for the rest and wakes
//! their futures, so a future can be polled from any executor.
//!
//! Buffers, paths, and descriptors belong to the ring until the kernel completes the operation
//! that uses them, so dropping a future with its operation in flight is safe.

mod open;
mod read_at;
mod ring;
#[cfg(test)]
mod tests;

pub use open::UringOpenable;
pub use read_at::UringReadAt;
pub use ring::UringRing;
pub use ring::UringStats;

#[cfg(doc)]
use crate::VortexReadAt;
#[cfg(doc)]
use crate::request::Openable;
