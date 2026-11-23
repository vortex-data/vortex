// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opt-in, request-derived S3 simulation for benchmark clients using the S3 HTTP protocol.

mod model;
mod server;

pub use model::Profile;
pub use server::Server;

#[cfg(test)]
mod tests;
