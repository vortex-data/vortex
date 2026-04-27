// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Useful utilities for discovering the desired level of parallelism

use std::sync::LazyLock;

/// Estimates the degree of parallelism the program should use, caching the result after the first call.
///
/// This is currently implemented using [`std::thread::available_parallelism`], but might change in the future.
///
/// Returns `None` if the underlying functions fails.
pub fn get_available_parallelism() -> Option<usize> {
    #[allow(clippy::disallowed_methods)]
    static PARALLELISM: LazyLock<Option<usize>> =
        LazyLock::new(|| std::thread::available_parallelism().ok().map(|n| n.get()));

    *PARALLELISM
}
