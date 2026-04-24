// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cpp;
use crate::duckdb::drop_boxed;
use crate::lifetime_wrapper;

// This data wrapper is used to create an external data object that can be passed to and
// freed by DuckDB.

lifetime_wrapper!(Data, cpp::duckdb_vx_data, |_| {});

impl<T> From<Box<T>> for Data {
    fn from(value: Box<T>) -> Self {
        unsafe {
            Self::own(cpp::duckdb_vx_data_create(
                Box::into_raw(value).cast(),
                Some(drop_boxed::<T>),
            ))
        }
    }
}
