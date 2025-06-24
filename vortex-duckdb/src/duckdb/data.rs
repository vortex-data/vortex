use crate::duckdb::drop_boxed;
use crate::{cpp, wrapper};

wrapper!(Data, cpp::duckdb_vx_data, |_| {});

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
