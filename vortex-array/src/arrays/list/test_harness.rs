// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::ListArray;
use crate::builders::ArrayBuilder;
use crate::builders::ListBuilder;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::scalar::Scalar;

impl ListArray {
    /// This is a convenience method to create a list array from an iterator of iterators.
    /// This method is slow however since each element is first converted to a scalar and then
    /// appended to the array.
    pub fn from_iter_slow<O: IntegerPType, I: IntoIterator>(
        iter: I,
        dtype: Arc<DType>,
    ) -> VortexResult<ArrayRef>
    where
        I::Item: IntoIterator,
        <I::Item as IntoIterator>::Item: Into<Scalar>,
    {
        let iter = iter.into_iter();
        let mut builder = ListBuilder::<O>::with_capacity(
            Arc::clone(&dtype),
            crate::dtype::Nullability::NonNullable,
            2 * iter.size_hint().0,
            iter.size_hint().0,
        );

        for v in iter {
            let elem = Scalar::list(
                Arc::clone(&dtype),
                v.into_iter().map(|x| x.into()).collect_vec(),
                dtype.nullability(),
            );
            builder.append_value(elem.as_list())?
        }
        Ok(builder.finish())
    }

    pub fn from_iter_opt_slow<O: IntegerPType, I: IntoIterator<Item = Option<T>>, T>(
        iter: I,
        dtype: Arc<DType>,
    ) -> VortexResult<ArrayRef>
    where
        T: IntoIterator,
        T::Item: Into<Scalar>,
    {
        let iter = iter.into_iter();
        let mut builder = ListBuilder::<O>::with_capacity(
            Arc::clone(&dtype),
            crate::dtype::Nullability::Nullable,
            2 * iter.size_hint().0,
            iter.size_hint().0,
        );

        for v in iter {
            if let Some(v) = v {
                let elem = Scalar::list(
                    Arc::clone(&dtype),
                    v.into_iter().map(|x| x.into()).collect_vec(),
                    dtype.nullability(),
                );
                builder.append_value(elem.as_list())?
            } else {
                builder.append_null()
            }
        }
        Ok(builder.finish())
    }
}
