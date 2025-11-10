// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::stats::ArrayStats;
use crate::vxo::vtable::ArrayVTable;
use std::any::Any;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::VortexResult;

/// A Vortex array.
///
/// Each array consists of a vtable, heap-allocated instance data, children and statistics.
pub struct Array2(Arc<Inner>);
struct Inner {
    vtable: ArrayVTable,
    data: Box<dyn Any>,
    dtype: DType,
    len: usize,
    children: Vec<Array2>,
    stats: ArrayStats,
}

impl Array2 {
    /// Creates a new array from its component parts, validating them using the vtable's
    /// validation function.
    pub fn try_from_parts(
        vtable: ArrayVTable,
        data: Box<dyn Any>,
        dtype: DType,
        len: usize,
        children: Vec<Array2>,
        stats: ArrayStats,
    ) -> VortexResult<Self> {
        let this = Array2(Arc::new(Inner {
            vtable,
            data,
            dtype,
            len,
            children,
            stats,
        }));

        // Validate the array using its vtable validation function
        this.0.vtable.as_dyn().validate(&this)?;

        Ok(this)
    }

    /// Creates a new array from its component parts without any validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the provided parts are valid according to the vtable's
    /// validation function.
    pub unsafe fn from_parts_unchecked(
        vtable: ArrayVTable,
        data: Box<dyn Any>,
        dtype: DType,
        len: usize,
        children: Vec<Array2>,
        stats: ArrayStats,
    ) -> Self {
        Array2(Arc::new(Inner {
            vtable,
            data,
            dtype,
            len,
            children,
            stats,
        }))
    }

    /// Returns the vtable for this array.
    pub fn vtable(&self) -> &ArrayVTable {
        &self.0.vtable
    }

    /// Returns the un-typed instance data for this array.
    ///
    /// Use [`ArrayView`](crate::vxo::view::ArrayView) to get typed access to the instance data.
    pub fn data(&self) -> &dyn Any {
        self.0.data.as_ref()
    }

    /// The data type of this array.
    pub fn dtype(&self) -> &DType {
        &self.0.dtype
    }

    /// The length of this array.
    pub fn len(&self) -> usize {
        self.0.len
    }

    /// The children of this array.
    pub fn children(&self) -> &[Array2] {
        &self.0.children
    }

    /// The statistics of this array.
    pub fn stats(&self) -> &ArrayStats {
        &self.0.stats
    }
}
