// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Map;
use crate::arrays::MapArray;
use crate::arrays::map::MapArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::ListViewBuilder;
use crate::dtype::DType;
use crate::dtype::MapDType;
use crate::dtype::Nullability;
use crate::dtype::OffsetBuilderPType;
use crate::scalar::MapScalar;
use crate::scalar::Scalar;

/// A builder for canonical [`MapArray`] values.
///
/// The builder owns a [`ListViewBuilder`] whose elements are non-nullable `{key, value}` structs.
/// It preserves the map dtype's `keys_sorted` assertion while delegating offsets, sizes, and outer
/// validity to that list-view builder.
pub struct MapBuilder<O: OffsetBuilderPType, S: OffsetBuilderPType> {
    dtype: DType,
    map_dtype: MapDType,
    entries_builder: ListViewBuilder<O, S>,
}

impl<O: OffsetBuilderPType, S: OffsetBuilderPType> MapBuilder<O, S> {
    /// Creates a map builder with the default capacity.
    pub fn new(map_dtype: MapDType, nullability: Nullability) -> Self {
        Self::with_capacity(map_dtype, nullability, DEFAULT_BUILDER_CAPACITY)
    }

    /// Creates a map builder with space for `capacity` map rows.
    pub fn with_capacity(map_dtype: MapDType, nullability: Nullability, capacity: usize) -> Self {
        let entries_builder = ListViewBuilder::with_capacity(
            Arc::new(map_dtype.entries_dtype()),
            nullability,
            capacity.saturating_mul(2),
            capacity,
        );
        let dtype = DType::Map(map_dtype.clone(), nullability);
        Self {
            dtype,
            map_dtype,
            entries_builder,
        }
    }

    /// Appends one map scalar.
    pub fn append_value(&mut self, value: MapScalar<'_>) -> VortexResult<()> {
        vortex_ensure!(
            value.dtype() == &self.dtype,
            "MapBuilder expected map scalar with dtype {}, got {}",
            self.dtype,
            value.dtype()
        );

        if value.is_null() {
            self.entries_builder.append_null();
            return Ok(());
        }

        let entry_dtype = self.map_dtype.entries_dtype();
        let entries = value
            .entries()
            .map(|(key, value)| Scalar::struct_(entry_dtype.clone(), vec![key, value]))
            .collect();
        let entries = Scalar::list(Arc::new(entry_dtype), entries, self.dtype.nullability());
        self.entries_builder.append_value(entries.as_list())
    }

    /// Finishes the builder directly into a [`MapArray`].
    pub fn finish_into_map(&mut self) -> MapArray {
        MapArray::new(
            self.map_dtype.clone(),
            self.entries_builder.finish_into_listview(),
        )
    }

    /// Appends the values of a [`Map`]-encoded `array` to this builder.
    pub fn append_map_array(
        &mut self,
        array: ArrayView<'_, Map>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure!(
            array.dtype() == self.dtype(),
            "MapBuilder expected map array with dtype {}, got {}",
            self.dtype(),
            array.dtype()
        );
        self.entries_builder
            .append_listview_array(array.entries(), ctx)
    }
}

impl<O: OffsetBuilderPType, S: OffsetBuilderPType> ArrayBuilder for MapBuilder<O, S> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.entries_builder.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.entries_builder.append_zeros(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        unsafe { self.entries_builder.append_nulls_unchecked(n) };
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "MapBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );
        self.append_value(scalar.as_map())
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.entries_builder.reserve_exact(additional);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        unsafe { self.entries_builder.set_validity_unchecked(validity) };
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_map().into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::Map(self.finish_into_map())
    }
}
