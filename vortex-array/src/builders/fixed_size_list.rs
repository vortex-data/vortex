// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::ListScalar;

use crate::arrays::FixedSizeListArray;
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::builders::{ArrayBuilder, ArrayBuilderExt, builder_with_capacity};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

/// An [`ArrayBuilder`] for creating [`FixedSizeListArray`].
pub struct FixedSizeListBuilder {
    /// The [`DType`] of the `FixedSizeList`. This **must** be a [`DType::FixedSizeList`].
    dtype: DType,

    /// The builder for the underlying elements of the [`FixedSizeListArray`].
    ///
    /// This builder will have a capacity equal to the `list_size * capacity`.
    elements_builder: Box<dyn ArrayBuilder>,

    /// The null map builder of the [`FixedSizeListArray`].
    ///
    /// We also use this type to store the length of the final output array.
    nulls: LazyNullBufferBuilder,
}

impl FixedSizeListBuilder {
    /// Creates a new [`FixedSizeListArray`] builder.
    pub fn with_capacity(
        element_dtype: Arc<DType>,
        list_size: u32,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        let elements_capacity = capacity * list_size as usize;

        let elements_builder = builder_with_capacity(&element_dtype, elements_capacity);
        let fsl_dtype = DType::FixedSizeList(element_dtype, list_size, nullability);
        let nulls = LazyNullBufferBuilder::new(capacity);

        Self {
            dtype: fsl_dtype,
            elements_builder,
            nulls,
        }
    }

    /// Adds a value to the [`FixedSizeListArray`] that we are building.
    ///
    /// Note that a [`ListScalar`] can represent both a [`ListArray`] scalar **and** a
    /// [`FixedSizeListArray`] scalar.
    ///
    /// [`ListArray`]: crate::arrays::ListArray
    pub fn append_value(&mut self, value: ListScalar) -> VortexResult<()> {
        // Validate that the list scalar has an equal size to our `list_size`.

        if value.len() != self.list_size() as usize {
            vortex_bail!(
                "Tried to append a `ListScalar` with length {} to a `FixedSizeListScalar` with fixed size of {}",
                value.len(),
                self.list_size()
            );
        }

        let Some(elements) = value.elements() else {
            // If `elements` is `None`, then the `value` is a null value.
            self.append_null();
            return Ok(());
        };

        for scalar in elements {
            // TODO(connor): This is slow, we should be able to append multiple values at once, or
            // the list scalar should hold an Array
            self.elements_builder.append_scalar(&scalar)?;
        }
        self.nulls.append_non_null();

        Ok(())
    }

    pub fn list_size(&self) -> u32 {
        let DType::FixedSizeList(_, list_size, _) = self.dtype else {
            vortex_panic!(
                "`FixedSizeListBuilder` has an incorrect dtype: {}",
                self.dtype
            );
        };

        list_size
    }

    pub fn nullability(&self) -> Nullability {
        let DType::FixedSizeList(_, _, nullability) = self.dtype else {
            vortex_panic!(
                "`FixedSizeListBuilder` has an incorrect dtype: {}",
                self.dtype
            );
        };

        nullability
    }

    /// Returns the current length of underlying `elements` array.
    fn elements_len(&self) -> usize {
        self.len() * self.list_size() as usize
    }
}

impl ArrayBuilder for FixedSizeListBuilder {
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
        self.nulls.len()
    }

    /// We define the "zero" value of a fixed-size list of size `m` to be a list of `m` zero values
    /// (where "zero value" is defined by the element [`DType`]).
    ///
    /// We append `n * m` of these values to the underlying `elements` array.
    fn append_zeros(&mut self, n: usize) {
        self.elements_builder
            .append_zeros(n * self.list_size() as usize);
        self.nulls.append_n_non_nulls(n);
    }

    /// We define the null value of a fixed-size list of size `m` to be a list of `m` null values.
    ///
    /// We append `n * m` of these null values to the underlying `elements` array.
    fn append_nulls(&mut self, n: usize) {
        self.elements_builder
            .append_nulls(n * self.list_size() as usize);
        self.nulls.append_n_nulls(n);
    }

    /// This will increase the capacity if extending with this `array` would go past the original
    /// capacity.
    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        // TODO(connor): This check should be in every single `extend_from_array` implementation.
        assert_eq!(
            self.dtype(),
            array.dtype(),
            "tried to extend an array with an array of a different `DType`"
        );

        let fsl = array.to_fixed_size_list()?;
        if fsl.is_empty() {
            return Ok(());
        }

        let new_elements = fsl.elements();

        self.elements_builder
            .ensure_capacity(self.elements_len() + new_elements.len());
        self.elements_builder.extend_from_array(new_elements)?;

        self.nulls.append_validity_mask(array.validity_mask());

        Ok(())
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.elements_builder
            .ensure_capacity(capacity * self.list_size() as usize);
        self.nulls.ensure_capacity(capacity);
    }

    fn set_validity(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        assert_eq!(
            self.elements_builder.len(),
            self.nulls.len() * self.list_size() as usize,
            "elements length must be equal to the null length times the list size"
        );

        // TODO(connor): Use `new_unchecked` here.
        FixedSizeListArray::try_new(
            self.elements_builder.finish(),
            self.list_size(),
            self.nulls.finish_with_nullability(self.nullability()),
            self.len(),
        )
        .vortex_expect("tried to create an invalid `FixedSizeListArray` from a builder")
        .into_array()
    }
}
