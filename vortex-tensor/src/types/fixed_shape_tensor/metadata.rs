// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use itertools::Either;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

/// Element type, logical shape, and layout of a fixed-shape tensor.
///
/// The storage is a fixed-size list of non-nullable primitive elements. Its size is the product of
/// the logical shape dimensions, or one for a scalar tensor with an empty shape. A zero dimension
/// gives a list size of zero. Dtype construction requires every dimension and the list size to fit
/// `u32`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FixedShapeTensorMetadata {
    /// The element type of the tensor.
    ///
    /// The serialized metadata leaves it out, because the storage dtype holds it.
    element_ptype: PType,
    stored_metadata: FixedShapeTensorStoredMetadata,
}

/// Shape, dimension names, and permutation stored alongside a tensor's storage dtype.
///
/// Shape operations are independent of the element type. [`FixedShapeTensorMetadata`] adds that
/// type to declare storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FixedShapeTensorStoredMetadata {
    /// The logical shape of the tensor.
    ///
    /// `logical_shape[i]` is the size of the `i`-th logical dimension. When a `permutation` is
    /// present, the physical shape (i.e., the row-major memory layout) is derived as
    /// `physical_shape[permutation[i]] = logical_shape[i]`.
    ///
    /// May be empty (0D scalar tensor) or contain dimensions of size 0 (degenerate tensor).
    logical_shape: Vec<usize>,
    /// Optional names for each logical dimension. Each name corresponds to an entry in
    /// `logical_shape`.
    ///
    /// If names exist, there must be an equal number of names to logical dimensions.
    dim_names: Option<Vec<String>>,
    /// The permutation of the tensor's dimensions. `permutation[i]` is the physical dimension
    /// index that logical dimension `i` maps to.
    ///
    /// If this is `None`, then the logical and physical layouts are identical, equivalent to
    /// the identity permutation `[0, 1, ..., N-1]`.
    permutation: Option<Vec<usize>>,
}

impl FixedShapeTensorStoredMetadata {
    /// Creates stored tensor metadata with the given logical shape.
    ///
    /// Use [`with_dim_names`](Self::with_dim_names) and
    /// [`with_permutation`](Self::with_permutation) to further configure the metadata.
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            logical_shape: shape,
            dim_names: None,
            permutation: None,
        }
    }

    /// Sets the dimension names for this tensor.
    ///
    /// An empty list leaves the metadata unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if a non-empty list has a different length from the logical shape.
    pub fn with_dim_names(mut self, names: Vec<String>) -> VortexResult<Self> {
        if !names.is_empty() {
            vortex_ensure_eq!(names.len(), self.logical_shape.len());
            self.dim_names = Some(names);
        }

        Ok(self)
    }

    /// Sets the permutation from logical dimensions to physical dimensions.
    ///
    /// An empty list leaves the metadata unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if a non-empty list is not a permutation of `[0, 1, ..., N-1]`, where `N`
    /// is the number of logical dimensions.
    pub fn with_permutation(mut self, permutation: Vec<usize>) -> VortexResult<Self> {
        if !permutation.is_empty() {
            vortex_ensure_eq!(permutation.len(), self.logical_shape.len());

            // Verify this is actually a permutation of [0..N).
            let mut seen = vec![false; permutation.len()];
            for &p in &permutation {
                vortex_ensure!(
                    p < permutation.len(),
                    "permutation index {p} is out of range for {} dimensions",
                    permutation.len()
                );
                vortex_ensure!(!seen[p], "permutation contains duplicate index {p}");
                seen[p] = true;
            }

            self.permutation = Some(permutation);
        }

        Ok(self)
    }

    /// Returns the number of dimensions (rank) of the tensor.
    pub fn ndim(&self) -> usize {
        self.logical_shape.len()
    }

    /// Returns the logical dimensions of the tensor as a slice.
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Returns the dimension names, if set.
    pub fn dim_names(&self) -> Option<&[String]> {
        self.dim_names.as_deref()
    }

    /// Returns the permutation, if set.
    pub fn permutation(&self) -> Option<&[usize]> {
        self.permutation.as_deref()
    }

    /// Returns an iterator over the physical shape of the tensor.
    ///
    /// The physical shape describes the row-major memory layout. It is derived from the logical
    /// shape by placing each logical dimension's size at its physical position:
    /// `physical_shape[permutation[i]] = logical_shape[i]`.
    ///
    /// When no permutation is present, the physical shape is identical to the logical shape.
    pub fn physical_shape(&self) -> impl Iterator<Item = usize> + '_ {
        let ndim = self.logical_shape.len();
        let permutation = self.permutation.as_deref();

        match permutation {
            None => Either::Left(self.logical_shape.iter().copied()),
            Some(perm) => Either::Right(
                (0..ndim).map(move |p| self.logical_shape[Self::inverse_perm(perm, p)]),
            ),
        }
    }

    /// Returns an iterator over the strides for each logical dimension of the tensor.
    ///
    /// The stride for a logical dimension is the number of elements to skip in the flat backing
    /// array in order to move one step along that logical dimension.
    ///
    /// When a permutation is present, the physical memory is laid out in row-major order over the
    /// physical dimensions (the logical dimensions reordered by the permutation), so the strides
    /// are derived from that physical layout.
    pub fn strides(&self) -> impl Iterator<Item = usize> + '_ {
        let ndim = self.logical_shape.len();
        let permutation = self.permutation.as_deref();

        match permutation {
            None => Either::Left(
                (0..ndim).map(|i| self.logical_shape[i + 1..].iter().product::<usize>()),
            ),
            Some(permutation) => {
                Either::Right((0..ndim).map(|i| self.permuted_stride(i, permutation)))
            }
        }
    }

    /// Computes the stride for logical dimension `i` given a `permutation`.
    ///
    /// The stride is the product of `logical_shape[j]` for all logical dimensions `j` whose
    /// physical position (`perm[j]`) comes after the physical position of dimension `i`
    /// (`perm[i]`).
    fn permuted_stride(&self, i: usize, perm: &[usize]) -> usize {
        let phys = perm[i];

        // Each call scans the full permutation, making `strides()` O(ndim^2) overall. Tensor rank
        // is typically small, so avoiding a Vec allocation is a net win.
        perm.iter()
            .enumerate()
            .filter(|&(_, &p)| p > phys)
            .map(|(l, _)| self.logical_shape[l])
            .product::<usize>()
    }

    /// Returns the logical dimension index that maps to physical position `p`. This is the
    /// inverse of the permutation: if `perm[i] == p`, returns `i`.
    ///
    /// Each call is a linear scan of `perm`, making callers that invoke this for every physical
    /// position O(ndim^2) overall. Tensor rank is typically small (2–5), so avoiding a Vec
    /// allocation for the full inverse permutation is a net win.
    fn inverse_perm(perm: &[usize], p: usize) -> usize {
        perm.iter()
            .position(|&pi| pi == p)
            .vortex_expect("permutation must contain every physical position exactly once")
    }
}

impl FixedShapeTensorMetadata {
    /// Creates a new [`FixedShapeTensorMetadata`] with the given element type and logical `shape`.
    ///
    /// Use [`with_dim_names`](Self::with_dim_names) and
    /// [`with_permutation`](Self::with_permutation) to further configure the metadata.
    pub fn new(element_ptype: PType, shape: Vec<usize>) -> Self {
        Self::from_stored_metadata(element_ptype, FixedShapeTensorStoredMetadata::new(shape))
    }

    /// Combines stored tensor metadata with its storage element type.
    pub fn from_stored_metadata(
        element_ptype: PType,
        stored_metadata: FixedShapeTensorStoredMetadata,
    ) -> Self {
        Self {
            element_ptype,
            stored_metadata,
        }
    }

    /// Returns the element type of the tensor.
    pub fn element_ptype(&self) -> PType {
        self.element_ptype
    }

    /// Sets the dimension names for this tensor.
    ///
    /// An empty list leaves the metadata unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if a non-empty list has a different length from the logical shape.
    pub fn with_dim_names(mut self, names: Vec<String>) -> VortexResult<Self> {
        self.stored_metadata = self.stored_metadata.with_dim_names(names)?;
        Ok(self)
    }

    /// Sets the permutation from logical dimensions to physical dimensions.
    ///
    /// An empty list leaves the metadata unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if a non-empty list is not a permutation of `[0, 1, ..., N-1]`, where `N`
    /// is the number of logical dimensions.
    pub fn with_permutation(mut self, permutation: Vec<usize>) -> VortexResult<Self> {
        self.stored_metadata = self.stored_metadata.with_permutation(permutation)?;
        Ok(self)
    }

    /// Returns the number of dimensions (rank) of the tensor.
    pub fn ndim(&self) -> usize {
        self.stored_metadata.ndim()
    }

    /// Returns the logical dimensions of the tensor as a slice.
    pub fn logical_shape(&self) -> &[usize] {
        self.stored_metadata.logical_shape()
    }

    /// Returns the dimension names, if set.
    pub fn dim_names(&self) -> Option<&[String]> {
        self.stored_metadata.dim_names()
    }

    /// Returns the permutation, if set.
    pub fn permutation(&self) -> Option<&[usize]> {
        self.stored_metadata.permutation()
    }

    /// Returns an iterator over the physical shape of the tensor.
    ///
    /// The physical shape describes the row-major memory layout. It is derived from the logical
    /// shape by placing each logical dimension's size at its physical position:
    /// `physical_shape[permutation[i]] = logical_shape[i]`.
    ///
    /// When no permutation is present, the physical shape is identical to the logical shape.
    pub fn physical_shape(&self) -> impl Iterator<Item = usize> + '_ {
        self.stored_metadata.physical_shape()
    }

    /// Returns an iterator over the strides for each logical dimension of the tensor.
    ///
    /// The stride for a logical dimension is the number of elements to skip in the flat backing
    /// array in order to move one step along that logical dimension.
    ///
    /// When a permutation is present, the physical memory is laid out in row-major order over the
    /// physical dimensions (the logical dimensions reordered by the permutation), so the strides
    /// are derived from that physical layout.
    pub fn strides(&self) -> impl Iterator<Item = usize> + '_ {
        self.stored_metadata.strides()
    }
}

impl fmt::Display for FixedShapeTensorMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(")?;

        match &self.stored_metadata.dim_names {
            Some(names) => {
                for (i, (dim, name)) in self
                    .stored_metadata
                    .logical_shape
                    .iter()
                    .zip(names.iter())
                    .enumerate()
                {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {dim}")?;
                }
            }
            None => {
                for (i, dim) in self.stored_metadata.logical_shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dim}")?;
                }
            }
        }

        if let Some(perm) = &self.stored_metadata.permutation {
            write!(f, ", [")?;
            for (i, p) in perm.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{p}")?;
            }
            write!(f, "]")?;
        }

        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use hegel::TestCase;
    use hegel::generators as gs;

    use super::*;

    /// Reference implementation that computes permuted strides in an explicit, step-by-step way.
    ///
    /// 1. Build the physical shape: `physical_shape[perm[i]] = logical_shape[i]`.
    /// 2. Compute row-major strides over the physical shape.
    /// 3. Map back to logical: `logical_stride[i] = physical_strides[perm[i]]`.
    fn slow_strides(shape: &[usize], perm: &[usize]) -> Vec<usize> {
        let ndim = shape.len();

        // Derive the physical shape from the logical shape and the permutation.
        let mut physical_shape = vec![0usize; ndim];
        for l in 0..ndim {
            physical_shape[perm[l]] = shape[l];
        }

        // Compute row-major strides over the physical shape.
        let mut physical_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            physical_strides[i] = physical_strides[i + 1] * physical_shape[i + 1];
        }

        // Map physical strides back to logical dimension order.
        (0..ndim).map(|l| physical_strides[perm[l]]).collect()
    }

    #[track_caller]
    fn check_layout(shape: Vec<usize>, permutation: Vec<usize>) -> VortexResult<()> {
        let metadata = FixedShapeTensorMetadata::new(PType::F32, shape.clone());
        let identity = (0..shape.len()).collect::<Vec<_>>();
        assert_eq!(metadata.physical_shape().collect::<Vec<_>>(), shape);
        assert_eq!(
            metadata.strides().collect::<Vec<_>>(),
            slow_strides(&shape, &identity)
        );

        let metadata = metadata.with_permutation(permutation.clone())?;
        let mut physical = vec![0; shape.len()];
        for (logical, &axis) in permutation.iter().enumerate() {
            physical[axis] = shape[logical];
        }

        assert_eq!(metadata.physical_shape().collect::<Vec<_>>(), physical);
        assert_eq!(
            metadata.strides().collect::<Vec<_>>(),
            slow_strides(&shape, &permutation)
        );

        Ok(())
    }

    #[hegel::composite]
    fn draw_layout(tc: TestCase) -> (Vec<usize>, Vec<usize>) {
        let shape = tc.draw(gs::vecs(gs::integers::<usize>().max_value(16)).max_size(6));
        let mut remaining: Vec<_> = (0..shape.len()).collect();
        let mut permutation = Vec::with_capacity(shape.len());
        while !remaining.is_empty() {
            let index = tc.draw(gs::integers::<usize>().max_value(remaining.len() - 1));
            permutation.push(remaining.remove(index));
        }

        (shape, permutation)
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    #[hegel::explicit_test_case(layout = (Vec::<usize>::new(), Vec::<usize>::new()))]
    #[hegel::explicit_test_case(layout = (vec![3usize, 0, 4], vec![1usize, 2, 0]))]
    fn layout_matches_reference(tc: TestCase) {
        let layout = tc.draw(draw_layout());
        check_layout(layout.0, layout.1).vortex_expect("tensor layout must match the reference");
    }

    // -- Display --

    #[test]
    fn display_shape_only() {
        let m = FixedShapeTensorMetadata::new(PType::F32, vec![2, 3, 4]);
        assert_eq!(m.to_string(), "Tensor(2, 3, 4)");
    }

    #[test]
    fn display_scalar_0d() {
        let m = FixedShapeTensorMetadata::new(PType::F32, vec![]);
        assert_eq!(m.to_string(), "Tensor()");
    }

    #[test]
    fn display_with_dim_names() -> VortexResult<()> {
        let m = FixedShapeTensorMetadata::new(PType::F32, vec![3, 4])
            .with_dim_names(vec!["rows".into(), "cols".into()])?;
        assert_eq!(m.to_string(), "Tensor(rows: 3, cols: 4)");
        Ok(())
    }

    #[test]
    fn display_with_permutation() -> VortexResult<()> {
        let m = FixedShapeTensorMetadata::new(PType::F32, vec![2, 3, 4])
            .with_permutation(vec![1, 0, 2])?;
        assert_eq!(m.to_string(), "Tensor(2, 3, 4, [1, 0, 2])");
        Ok(())
    }

    #[test]
    fn display_with_dim_names_and_permutation() -> VortexResult<()> {
        let m = FixedShapeTensorMetadata::new(PType::F32, vec![2, 3, 4])
            .with_dim_names(vec!["x".into(), "y".into(), "z".into()])?
            .with_permutation(vec![1, 2, 0])?;
        assert_eq!(m.to_string(), "Tensor(x: 2, y: 3, z: 4, [1, 2, 0])");
        Ok(())
    }

    #[test]
    fn dim_names_wrong_length() {
        let result =
            FixedShapeTensorMetadata::new(PType::F32, vec![2, 3]).with_dim_names(vec!["x".into()]);
        assert!(result.is_err());
    }

    #[test]
    fn permutation_wrong_length() {
        let result =
            FixedShapeTensorMetadata::new(PType::F32, vec![2, 3]).with_permutation(vec![0]);
        assert!(result.is_err());
    }

    #[test]
    fn permutation_out_of_range() {
        let result =
            FixedShapeTensorMetadata::new(PType::F32, vec![2, 3]).with_permutation(vec![0, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn permutation_duplicate_index() {
        let result =
            FixedShapeTensorMetadata::new(PType::F32, vec![2, 3]).with_permutation(vec![0, 0]);
        assert!(result.is_err());
    }
}
