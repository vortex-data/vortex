// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use itertools::Either;
use vortex_error::VortexExpect;

/// Metadata for a `FixedShapeTensor` extension type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FixedShapeTensorMetadata {
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

impl FixedShapeTensorMetadata {
    /// Creates a new [`FixedShapeTensorMetadata`] with the given logical `shape`.
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
    pub fn with_dim_names(mut self, names: Vec<String>) -> Self {
        self.dim_names = Some(names);
        self
    }

    /// Sets the permutation for this tensor.
    pub fn with_permutation(mut self, permutation: Vec<usize>) -> Self {
        self.permutation = Some(permutation);
        self
    }

    /// Returns the number of dimensions (rank) of the tensor.
    pub fn ndim(&self) -> usize {
        self.logical_shape.len()
    }

    /// Returns the logical dimensions of the tensor as a slice.
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
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
        // is typically small (2–5), so avoiding a Vec allocation is a net win.
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

impl fmt::Display for FixedShapeTensorMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.dim_names {
            Some(names) => {
                write!(f, "Tensor(")?;
                for (i, (dim, name)) in self.logical_shape.iter().zip(names.iter()).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {dim}")?;
                }
                write!(f, ")")
            }
            None => {
                write!(f, "Tensor(")?;
                for (i, dim) in self.logical_shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dim}")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn strides_1d() {
        let m = FixedShapeTensorMetadata::new(vec![5]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn strides_2d_row_major() {
        let m = FixedShapeTensorMetadata::new(vec![3, 4]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![4, 1]);
    }

    #[test]
    fn strides_3d_row_major() {
        let m = FixedShapeTensorMetadata::new(vec![2, 3, 4]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![12, 4, 1]);
    }

    #[test]
    fn strides_2d_transposed() {
        // Logical shape [3, 4] with perm [1, 0] (transpose).
        // Physical shape = [4, 3], so logical strides = [1, 3].
        let m = FixedShapeTensorMetadata::new(vec![3, 4]).with_permutation(vec![1, 0]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn strides_3d_permuted() {
        // Logical shape [2, 3, 4] with perm [2, 0, 1].
        // Physical shape = [3, 4, 2], so logical strides = [1, 8, 2].
        let m = FixedShapeTensorMetadata::new(vec![2, 3, 4]).with_permutation(vec![2, 0, 1]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![1, 8, 2]);
    }

    #[test]
    fn strides_0d_scalar() {
        let m = FixedShapeTensorMetadata::new(vec![]);
        assert_eq!(m.strides().collect::<Vec<_>>(), Vec::<usize>::new());
    }

    #[test]
    fn strides_zero_size_dimension() {
        let m = FixedShapeTensorMetadata::new(vec![3, 0, 4]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![0, 4, 1]);
    }

    #[test]
    fn strides_zero_size_dimension_permuted() {
        // Logical shape [3, 0, 4] with perm [1, 2, 0].
        // Physical shape = [4, 3, 0], so logical strides = [0, 1, 0].
        let m = FixedShapeTensorMetadata::new(vec![3, 0, 4]).with_permutation(vec![1, 2, 0]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![0, 1, 0]);
    }

    #[test]
    fn strides_identity_permutation_matches_row_major() {
        let row_major = FixedShapeTensorMetadata::new(vec![2, 3, 4]);
        let identity = FixedShapeTensorMetadata::new(vec![2, 3, 4]).with_permutation(vec![0, 1, 2]);
        assert_eq!(
            row_major.strides().collect::<Vec<_>>(),
            identity.strides().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn physical_shape_no_permutation() {
        let m = FixedShapeTensorMetadata::new(vec![2, 3, 4]);
        assert_eq!(m.physical_shape().collect::<Vec<_>>(), vec![2, 3, 4]);
    }

    #[test]
    fn physical_shape_2d_transposed() {
        // Logical [3, 4] with perm [1, 0]: physical dim 0 gets logical dim 1's size (4),
        // physical dim 1 gets logical dim 0's size (3).
        let m = FixedShapeTensorMetadata::new(vec![3, 4]).with_permutation(vec![1, 0]);
        assert_eq!(m.physical_shape().collect::<Vec<_>>(), vec![4, 3]);
    }

    #[test]
    fn physical_shape_3d_permuted() {
        // Logical [2, 3, 4] with perm [2, 0, 1]: logical 0 -> phys 2, logical 1 -> phys 0,
        // logical 2 -> phys 1. So physical shape = [3, 4, 2].
        let m = FixedShapeTensorMetadata::new(vec![2, 3, 4]).with_permutation(vec![2, 0, 1]);
        assert_eq!(m.physical_shape().collect::<Vec<_>>(), vec![3, 4, 2]);
    }

    #[test]
    fn physical_shape_identity_permutation() {
        let no_perm = FixedShapeTensorMetadata::new(vec![2, 3, 4]);
        let identity = FixedShapeTensorMetadata::new(vec![2, 3, 4]).with_permutation(vec![0, 1, 2]);
        assert_eq!(
            no_perm.physical_shape().collect::<Vec<_>>(),
            identity.physical_shape().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn physical_shape_zero_size_dimension() {
        // Logical [3, 0, 4] with perm [1, 2, 0]: physical shape = [4, 3, 0].
        let m = FixedShapeTensorMetadata::new(vec![3, 0, 4]).with_permutation(vec![1, 2, 0]);
        assert_eq!(m.physical_shape().collect::<Vec<_>>(), vec![4, 3, 0]);
    }

    /// Verifies that the fast `permuted_stride` matches the explicit reference `slow_strides`
    /// across a variety of shapes and permutations.
    #[test]
    fn fast_strides_match_slow_reference() {
        let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
            // 2D transpose.
            (vec![3, 4], vec![1, 0]),
            // 3D permutations.
            (vec![2, 3, 4], vec![2, 0, 1]),
            (vec![2, 3, 4], vec![1, 2, 0]),
            (vec![2, 3, 4], vec![0, 2, 1]),
            // 3D identity.
            (vec![2, 3, 4], vec![0, 1, 2]),
            // 3D with a zero-sized dimension.
            (vec![3, 0, 4], vec![1, 2, 0]),
            (vec![0, 3, 4], vec![2, 0, 1]),
            // 4D permutations.
            (vec![2, 3, 4, 5], vec![3, 2, 1, 0]),
            (vec![2, 3, 4, 5], vec![1, 0, 3, 2]),
            (vec![2, 3, 4, 5], vec![2, 3, 0, 1]),
        ];

        for (shape, perm) in &cases {
            let m = FixedShapeTensorMetadata::new(shape.clone()).with_permutation(perm.clone());
            let fast: Vec<usize> = m.strides().collect();
            let slow = slow_strides(shape, perm);
            assert_eq!(fast, slow, "mismatch for shape={shape:?}, perm={perm:?}");
        }
    }
}
