// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use itertools::Either;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

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

    /// Sets the dimension names for this tensor. An empty vec is normalized to `None` since a
    /// 0-dimensional tensor has no dimensions to name.
    ///
    /// The number of names must match the number of logical dimensions.
    pub fn with_dim_names(mut self, names: Vec<String>) -> VortexResult<Self> {
        if !names.is_empty() {
            vortex_ensure_eq!(
                names.len(),
                self.logical_shape.len(),
                "dim_names length ({}) must match logical_shape length ({})",
                names.len(),
                self.logical_shape.len()
            );
            self.dim_names = Some(names);
        }

        Ok(self)
    }

    /// Sets the permutation for this tensor. An empty vec is normalized to `None` since a
    /// 0-dimensional tensor has no dimensions to permute.
    ///
    /// The permutation must be a valid permutation of `[0, 1, ..., N-1]` where `N` is the
    /// number of logical dimensions.
    pub fn with_permutation(mut self, permutation: Vec<usize>) -> VortexResult<Self> {
        if !permutation.is_empty() {
            vortex_ensure_eq!(
                permutation.len(),
                self.logical_shape.len(),
                "permutation length ({}) must match logical_shape length ({})",
                permutation.len(),
                self.logical_shape.len()
            );

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

impl fmt::Display for FixedShapeTensorMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(")?;

        match &self.dim_names {
            Some(names) => {
                for (i, (dim, name)) in self.logical_shape.iter().zip(names.iter()).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {dim}")?;
                }
            }
            None => {
                for (i, dim) in self.logical_shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dim}")?;
                }
            }
        }

        if let Some(perm) = &self.permutation {
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
    use rstest::rstest;

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

    // -- Row-major strides (no permutation) --

    #[rstest]
    #[case::scalar_0d(vec![],        vec![])]
    #[case::vector_1d(vec![5],       vec![1])]
    #[case::matrix_2d(vec![3, 4],    vec![4, 1])]
    #[case::tensor_3d(vec![2, 3, 4], vec![12, 4, 1])]
    #[case::zero_dim( vec![3, 0, 4], vec![0, 4, 1])]
    fn strides_row_major(#[case] shape: Vec<usize>, #[case] expected: Vec<usize>) {
        let m = FixedShapeTensorMetadata::new(shape);
        assert_eq!(m.strides().collect::<Vec<_>>(), expected);
    }

    // -- Permuted strides --
    //
    // Each case is checked against the expected value and cross-validated against the
    // `slow_strides` reference implementation.

    #[rstest]
    // 2D transpose: physical shape = [4, 3].
    #[case::transpose_2d(vec![3, 4],    vec![1, 0],    vec![1, 3])]
    // 3D: physical shape = [3, 4, 2].
    #[case::perm_3d_201( vec![2, 3, 4], vec![2, 0, 1], vec![1, 8, 2])]
    // 3D with zero-sized dimension: physical shape = [4, 3, 0].
    #[case::zero_dim(    vec![3, 0, 4], vec![1, 2, 0], vec![0, 1, 0])]
    fn strides_permuted(
        #[case] shape: Vec<usize>,
        #[case] perm: Vec<usize>,
        #[case] expected: Vec<usize>,
    ) -> VortexResult<()> {
        let m = FixedShapeTensorMetadata::new(shape.clone()).with_permutation(perm.clone())?;
        let actual: Vec<usize> = m.strides().collect();
        assert_eq!(actual, expected);
        assert_eq!(actual, slow_strides(&shape, &perm));
        Ok(())
    }

    #[test]
    fn strides_identity_permutation_matches_row_major() -> VortexResult<()> {
        let row_major = FixedShapeTensorMetadata::new(vec![2, 3, 4]);
        let identity =
            FixedShapeTensorMetadata::new(vec![2, 3, 4]).with_permutation(vec![0, 1, 2])?;
        assert_eq!(
            row_major.strides().collect::<Vec<_>>(),
            identity.strides().collect::<Vec<_>>(),
        );
        Ok(())
    }

    /// Cross-validates the fast `permuted_stride` against the reference `slow_strides` across a
    /// broader set of shapes and permutations.
    #[rstest]
    #[case::perm_3d_120(vec![2, 3, 4],    vec![1, 2, 0])]
    #[case::perm_3d_021(vec![2, 3, 4],    vec![0, 2, 1])]
    #[case::identity_3d(vec![2, 3, 4],    vec![0, 1, 2])]
    #[case::zero_lead(  vec![0, 3, 4],    vec![2, 0, 1])]
    #[case::rev_4d(     vec![2, 3, 4, 5], vec![3, 2, 1, 0])]
    #[case::swap_4d(    vec![2, 3, 4, 5], vec![1, 0, 3, 2])]
    #[case::half_4d(    vec![2, 3, 4, 5], vec![2, 3, 0, 1])]
    fn strides_match_slow_reference(
        #[case] shape: Vec<usize>,
        #[case] perm: Vec<usize>,
    ) -> VortexResult<()> {
        let m = FixedShapeTensorMetadata::new(shape.clone()).with_permutation(perm.clone())?;
        assert_eq!(m.strides().collect::<Vec<_>>(), slow_strides(&shape, &perm));
        Ok(())
    }

    // -- Physical shape --

    #[test]
    fn physical_shape_no_permutation() {
        let m = FixedShapeTensorMetadata::new(vec![2, 3, 4]);
        assert_eq!(m.physical_shape().collect::<Vec<_>>(), vec![2, 3, 4]);
    }

    #[rstest]
    // Logical [3, 4] with perm [1, 0] → physical [4, 3].
    #[case::transpose_2d(vec![3, 4],    vec![1, 0],    vec![4, 3])]
    // Logical [2, 3, 4] with perm [2, 0, 1] → physical [3, 4, 2].
    #[case::perm_3d(     vec![2, 3, 4], vec![2, 0, 1], vec![3, 4, 2])]
    // Identity: physical = logical.
    #[case::identity(    vec![2, 3, 4], vec![0, 1, 2], vec![2, 3, 4])]
    // Logical [3, 0, 4] with perm [1, 2, 0] → physical [4, 3, 0].
    #[case::zero_dim(    vec![3, 0, 4], vec![1, 2, 0], vec![4, 3, 0])]
    fn physical_shape_permuted(
        #[case] shape: Vec<usize>,
        #[case] perm: Vec<usize>,
        #[case] expected: Vec<usize>,
    ) -> VortexResult<()> {
        let m = FixedShapeTensorMetadata::new(shape).with_permutation(perm)?;
        assert_eq!(m.physical_shape().collect::<Vec<_>>(), expected);
        Ok(())
    }

    #[test]
    fn dim_names_wrong_length() {
        let result = FixedShapeTensorMetadata::new(vec![2, 3]).with_dim_names(vec!["x".into()]);
        assert!(result.is_err());
    }

    #[test]
    fn permutation_wrong_length() {
        let result = FixedShapeTensorMetadata::new(vec![2, 3]).with_permutation(vec![0]);
        assert!(result.is_err());
    }

    #[test]
    fn permutation_out_of_range() {
        let result = FixedShapeTensorMetadata::new(vec![2, 3]).with_permutation(vec![0, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn permutation_duplicate_index() {
        let result = FixedShapeTensorMetadata::new(vec![2, 3]).with_permutation(vec![0, 0]);
        assert!(result.is_err());
    }
}
