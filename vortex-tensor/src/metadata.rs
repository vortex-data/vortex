// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use itertools::Either;

/// Metadata for a [`Tensor`] extension type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FixedShapeTensorMetadata {
    /// The shape of the tensor.
    ///
    /// The shape is always defined over row-major storage. May be empty (0D scalar tensor) or
    /// contain dimensions of size 0 (degenerate tensor).
    shape: Vec<usize>,

    /// Optional names for each dimension. Each name corresponds to a dimension in the `shape`.
    ///
    /// If names exist, there must be an equal number of names to dimensions.
    dim_names: Option<Vec<String>>,

    /// The permutation of the tensor's dimensions, mapping each logical dimension to its
    /// corresponding physical dimension: `permutation[logical] = physical`.
    ///
    /// If this is `None`, then the logical and physical layout are equal, and the permutation is
    /// in-order `[0, 1, ..., N-1]`.
    permutation: Option<Vec<usize>>,
}

impl FixedShapeTensorMetadata {
    /// Creates a new [`FixedShapeTensorMetadata`] with the given `shape`.
    ///
    /// The shape defines the logical dimensions in row-major order. Use
    /// [`with_dim_names`][Self::with_dim_names] and [`with_permutation`][Self::with_permutation]
    /// to further configure the metadata.
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
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

    /// Returns the dimensions of the tensor as a slice.
    pub fn dimensions(&self) -> &[usize] {
        &self.shape
    }

    /// Returns an iterator over the strides for each logical dimension of the tensor.
    ///
    /// The stride for each dimension is the number of elements to skip in the flat backing array
    /// in order to move one step along that dimension.
    ///
    /// When a permutation is present, the physical memory is laid out in row-major order over the
    /// permuted dimensions, so the strides are computed accordingly.
    pub fn strides(&self) -> impl Iterator<Item = usize> + '_ {
        let ndim = self.shape.len();
        let permutation = self.permutation.as_deref();

        match permutation {
            None => Either::Left((0..ndim).map(|i| self.shape[i + 1..].iter().product::<usize>())),
            Some(permutation) => {
                Either::Right((0..ndim).map(|i| self.permuted_stride(i, permutation)))
            }
        }
    }

    /// Computes the stride for logical dimension `i` given a `permutation`.
    ///
    /// The stride is the product of `shape[l]` for all logical dimensions `l` whose physical
    /// position comes after `perm[i]`.
    fn permuted_stride(&self, i: usize, perm: &[usize]) -> usize {
        let phys = perm[i];

        // Note that this is O(n^2), but since the number of dimensions is likely low and doing this
        // avoids allocations, this is usually much faster.
        perm.iter()
            .enumerate()
            .filter(|&(_, &p)| p > phys)
            .map(|(l, _)| self.shape[l])
            .product::<usize>()
    }
}

impl fmt::Display for FixedShapeTensorMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.dim_names {
            Some(names) => {
                write!(f, "Tensor(")?;
                for (i, (dim, name)) in self.shape.iter().zip(names.iter()).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {dim}")?;
                }
                write!(f, ")")
            }
            None => {
                write!(f, "Tensor(")?;
                for (i, dim) in self.shape.iter().enumerate() {
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

    #[test]
    fn strides_1d() {
        let m = FixedShapeTensorMetadata::new(vec![5]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn strides_2d_row_major() {
        // Logical shape [3, 4], no permutation.
        // Physical shape is [3, 4], physical strides are [4, 1].
        let m = FixedShapeTensorMetadata::new(vec![3, 4]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![4, 1]);
    }

    #[test]
    fn strides_3d_row_major() {
        // Logical shape [2, 3, 4], no permutation.
        // Physical shape is [2, 3, 4], physical strides are [12, 4, 1].
        let m = FixedShapeTensorMetadata::new(vec![2, 3, 4]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![12, 4, 1]);
    }

    #[test]
    fn strides_2d_transposed() {
        // Logical shape [3, 4], permutation [1, 0].
        // perm[logical] = physical: logical 0 -> phys 1, logical 1 -> phys 0.
        // Physical shape (arrange logical sizes by phys pos):
        //   phys 0 <- logical 1 (size 4)
        //   phys 1 <- logical 0 (size 3)
        //   => physical shape [4, 3], physical strides [3, 1].
        // Logical strides = phys_stride[perm[i]]:
        //   logical 0 -> phys 1 -> stride 1
        //   logical 1 -> phys 0 -> stride 3
        let m = FixedShapeTensorMetadata::new(vec![3, 4]).with_permutation(vec![1, 0]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn strides_3d_permuted() {
        // Logical shape [2, 3, 4], permutation [2, 0, 1].
        // perm[logical] = physical: logical 0 -> phys 2, logical 1 -> phys 0, logical 2 -> phys 1.
        // Physical shape (arrange logical sizes by phys pos):
        //   phys 0 <- logical 1 (size 3)
        //   phys 1 <- logical 2 (size 4)
        //   phys 2 <- logical 0 (size 2)
        //   => physical shape [3, 4, 2], physical strides [8, 2, 1].
        // Logical strides = phys_stride[perm[i]]:
        //   logical 0 -> phys 2 -> stride 1
        //   logical 1 -> phys 0 -> stride 8
        //   logical 2 -> phys 1 -> stride 2
        let m = FixedShapeTensorMetadata::new(vec![2, 3, 4]).with_permutation(vec![2, 0, 1]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![1, 8, 2]);
    }

    #[test]
    fn strides_0d_scalar() {
        // A 0D tensor (scalar) has no dimensions and thus no strides.
        // numel = 1 (empty product), not 0.
        let m = FixedShapeTensorMetadata::new(vec![]);
        assert_eq!(m.strides().collect::<Vec<_>>(), Vec::<usize>::new());
    }

    #[test]
    fn strides_zero_size_dimension() {
        // Logical shape [3, 0, 4], no permutation. numel = 0.
        // Physical strides are still well-defined products of trailing dimensions.
        let m = FixedShapeTensorMetadata::new(vec![3, 0, 4]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![0, 4, 1]);
    }

    #[test]
    fn strides_zero_size_dimension_permuted() {
        // Logical shape [3, 0, 4], permutation [1, 2, 0].
        // perm[logical] = physical: logical 0 -> phys 1, logical 1 -> phys 2, logical 2 -> phys 0.
        // Physical shape (arrange logical sizes by phys pos):
        //   phys 0 <- logical 2 (size 4)
        //   phys 1 <- logical 0 (size 3)
        //   phys 2 <- logical 1 (size 0)
        //   => physical shape [4, 3, 0], physical strides [0, 0, 1].
        // Logical strides = phys_stride[perm[i]]:
        //   logical 0 -> phys 1 -> stride 0
        //   logical 1 -> phys 2 -> stride 1
        //   logical 2 -> phys 0 -> stride 0
        let m = FixedShapeTensorMetadata::new(vec![3, 0, 4]).with_permutation(vec![1, 2, 0]);
        assert_eq!(m.strides().collect::<Vec<_>>(), vec![0, 1, 0]);
    }

    #[test]
    fn strides_identity_permutation_matches_row_major() {
        // An identity permutation should produce the same strides as no permutation.
        let row_major = FixedShapeTensorMetadata::new(vec![2, 3, 4]);
        let identity = FixedShapeTensorMetadata::new(vec![2, 3, 4]).with_permutation(vec![0, 1, 2]);
        assert_eq!(
            row_major.strides().collect::<Vec<_>>(),
            identity.strides().collect::<Vec<_>>(),
        );
    }
}
