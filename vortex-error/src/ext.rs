use crate::VortexResult;

/// Extension trait for VortexResult
pub trait ResultExt<T>: private::Sealed {
    /// Unnest a nested [`VortexResult`]. Helper function until <https://github.com/rust-lang/rust/issues/70142> is stabilized.
    fn unnest(self) -> VortexResult<T>;
}

mod private {
    use crate::VortexResult;

    pub trait Sealed {}

    impl<T> Sealed for VortexResult<VortexResult<T>> {}
}

impl<T> ResultExt<T> for VortexResult<VortexResult<T>> {
    fn unnest(self) -> VortexResult<T> {
        match self {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) | Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_example() {
        let r: VortexResult<VortexResult<usize>> = Ok(Ok(5_usize));

        // Only need to unwrap once!
        assert_eq!(5, r.unnest().unwrap());
    }
}
