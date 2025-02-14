use crate::VortexResult;

/// Extension trait for VortexResult
pub trait ResultExt<T>: private::Sealed {
    /// Flatten a nested [`VortexResult`]. Helper function until <https://github.com/rust-lang/rust/issues/70142> is stabilized.
    fn flatten(self) -> VortexResult<T>;
}

mod private {
    use crate::VortexResult;

    pub trait Sealed {}

    impl<T> Sealed for VortexResult<VortexResult<T>> {}
}

impl<T> ResultExt<T> for VortexResult<VortexResult<T>> {
    fn flatten(self) -> VortexResult<T> {
        match self {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) | Err(e) => Err(e),
        }
    }
}
