use pco::errors::{ErrorKind, PcoError};
use vortex_error::{VortexError, vortex_err};

impl From<PcoError> for VortexError {
    fn from(value: PcoError) -> Self {
        match value.kind {
            ErrorKind::Io(io_error_kind) => {
                VortexError::IOError(std::io::Error::new(io_error_kind, value.message))
            }
            _ => vortex_err!(format!("Pco {:?} error: {}", kind, value.message)),
        }
    }
}
