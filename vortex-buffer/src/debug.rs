use std::fmt::{Debug, Formatter};

/// A wrapper around a slice that truncates the debug output if it is too long.
pub(crate) struct TruncatedDebug<'a, T>(pub(crate) &'a [T]);

impl<T: Debug> Debug for TruncatedDebug<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        const TRUNC_SIZE: usize = 16;
        if self.0.len() <= TRUNC_SIZE {
            write!(f, "{:?}", self.0)
        } else {
            write!(f, "[")?;
            for elem in self.0.iter().take(TRUNC_SIZE) {
                write!(f, "{:?}, ", *elem)?;
            }
            write!(f, "...")?;
            write!(f, "]")
        }
    }
}
