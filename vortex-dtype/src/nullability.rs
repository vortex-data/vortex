use std::fmt::{Display, Formatter};

/// Whether an instance of a DType can be `null or not
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Nullability {
    /// Instances of this DType are guaranteed to be non-nullable
    #[default]
    NonNullable,
    /// Instances of this DType may contain a null value
    Nullable,
}

impl From<bool> for Nullability {
    fn from(value: bool) -> Self {
        if value {
            Self::Nullable
        } else {
            Self::NonNullable
        }
    }
}

impl From<Nullability> for bool {
    fn from(value: Nullability) -> Self {
        match value {
            Nullability::NonNullable => false,
            Nullability::Nullable => true,
        }
    }
}

impl Display for Nullability {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonNullable => write!(f, ""),
            Self::Nullable => write!(f, "?"),
        }
    }
}
