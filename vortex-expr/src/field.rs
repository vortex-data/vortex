use std::fmt::Display;

use vortex_dtype::FieldNames;

pub struct DisplayFieldNames<'a>(pub &'a FieldNames);

impl Display for DisplayFieldNames<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, field) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            field.fmt(f)?
        }
        Ok(())
    }
}
