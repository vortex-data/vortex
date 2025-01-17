use std::fmt::Display;

use vortex_dtype::{FieldName, FieldNames};

pub struct DisplayFieldName<'a>(pub &'a FieldName);

impl Display for DisplayFieldName<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.0)
    }
}

pub struct DisplayFieldNames<'a>(pub &'a FieldNames);

impl Display for DisplayFieldNames<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, field) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            DisplayFieldName(field).fmt(f)?
        }
        Ok(())
    }
}
