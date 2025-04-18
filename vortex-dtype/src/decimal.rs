#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DecimalDType {
    pub precision: u8,
    pub scale: u8,
}
