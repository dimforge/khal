use core::ops::Range;

/// A custom replacement for `(a..b).step_by(c)`.
/// This is useful as the iterator combinator adds some early-exists that break uniform control
/// flow; this is incompatible with using barriers when targetting WebGpu on the browser.
pub struct StepRng {
    /// Current position of the iterator.
    pub from: u32,
    /// Exclusive upper bound.
    pub end: u32,
    /// Step size between iterations.
    pub step: u32,
}

impl StepRng {
    /// Creates a new stepped range iterator over `rng` with the given `step` size.
    pub fn new(rng: Range<u32>, step: u32) -> Self {
        Self {
            from: rng.start,
            end: rng.end,
            step,
        }
    }
}

impl Iterator for StepRng {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.from;
        if result >= self.end {
            None
        } else {
            self.from += self.step;
            Some(result)
        }
    }
}
