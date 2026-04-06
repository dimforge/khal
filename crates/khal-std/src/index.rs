/// Indexing trait that optionally removes bounds checks on GPU targets.
///
/// When the `unsafe_remove_boundchecks` feature is enabled, methods use
/// unchecked indexing for performance on SPIR-V and CUDA targets.
/// Otherwise, standard bounds-checked indexing is used.
pub trait MaybeIndexUnchecked<T> {
    /// Returns a reference to the element at `id`.
    fn at(&self, id: usize) -> &T;
    /// Returns a mutable reference to the element at `id`.
    fn at_mut(&mut self, id: usize) -> &mut T;
    /// Copies and returns the element at `id`.
    fn read(&self, id: usize) -> T;
    /// Writes `data` to the element at `id`.
    fn write(&mut self, id: usize, data: T);
}

impl<T: Copy> MaybeIndexUnchecked<T> for [T] {
    #[inline(always)]
    fn at(&self, id: usize) -> &T {
        #[cfg(all(feature = "unsafe_remove_boundchecks", target_arch = "nvptx64"))]
        return unsafe { self.get_unchecked(id) };
        #[cfg(all(feature = "unsafe_remove_boundchecks", not(target_arch = "nvptx64")))]
        return unsafe {
            use spirv_std::arch::IndexUnchecked;
            self.index_unchecked(id)
        };
        #[cfg(not(feature = "unsafe_remove_boundchecks"))]
        return &self[id];
    }

    #[inline(always)]
    fn at_mut(&mut self, id: usize) -> &mut T {
        #[cfg(all(feature = "unsafe_remove_boundchecks", target_arch = "nvptx64"))]
        return unsafe { self.get_unchecked_mut(id) };
        #[cfg(all(feature = "unsafe_remove_boundchecks", not(target_arch = "nvptx64")))]
        return unsafe {
            use spirv_std::arch::IndexUnchecked;
            self.index_unchecked_mut(id)
        };
        #[cfg(not(feature = "unsafe_remove_boundchecks"))]
        return &mut self[id];
    }

    #[inline(always)]
    fn read(&self, id: usize) -> T {
        *self.at(id)
    }

    #[inline(always)]
    fn write(&mut self, id: usize, data: T) {
        *self.at_mut(id) = data;
    }
}

impl<T: Copy, const N: usize> MaybeIndexUnchecked<T> for [T; N] {
    #[inline(always)]
    fn at(&self, id: usize) -> &T {
        #[cfg(all(feature = "unsafe_remove_boundchecks", target_arch = "nvptx64"))]
        return unsafe { self.get_unchecked(id) };
        #[cfg(all(feature = "unsafe_remove_boundchecks", not(target_arch = "nvptx64")))]
        return unsafe {
            use spirv_std::arch::IndexUnchecked;
            self.index_unchecked(id)
        };
        #[cfg(not(feature = "unsafe_remove_boundchecks"))]
        return &self[id];
    }

    #[inline(always)]
    fn at_mut(&mut self, id: usize) -> &mut T {
        #[cfg(all(feature = "unsafe_remove_boundchecks", target_arch = "nvptx64"))]
        return unsafe { self.get_unchecked_mut(id) };
        #[cfg(all(feature = "unsafe_remove_boundchecks", not(target_arch = "nvptx64")))]
        return unsafe {
            use spirv_std::arch::IndexUnchecked;
            self.index_unchecked_mut(id)
        };
        #[cfg(not(feature = "unsafe_remove_boundchecks"))]
        return &mut self[id];
    }

    #[inline(always)]
    fn read(&self, id: usize) -> T {
        *self.at(id)
    }

    #[inline(always)]
    fn write(&mut self, id: usize, data: T) {
        *self.at_mut(id) = data;
    }
}
