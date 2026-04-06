#[cfg(target_arch = "nvptx64")]
pub use memory_nvptx::*;
#[cfg(not(target_arch = "nvptx64"))]
pub use spirv_std::memory::*;

// On nvptx64, provide the memory scope constants.
#[cfg(target_arch = "nvptx64")]
pub mod memory_nvptx {
    /// Memory scope levels matching SPIR-V semantics for use on the CUDA backend.
    #[derive(Copy, Clone)]
    #[repr(u32)]
    pub enum Scope {
        /// Visible across all devices.
        CrossDevice = 0,
        /// Visible within the current device.
        Device = 1,
        /// Visible within the current workgroup.
        Workgroup = 2,
        /// Visible within the current subgroup.
        Subgroup = 3,
        /// Visible only to the current invocation.
        Invocation = 4,
        /// Visible within the queue family.
        QueueFamily = 5,
    }
    /// Memory ordering semantics matching SPIR-V for use on the CUDA backend.
    #[derive(Copy, Clone)]
    pub struct Semantics(u32);
    impl Semantics {
        /// No memory ordering constraints.
        pub const NONE: Self = Self(0);
        /// Acquire semantics.
        pub const ACQUIRE: Self = Self(0x2);
        /// Release semantics.
        pub const RELEASE: Self = Self(0x4);
        /// Acquire-release semantics.
        pub const ACQUIRE_RELEASE: Self = Self(0x8);
        /// Applies to uniform (constant) memory.
        pub const UNIFORM_MEMORY: Self = Self(0x40);
        /// Applies to workgroup (shared) memory.
        pub const WORKGROUP_MEMORY: Self = Self(0x100);
        /// Returns the raw bit representation.
        pub const fn bits(self) -> u32 {
            self.0
        }
    }

    impl core::ops::BitOr for Semantics {
        type Output = Self;
        fn bitor(self, rhs: Self) -> Self {
            Self(self.0 | rhs.0)
        }
    }
}
