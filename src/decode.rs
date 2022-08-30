use std::mem::ManuallyDrop;

use rav1e::prelude::{ChromaSamplePosition, ChromaSampling, Frame, Pixel, Rational};

#[derive(Debug, Clone, Copy)]
pub struct VideoDetails {
    pub width: usize,
    pub height: usize,
    pub bit_depth: usize,
    pub chroma_sampling: ChromaSampling,
    pub chroma_sample_position: ChromaSamplePosition,
    pub time_base: Rational,
    pub luma_padding: usize,
}

pub enum Frame2<T: Pixel> {
    /// Don't call the destructor on this frame.
    Ref(ManuallyDrop<Frame<T>>),
    /// Owned frame.
    Owned(Frame<T>),
}

pub trait Decoder2<InternalFrame> {
    // TODO maybe change order of arguments
    unsafe fn get_frame_ref<T: Pixel>(
        frame: &InternalFrame,
        height: usize,
        width: usize,
        stride: usize,
        alloc_height: usize,
        strict: bool,
    ) -> Frame2<T>;

    fn receive_frame_init<T: Pixel>(
        &mut self,
        stride: usize,
        alloc_height: usize,
    ) -> Option<InternalFrame>;

    fn receive_frame<T: Pixel>(&mut self, alloc: &mut InternalFrame) -> bool;

    /// Get the Video Details
    fn get_video_details(&self) -> VideoDetails;

    /// Get the Bit Depth
    fn get_bit_depth(&self) -> usize;
}
