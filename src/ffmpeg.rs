use std::{mem::ManuallyDrop, path::Path};

use ffmpeg::{
    codec::decoder,
    format,
    format::context::input::PacketIter,
    frame,
    media::Type,
    threading,
    threading::Config,
};
use rav1e::prelude::{
    ChromaSamplePosition,
    ChromaSampling,
    Frame,
    Pixel,
    Plane,
    PlaneConfig,
    PlaneData,
    Rational,
};

use crate::decode::{Decoder, FrameView, VideoDetails};

/// An interface that is used for decoding a video stream using FFmpeg
pub struct FfmpegDecoder<'a> {
    packet_iter: PacketIter<'a>,
    decoder: decoder::Video,
    video_details: VideoDetails,
    stream_index: usize,
    receiving_eof_frames: bool,
    receiving_frames: bool,
    format: format::Pixel,
}

impl<'a> FfmpegDecoder<'a> {
    /// Get decoding context used to initialize the decoder
    ///
    /// This step is needed separately to avoid lifetime issues
    /// regarding a field in a struct containing a reference to
    /// another field in the same struct.
    pub fn get_ctx(input: &Path) -> Result<format::context::Input, ffmpeg::Error> {
        format::input(&input)
    }

    // TODO Don't use strings for error handling.
    /// Initialize a new FFmpeg decoder for a given input file
    pub fn new(input_ctx: &'a mut format::context::Input) -> Result<Self, String> {
        ffmpeg::init().map_err(|e| e.to_string())?;

        let input = input_ctx
            .streams()
            .best(Type::Video)
            .ok_or_else(|| "Could not find video stream".to_string())?;
        let stream_index = input.index();
        let mut decoder_context =
            ffmpeg::codec::context::Context::from_parameters(input.parameters())
                .map_err(|e| e.to_string())?;

        // This needs to be done on the decoder context BEFORE
        // creating the decoder itself, otherwise the decoder
        // will still be single-threaded.
        decoder_context.set_threading(Config {
            // 0 = decoder automatically decides number of threads to use
            count: 0,
            // TODO set threading kind based on supported features of codec
            kind: threading::Type::Frame,
            safe: false,
        });

        let mut decoder = decoder_context
            .decoder()
            .video()
            .map_err(|e| e.to_string())?;

        decoder
            .set_parameters(input.parameters())
            .map_err(|e| e.to_string())?;

        let format = decoder.format();

        let frame_rate = input.avg_frame_rate();
        Ok(Self {
            video_details: VideoDetails {
                width: decoder.width() as usize,
                height: decoder.height() as usize,
                bit_depth: match decoder.format() {
                    format::pixel::Pixel::YUV420P
                    | format::pixel::Pixel::YUV422P
                    | format::pixel::Pixel::YUV444P => 8,
                    format::pixel::Pixel::YUV420P10LE
                    | format::pixel::Pixel::YUV422P10LE
                    | format::pixel::Pixel::YUV444P10LE => 10,
                    format::pixel::Pixel::YUV420P12LE
                    | format::pixel::Pixel::YUV422P12LE
                    | format::pixel::Pixel::YUV444P12LE => 12,
                    _ => {
                        return Err(format!("Unsupported pixel format {:?}", decoder.format()));
                    }
                },
                chroma_sampling: match decoder.format() {
                    format::pixel::Pixel::YUV420P
                    | format::pixel::Pixel::YUV420P10LE
                    | format::pixel::Pixel::YUV420P12LE => ChromaSampling::Cs420,
                    format::pixel::Pixel::YUV422P
                    | format::pixel::Pixel::YUV422P10LE
                    | format::pixel::Pixel::YUV422P12LE => ChromaSampling::Cs422,
                    format::pixel::Pixel::YUV444P
                    | format::pixel::Pixel::YUV444P10LE
                    | format::pixel::Pixel::YUV444P12LE => ChromaSampling::Cs444,
                    _ => {
                        return Err(format!("Unsupported pixel format {:?}", decoder.format()));
                    }
                },
                chroma_sample_position: match decoder.format() {
                    format::pixel::Pixel::YUV422P
                    | format::pixel::Pixel::YUV422P10LE
                    | format::pixel::Pixel::YUV422P12LE => ChromaSamplePosition::Vertical,
                    _ => ChromaSamplePosition::Colocated,
                },
                time_base: Rational::new(
                    frame_rate.denominator() as u64,
                    frame_rate.numerator() as u64,
                ),
                luma_padding: 0,
            },
            decoder,
            packet_iter: input_ctx.packets(),
            stream_index,
            receiving_eof_frames: false,
            receiving_frames: false,
            format,
        })
    }

    pub fn receive_frame_init<T: Pixel>(
        &mut self,
        stride: usize,
        alloc_height: usize,
    ) -> Option<frame::Video> {
        let mut frame = frame::Video::new(self.format, stride as u32, alloc_height as u32);

        if self.receive_frame::<T>(&mut frame) {
            Some(frame)
        } else {
            None
        }
    }

    /// Same as [`read_video_frame`] but does not create an additional allocation
    pub fn receive_frame<T: Pixel>(&mut self, alloc: &mut frame::Video) -> bool {
        // Get packet until we find one that is the index we need

        // Only return false after packet_iter stops returning and after we've returned
        // all frames after eof

        // Only return false when packet_iter stops returning, otherwise keep decoding
        // We have this loop ONLY IF receive_frame doesn't return true, we have to keep
        // on asking until it does
        loop {
            // Are we receiving a frame
            // We keep receiving frames while the decoder is returning true for receive_frame
            if self.receiving_frames {
                let res = self.decoder.receive_frame(alloc).is_ok();
                if res {
                    // Keep receiving frames
                    return true;
                } else {
                    // We need to send more packets once the decoder stops returning
                    self.receiving_frames = false;
                }
            } else if self.receiving_eof_frames {
                return self.decoder.receive_frame(alloc).is_ok();
            }

            if let Some((stream, packet)) = self.packet_iter.next() {
                // Skip irrelevant indexes
                if stream.index() != self.stream_index {
                    continue;
                }

                self.decoder.send_packet(&packet).unwrap();
                self.receiving_frames = true;
                continue;
            } else {
                self.decoder.send_eof().unwrap();
                self.receiving_eof_frames = true;
                continue;
            }
        }
    }
}

// TODO these trait methods shouldn't be public,
// only the traits themselves
impl<'a> Decoder<frame::Video> for FfmpegDecoder<'a> {
    unsafe fn get_frame_ref<T: Pixel>(
        frame: &frame::Video,
        height: usize,
        width: usize,
        stride: usize,
        alloc_height: usize,
        _strict: bool,
        _alloc: Option<&mut Plane<T>>,
    ) -> FrameView<T> {
        let empty_plane = || Plane::<T> {
            cfg: PlaneConfig {
                alloc_height: 0,
                height: 0,
                stride: 0,
                width: 0,
                xdec: 0,
                xorigin: 0,
                xpad: 0,
                ydec: 0,
                yorigin: 0,
                ypad: 0,
            },
            data: PlaneData::new_ref(&[]),
        };

        let plane_cfg_luma: PlaneConfig = PlaneConfig {
            alloc_height,
            height,
            stride,
            width,
            xdec: 0,
            xorigin: 0,
            xpad: 0,
            ydec: 0,
            yorigin: 0,
            ypad: 0,
        };

        // Dimensions are always as requested
        FrameView::Ref(ManuallyDrop::new(Frame::<T> {
            planes: [
                {
                    Plane::<T> {
                        cfg: plane_cfg_luma,
                        data: PlaneData::new_ref(std::slice::from_raw_parts(
                            frame.data(0).as_ptr().cast(),
                            stride * height,
                        )),
                    }
                },
                empty_plane(),
                empty_plane(),
            ],
        }))
    }

    /// This function should not be called.
    fn make_copy<T: Pixel>(
        _frame: &frame::Video,
        height: usize,
        width: usize,
        stride: usize,
        alloc_height: usize,
        _alloc: Option<&mut Plane<T>>,
    ) -> Option<Plane<T>> {
        None
    }

    fn stride_matches<T: Pixel>(&mut self, _stride: usize, _alloc_height: usize) -> bool {
        // Stride always matches
        true
    }

    fn get_video_details(&self) -> VideoDetails {
        self.video_details
    }

    fn receive_frame<T: Pixel>(&mut self, alloc: &mut frame::Video) -> bool {
        self.receive_frame::<T>(alloc)
    }

    fn receive_frame_init<T: Pixel>(
        &mut self,
        stride: usize,
        alloc_height: usize,
    ) -> Option<frame::Video> {
        self.receive_frame_init::<T>(stride, alloc_height)
    }

    fn get_bit_depth(&self) -> usize {
        self.video_details.bit_depth
    }
}
