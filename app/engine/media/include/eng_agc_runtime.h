#ifndef ENG_AGC_RUNTIME_H
#define ENG_AGC_RUNTIME_H

#include "sce/sce_agc.h"
#include "eng_agc_writer.h"
#include "eng_agc_transient_ring.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ENG_AGC_PIPE_UI = 0,
    ENG_AGC_PIPE_VIDEO_NV12 = 1,
    ENG_AGC_PIPE_VIDEO_HDR = 2,
    ENG_AGC_PIPE_VIDEO_HLG = 3,
    ENG_AGC_PIPE_VIDEO_PLANAR = 4,
    ENG_AGC_PIPE_UI_BLUR = 5,
    /* #103 upscaler. Sharp = FSR1 EASU + RCAS; AI = Anime4K CNN x2, network S
     * (4 convs) on a base PS5 and M (7 convs + a 1x1 conv split into
     * per-layer accumulate passes) on a PS5 Pro. See agc_upscale_* in
     * eng_agc_runtime.c and tools/gen_upscale_pipes.py. */
    ENG_AGC_PIPE_UP_EASU = 6,
    ENG_AGC_PIPE_UP_RCAS = 7,
    ENG_AGC_PIPE_UP_A4K_FINAL = 8,
    ENG_AGC_PIPE_UP_S_CONV0 = 9,     /* ..12 */
    ENG_AGC_PIPE_UP_M_CONV0 = 13,    /* ..19 */
    ENG_AGC_PIPE_UP_M_ACC0 = 20,     /* ..26 */
    /* Anime4K UL ("Maximum"): 7 layers x 3 RGBA16F textures, and a 1x1 conv
     * over layers 2..6 split into 5 layers x 3 outputs of accumulate passes.
     * Rows come from the generated upscale_wide_pipes.inc. */
    ENG_AGC_PIPE_UP_UL_CONV0 = 27,   /* ..47 */
    ENG_AGC_PIPE_UP_UL_ACC0 = 48,    /* ..62 */
    ENG_AGC_PIPE_UP_RGB_FINAL = 63,  /* depth-to-space with an RGB residual */
    /* Real HDR10 output (eng_agc_runtime_set_hdr_output): the scanout is
     * Bgr10A2Bt2100Pq and these replace their SDR twins when drawing to it. */
    ENG_AGC_PIPE_VIDEO_HDR_PQ = 64,  /* HDR10 source -> PQ passthrough */
    ENG_AGC_PIPE_VIDEO_HLG_PQ = 65,  /* HLG source -> PQ */
    ENG_AGC_PIPE_UI_PQ = 66,         /* OSD / subtitles at 203-nit graphics white */
    /* 8-bit HEVC carrying HLG / PQ (broadcast 4K HLG channels): the NV12
     * twins of the four P010 HDR pipelines. */
    ENG_AGC_PIPE_NV12_HDR = 67,       /* PQ  -> SDR */
    ENG_AGC_PIPE_NV12_HLG = 68,       /* HLG -> SDR */
    ENG_AGC_PIPE_NV12_HDR_PQ = 69,    /* PQ  -> HDR10 */
    ENG_AGC_PIPE_NV12_HLG_PQ = 70,    /* HLG -> HDR10 */
    ENG_AGC_PIPE_VIDEO_P010_SDR = 71, /* 10-bit P010 SDR (BT.709) */
    ENG_AGC_PIPE_VIDEO_DV5 = 72,      /* Dolby Vision profile 5 -> SDR */
    ENG_AGC_PIPE_VIDEO_DV5_PQ = 73,   /* Dolby Vision profile 5 -> HDR10 */
    /* PS5VR frame generation (tools/shaders/gen_framegen_pipes.py) */
    ENG_AGC_PIPE_FG_DOWN = 74,        /* 4x4 luma box */
    ENG_AGC_PIPE_FG_ME_COARSE = 75,   /* block matching, 1/16 size */
    ENG_AGC_PIPE_FG_ME_REFINE = 76,   /* ... 1/4 size */
    ENG_AGC_PIPE_FG_INTERP = 77,      /* the picture halfway */
    ENG_AGC_PIPE_VR_PQ_EXT = 78,      /* HDR10 -> extended-range BT.709 (headset HDR) */
    ENG_AGC_PIPE_VR_HLG_EXT = 79,     /* HLG -> the same */
    ENG_AGC_PIPE_COUNT = 80,

    ENG_AGC_UP_S_CONVS = 4,
    ENG_AGC_UP_M_CONVS = 7,
    ENG_AGC_UP_UL_WIDTH = 3,
    ENG_AGC_UP_UL_LAYERS = 7,
    ENG_AGC_UP_UL_FED_FIRST = 2,     /* first layer the 1x1 conv reads */
    ENG_AGC_UP_UL_CONVS = 21,
    ENG_AGC_UP_UL_ACCS = 15,

    ENG_AGC_FRAME_SLOTS = 3,
};

/* Full-canvas RGBA8 layer surfaces for RmlUi PushLayer / CompositeLayers
 * (backdrop blur, drop-shadow, filter composition). Sized for the maximum
 * render size (4K): pitch-aligned 256, standard pitch = 4K*4 = 15360 bytes.
 * The GPU writes RGBA (no COMP_SWAP) so the C++ side can read them back via
 * the rgba8 T# swizzle; the scanout backbuffer uses COMP_SWAP=ALT for BGRA,
 * so its T# must use the bgra8 swizzle. */
#define ENG_AGC_MAX_LAYERS 4
typedef struct eng_agc_layer_surface {
    uint32_t width;          /* render-size canvas width */
    uint32_t height;         /* render-size canvas height */
    uint32_t pitch_bytes;    /* aligned to 256 */
    uint64_t gpu_addr;       /* GPU-mapped direct-mem base of the surface pixels */
    uint8_t *cpu_base;       /* CPU VA for one-time cache flush / clear */
    SceAgcRegister *mrt;     /* colour-target registers, GPU-mapped (inside gpu_regs) */
    int      in_use;         /* 1 = acquired by CompositeLayers, 0 = free */
    int      pool_index;     /* 0..ENG_AGC_MAX_LAYERS-1, for debug tracking */
} eng_agc_layer_surface_t;

/* Where each [ResourceMapping] pointer goes in the user-SGPR block, copied
 * from the compiled pipeline's PAL metadata. Callers writing user data must
 * read these rather than hardcoding dword indices. */
typedef struct eng_agc_user_data_layout {
    uint32_t vs_count;
    uint32_t ps_count;
    int32_t  vs_const_table_dword;
    int32_t  vs_vertex_table_dword;
    /* The blur pipe's BlurConstants live in the fragment stage, so its const
     * table has a PS slot; ui/video pipes leave this at -1. */
    int32_t  ps_const_table_dword;
    int32_t  ps_texture_table_dword;
} eng_agc_user_data_layout_t;

typedef struct eng_agc_pipeline {
    void            *vs_shader;
    void            *ps_shader;
    eng_agc_user_data_layout_t user_data;
    /* From the compiled pipeline's PAL metadata: which of the automatic
     * user-data values (base vertex / base instance / draw index) the shader
     * consumes. sceAgcDcbDrawIndex/DrawIndexAuto need it to emit the matching
     * packet; the psbc path passed a hardcoded 0 and 2. */
    uint64_t         draw_modifier;
    SceAgcRegister  *cx_regs;
    uint32_t         cx_reg_count;
    SceAgcRegister  *sh_regs;
    uint32_t         sh_reg_count;
    SceAgcRegister  *uc_regs;
    uint32_t         uc_reg_count;
    int              valid;
} eng_agc_pipeline_t;

int  eng_agc_runtime_init(int width, int height, int hdr);
void eng_agc_runtime_shutdown(void);

/* Block until every submitted command buffer has retired, or timeout_ms passes.
 * Used by shutdown, and by the soft close, which parks the app with the GPU
 * quiescent so the switcher can reap it without work in flight. */
void eng_agc_runtime_wait_idle(unsigned timeout_ms);
int  eng_agc_runtime_is_active(void);

void eng_agc_runtime_frame_begin(void);
void eng_agc_runtime_frame_end(void);
void eng_agc_runtime_present(void);
void eng_agc_runtime_set_player_mode(int is_player);
/* Mark the current backbuffer as carrying composited UI, so the next frame that
 * reuses it clears first instead of stacking OSD/subtitle pixels. */
void eng_agc_runtime_note_ui_drawn(void);

/* Whether the backbuffer about to be drawn already holds this video PTS. With
 * two scanout buffers, presenting without redrawing the quad shows the picture
 * from two presents ago; the render loop uses this to redraw a frame that is
 * slower than the panel into both buffers. Stamp with note_video_pts after the
 * blit, before the present. */
int  eng_agc_runtime_video_slot_stale(int64_t pts_us);
void eng_agc_runtime_note_video_pts(int64_t pts_us);

/* Fill a GPU-visible range with a 32-bit pattern using non-temporal stores, so
 * it needs no cache flush. Falls back to stores + clflush when the range is
 * not 16-byte aligned/sized. */
void eng_agc_runtime_stream_fill(void *dst, uint32_t value32, size_t bytes);

void eng_agc_runtime_bind_pipeline(int pipeline_id);
void eng_agc_runtime_set_scissor(int x, int y, int w, int h);
void eng_agc_runtime_set_blend(int blend_mode);

/* RmlUi clip masks, backed by the stencil buffer. Without these, border-radius
 * clipping and masked overlays silently do nothing - children of a rounded
 * container are not clipped to it. operation matches Rml::ClipMaskOperation:
 * 0 = Set, 1 = SetInverse, 2 = Intersect. Bracket the mask geometry with
 * _begin()/_end(); _set_clip_mask() toggles the test for normal drawing. */
/* 0 when the stencil clip-mask path is compiled out. Callers MUST skip drawing
 * the mask geometry entirely in that case: it is shape-only geometry that is
 * never meant to reach the colour buffer, so drawing it paints opaque
 * rectangles over the UI (RmlUi issues ~13 of them per frame). */
int  eng_agc_runtime_clip_mask_supported(void);
void eng_agc_runtime_set_clip_mask(int enable);
void eng_agc_runtime_clip_mask_begin(int operation);
void eng_agc_runtime_clip_mask_end(void);

/* clflush + mfence a range of CPU-written, GPU-read direct memory. Mandatory
 * before the GPU (shader texture-fetch, PM4 indirect-register DMA, or VideoOut
 * scanout) reads anything the CPU just wrote into it - direct memory here is
 * SCE_KERNEL_WB_ONION (write-back cached), so a fresh CPU write can sit in
 * L1/L2 indefinitely with nothing to force it out to the DRAM the GPU actually
 * reads. Matches the historical hardware-verified AGC present path (#27,
 * pp_agc.c's flush_gpu_data() - "Mandatory before every SubmitDcb"), which
 * this bare-metal rewrite omitted. No-op on NULL/zero-length. */
void eng_agc_runtime_cache_flush(const void *address, size_t bytes);

/* Copy the scanout buffer currently being DISPLAYED (the front buffer, i.e.
 * the one the last SetFlip handed to VideoOut) into a caller BGRA buffer -
 * the AGC equivalent of eng_gl_read_default_fb(). Without this the L3+R3
 * screenshot path captures gl_scratch, which an --agc build never renders
 * into, so every capture came back pure black. Reads LINEARLY: if the display
 * is actually interpreting the buffer as tiled, a capture taken through here
 * will look CORRECT while the panel looks scrambled - that difference is
 * itself the diagnosis. No-op when the runtime is not up. */
void eng_agc_runtime_read_scanout(uint32_t *bgra, int width, int height);

/* Probe `n` pixels from a fixed low-discrepancy spread across the front buffer
 * scanout into `rgb` (3 bytes each, R,G,B). Used by the codec sweep's colour
 * probe (eng_sweep_probe_colour) to test colour accuracy and detect matrix /
 * range / transfer mismatches. Returns the number of pixels probed. */
int eng_agc_probe_rgb(uint8_t *rgb, int n);

/* Mark the current frame as having real draw content. A frame that never calls
 * this is closed without being submitted or flipped, so it cannot blank the
 * screen between the UI's infrequent redraws. */
void eng_agc_runtime_note_draw(void);

/* Record that a draw was dropped: kind 0 = transient-ring allocation failed,
 * kind 1 = texture allocation/descriptor rejected. Both make UI content simply
 * not appear, with no error anywhere else. */
void eng_agc_runtime_note_drop(int kind);

/* User-SGPR layout of a compiled pipeline; zeroed counts mean "not valid". */
eng_agc_user_data_layout_t eng_agc_runtime_get_user_data_layout(int pipeline_id);

/* Compiled draw modifier of a pipeline. The blur pass draws the backdrop quad
 * with sceAgcDcbDrawIndex + the pipeline's modifier so the VS reads
 * gl_VertexIndex from the index buffer; 0 when the pipeline is not valid. */
uint64_t eng_agc_runtime_get_pipe_draw_modifier(int pipeline_id);

SceAgcCommandBuffer      *eng_agc_runtime_get_current_cb(void);
eng_agc_transient_ring_t *eng_agc_runtime_get_transient_ring(void);
uint32_t                  eng_agc_runtime_get_current_slot(void);
void                      eng_agc_runtime_get_size(int *width, int *height);
int                       eng_agc_runtime_is_display_hdr(void);
int                       eng_agc_runtime_get_display_dynamic_range(void);

/* Refresh rate control & 120 Hz output */
int                       eng_agc_runtime_supports_120hz(void);
int                       eng_agc_runtime_is_120hz(void);
int                       eng_agc_runtime_get_refresh_rate(void);
int                       eng_agc_runtime_set_120hz(int enable);

/*
 * Real HDR10 output. enable=1 re-registers the display buffers as
 * Bgr10A2Bt2100Pq (10:10:10:2, BT.2020, SMPTE ST.2084 PQ) - the TV switches
 * into HDR - and draws the video / UI with the *_PQ pipelines; enable=0 goes
 * back to 8-bit SDR. Call between frames (the loop does it before
 * frame_begin). Returns 0 on success, <0 if the display refused (it is then
 * left in SDR).
 */
int                       eng_agc_runtime_set_hdr_output(int enable);
int                       eng_agc_runtime_hdr_output_active(void);
/* color_trc of the last video frame presented (16 = PQ, 18 = HLG, else SDR),
 * -1 before any - whatever its bit depth. What the loop uses to decide
 * whether HDR10 is wanted. */
int                       eng_agc_runtime_last_video_trc(void);

/* Layer surfaces for RmlUi PushLayer / CompositeLayers (backdrop-filter: blur).

 *
 * Each layer is a full-canvas RGBA8 render target, allocated from the direct
 * memory carve and built with standard COMP_SWAP (memory = R,G,B,A bytes)
 * so the C++ render interface can sample it back with the rgba8 T# swizzle.
 * The scanout backbuffer uses COMP_SWAP=ALT (memory = B,G,R,A bytes) and
 * therefore requires the bgra8 T# swizzle for sampling.  This distinction
 * is critical and silent-on-failure: getting it wrong produces visually
 * identical textures with swapped red and blue, not a crash.
 *
 * The render target is switched by eng_agc_set_layer_target(); the blur pipe
 * and the copy pass draw through it.  Layers are cleaned to transparent black
 * on acquire (memset the scissor region + clflush, not the full surface) and
 * released when CompositeLayers finishes.  Pool exhaustion returns NULL and
 * the C++ side degrades gracefully: the UI draws, but the backdrop blur
 * simply does not appear. */
int                       eng_agc_has_layers(void);
int                       eng_agc_layer_acquire(eng_agc_layer_surface_t **out);
void                      eng_agc_layer_release(eng_agc_layer_surface_t *layer);
/* NULL = switch to the scanout backbuffer (the default render target for UI
 * drawing).  Builds SceAgcRegister colour-target registers internally;
 * the caller must emit the write through the DCB.  Flushes the new MRT
 * register block so the GPU sees it even when the flush in frame_begin
 * already happened. */
int                       eng_agc_set_layer_target(const eng_agc_layer_surface_t *layer);
/* Fill a surface descriptor for the active scanout backbuffer, for use by
 * the blur pipeline's source texture (the base layer is always the scanout
 * in non-layer drawing, and must be sampled with the bgra8 T# swizzle). */
void                      eng_agc_get_scanout_layer(eng_agc_layer_surface_t *out);
/* CB colour-buffer flush (event 45).  Mandatory between H and V passes of
 * the blur and between any two same-frame passes that read-then-write the
 * same surface. */
void                      eng_agc_flush_color_target(void);

/* Composite a premultiplied 0xAABBGGRR OSD buffer over the current frame.
 * `upload` = "the buffer changed since last call"; when 0 the previous upload is
 * redrawn. Called via eng_gl_composite_bgra() so main.c stays backend-agnostic. */
void eng_agc_composite_bgra(const uint32_t *fb, int w, int h, int upload);

/* PS5VR: composite a premultiplied 0xAABBGGRR overlay canvas (at most
 * 1920x1080) stretched over the whole display (bilinear) at `opacity` (0..1).
 * layer 0 = subtitles, 1 = controls; call 0 before 1. upload=0 redraws the
 * layer's last upload. */
void eng_agc_composite_overlay(int layer, const uint32_t *fb, int w, int h, int upload,
                               float opacity);

/* PS5VR: the whole backbuffer to opaque black, drawn on the GPU (call right
 * after frame_begin, before the video quad). */
void eng_agc_runtime_clear_black(void);

/* #103 video upscaler.
 *
 * Settings -> Playback & Video -> UPSCALING picks the requested mode; the
 * render loop pushes it here every frame (a plain store). What a frame
 * actually got can be less:
 *   - bypass: the source is not smaller than the image on the panel, it is
 *     10-bit/HDR (v1 is SDR 8-bit only), a pipeline is missing, or the scratch
 *     surfaces could not be allocated. Logged once per change of source/plan.
 *   - cap: the GPU went over budget while upscaling, so the runtime stepped
 *     AI -> Sharp -> Off for the rest of the session (toast once). Picking a
 *     mode again in Settings clears the cap.
 * Off is byte-for-byte the pre-#103 single pass. */
enum {
    ENG_AGC_UPSCALE_OFF = 0,
    ENG_AGC_UPSCALE_SHARP = 1,
    ENG_AGC_UPSCALE_AI = 2,
};
void        eng_agc_upscale_set_mode(int mode);
/* Which Anime4K network AI mode runs. AUTO = Large on a detected PS5 Pro,
 * Standard otherwise; STANDARD / LARGE / MAXIMUM are the Settings override
 * (the Pro probe cannot identify every Pro). Each steps down one network
 * (Maximum -> Large -> Standard) when its pipelines or scratch memory are
 * missing or it goes over the GPU budget. */
enum {
    ENG_AGC_UPNET_AUTO = 0,
    ENG_AGC_UPNET_STANDARD = 1,
    ENG_AGC_UPNET_LARGE = 2,
    ENG_AGC_UPNET_MAXIMUM = 3,       /* Anime4K UL - meant for a PS5 Pro */
};
void        eng_agc_upscale_set_network(int pref);
/* What the last video frame actually used: "Off", "Sharp", "AI (Standard)",
 * "AI (Large)",
 * with the bypass reason appended when the requested mode was not Off. */
const char *eng_agc_upscale_label(void);
/* -1, or the mode the runtime just downgraded to because of GPU time
 * (ENG_AGC_UPSCALE_AI means one AI network down: Maximum -> Large ->
 * Standard; eng_agc_upscale_label() names the new one). Reading it clears it, so the
 * caller toasts exactly once. */
int         eng_agc_upscale_take_downgrade(void);

/* Draw the video quad. Returns 0 when the quad was emitted (and the current
 * backbuffer stamped with pts_us), -1 when the frame was rejected. */
int  eng_agc_blit_yuv(const uint8_t *y,  int y_pitch,
                      const uint8_t *uv, int uv_pitch,
                      const uint8_t *u,  int u_pitch,
                      const uint8_t *v,  int v_pitch,
                      int coded_w, int coded_h,
                      int disp_w, int disp_h,
                      int view_mode, int ten_bit, int color_trc,
                      int is_direct, int64_t pts_us);

/* 1 if that pipeline compiled and can be bound. */
int eng_agc_runtime_pipeline_valid(int pipeline_id);

#ifdef APP_VR
/*
 * PS5VR: video in the PSVR2 headset. While a view is set, eng_agc_blit_yuv
 * draws the decoded picture at source size into an offscreen surface (only when
 * the picture changes) and projects it, per eye, into one of the side-by-side
 * eye buffers; frame_end then calls the submit hook (Hmd2 SetParam + VR flip)
 * after the GPU fence instead of flipping the TV.
 */
enum {
    ENG_AGC_VR_FLAT = 0,       /* a flat screen in front of the viewer */
    ENG_AGC_VR_180 = 180,      /* equirectangular half sphere */
    ENG_AGC_VR_360 = 360,      /* equirectangular full sphere */
};
enum {
    ENG_AGC_VR_MONO = 0,
    ENG_AGC_VR_SBS = 1,        /* left eye in the left half */
    ENG_AGC_VR_TB = 2,         /* left eye in the top half */
    ENG_AGC_VR_TB_TOP = 3,     /* top half to both eyes (bottom = depth map) */
    ENG_AGC_VR_SBS_LEFT = 4,   /* left half to both eyes (right = depth map) */
};
typedef struct eng_agc_vr_view {
    int   buffer;              /* eye buffer to draw */
    float q[2][4];             /* eye orientation x,y,z,w: eye -> video space */
    float tan[2][4];           /* render FOV tangents left, right, up, down */
    int   projection;          /* ENG_AGC_VR_* */
    int   stereo;              /* ENG_AGC_VR_MONO / _SBS / _TB */
    float screen_deg;          /* flat screen: horizontal size in degrees */
    int   sharpen;             /* -1 auto, 0 off, 1 RCAS, 2 EASU upscale + RCAS */
    int   swap;                /* 1: the picture's left half / top is the right eye */
} eng_agc_vr_view_t;

/* Allocate `count` eye buffers of (2 * eye_w) x eye_h RGBA8 (render-target
 * tiling) and return each one's 32-byte T# for the Hmd2 layer. 0 on success. */
int  eng_agc_vr_targets(uint32_t eye_w, uint32_t eye_h, int count, uint32_t tsharp[][8],
                        int ten_bit);   /* 1: 10:10:10:2 buffers (Horizon's format) */
/* The UI panel shown in the headset over the video (or the empty space when
 * none plays): a canvas of premultiplied pixels (ui_canvas layout), uploaded
 * when `upload`; placed world-locked at yaw/pitch degrees (clockwise / up from
 * the video's front), `dist` metres away, `width` metres wide. show = 0 hides. */
void eng_agc_vr_panel(const uint32_t *px, int w, int h, int upload, int show, float yaw,
                      float pitch, float dist, float width, float opacity);
/* Overlay quads for the next headset frame: corners in the head-relative space
 * panels are drawn in (metres), premultiplied RGBA per corner. */
void eng_agc_vr_overlay_clear(void);
void eng_agc_vr_overlay_quad(const float p[4][3], const uint32_t rgba[4]);
/* The hand pointer's dot on the shown panel at (u, v) in 0..1, blue when pressed. */
void eng_agc_vr_panel_cursor(int show, float u, float v, int pressed);
/* The view for the frame being built; NULL leaves VR (TV flips resume). */
void eng_agc_vr_set_view(const eng_agc_vr_view_t *view);
/* Called by frame_end once the GPU has finished the frame. */
void eng_agc_vr_set_submit(void (*submit)(void));
/* Project the last picture again (no new video frame); 0 if anything drew. */
/* Test loop: the last eye buffer (both eyes side by side, or the left one) as
 * BGRA, every `shrink`-th pixel. 0 on success. */
int  eng_agc_vr_read_eye(uint32_t *bgra, int shrink, int left_only, int *out_w, int *out_h);
int  eng_agc_vr_draw(void);
/* Frame generation for 30 fps pictures (on by default). */
void eng_agc_vr_set_framegen(int on);
/* A generated picture is showing: the real frame needs the next slot. */
int  eng_agc_vr_fg_pending(void);
/* The TV VideoOut port. */
int  eng_agc_runtime_video_handle(void);
/* 1 once a picture has been drawn into the source surface. */
int  eng_agc_vr_has_picture(void);
/* Forget the held picture (a new video starts). */
void eng_agc_vr_reset_picture(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* ENG_AGC_RUNTIME_H */
