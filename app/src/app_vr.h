/*
 * app_vr - PSVR2 output for the PS5VR build (APP_VR).
 *
 * The player hands a VR format to app_vr_start when a VR video's first frame
 * is ready; from then on every loop iteration calls app_vr_begin_frame (Hmd2
 * paces it to the headset) and draws through eng_agc_blit_yuv / eng_agc_vr_draw,
 * and the engine hands each finished frame back to Hmd2. In a normal PS5VR
 * build every call is a no-op and app_vr_parse finds no format.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct app_vr_format {
    int projection;     /* 360, 180, or 0 = flat screen */
    int stereo;         /* 0 mono, 1 side by side, 2 top-bottom */
} app_vr_format;

/* Picture options (defaults: sharpen -1 = auto, 10-bit, eye 0 = 2816 px). */
typedef struct app_vr_options {
    int sharpen;        /* -1 auto, 0 off, 1 sharpen, 2 upscale + sharpen */
    int bits;           /* 8 or 10 (at start) */
    int eye_px;         /* per-eye buffer size, 1024..4096; 0 = default (at start) */
    int swap;           /* stereo: the eyes are the other way round */
    float screen_deg;   /* flat screen width in degrees; 0 = 70 */
} app_vr_options;
void app_vr_set_options(const app_vr_options *o);
void app_vr_get_options(app_vr_options *o);
/* The video format, changed while running (playback start, the VR menu). */
void app_vr_set_format(const app_vr_format *fmt);
void app_vr_get_format(app_vr_format *fmt);
/* Where the head points in the video's space: yaw clockwise, pitch up (deg). */
void app_vr_head_dir(float *yaw, float *pitch);
/* 1 while the headset is on the head (polled at most twice a second). */
int  app_vr_worn(void);
/* Show the headset's view on the TV (Hmd2 mirroring); off by default. */
void app_vr_set_mirror(int on);

/* "360_tb", "360_sbs", "360", "180_sbs", "180", "flat_sbs", "flat" (case and
 * separators ignored; "lr" = sbs, "ou" = tb). 1 if s names a VR format. */
int  app_vr_parse(const char *s, app_vr_format *out);
/* Guess the format from a file name or title ("..._360_TB.mp4"). */
int  app_vr_guess(const char *name, app_vr_format *out);
/* Check a 180/360 guess against the decoded frame's shape: pick the stereo
 * layout that fits it, or fall back to a flat screen if none does. */
void app_vr_fit_frame(app_vr_format *f, int w, int h);

/* Enter VR output (fmt NULL: nothing playing - the empty space). 0 on success. */
/* Loads the headset modules; must run before the sandbox is opened. */
int  app_vr_preload(void);
/* Binds the named exports of another VR system library (before the sandbox
 * opens). 0 when all were found. */
int  app_vr_bind(const char *lib, const char *const *names, void **slots, int n);
/* The predicted display time of the frame being drawn (tracker time). */
uint64_t app_vr_frame_time(void);
/* Logs whether hand tracking's camera opens succeed. */
void app_vr_camera_probe(void);
/* Where libSceVrTracker2 is loaded (0 before app_vr_preload). */
uintptr_t app_vr_tracker_base(void);
/* The head position in tracker space (metres) and the rotation from tracker
 * space to the space the picture and panels are drawn in. */
void app_vr_head_pose(float pos[3], float yaw_q[4]);
/* A Sense controller (0 left, 1 right): head-relative position (m) and
 * orientation in picture space, its buttons and stick. 1 when tracked. */
int app_vr_controller(int side, float pos[3], float q[4], uint32_t *buttons, float stick[2]);
/* A tracker-space point (a hand joint) into head-relative picture space. */
void app_vr_to_view(const float p[3], float out[3]);
/* The headset's rumble, 0 (off) to 25. */
void app_vr_vibrate(int strength);
/* The head orientation in tracker space. */
void app_vr_head_quat(float q[4]);
/* Eye tracking: Hmd2's gaze result (0x38 bytes); its return code. */
int app_vr_gaze(void *result);
/* Where the eyes look (unit vector, picture space); 1 when tracked. */
int app_vr_gaze_dir(float dir[3]);
/* The headset VideoOut port (bus 32), -1 when the session is off. */
int app_vr_port(void);
int  app_vr_start(const app_vr_format *fmt, int user_id);
void app_vr_stop(void);
/* Stop and start again (new eye-buffer format). */
int  app_vr_restart(void);
/* Everything down, before the app ends. */
void app_vr_shutdown(void);
int  app_vr_active(void);
/* Sleep to the next 60 Hz slot (the headset shows each frame we submit for two
 * 120 Hz refreshes, so slot-aligned frames keep an even cadence). */
void app_vr_wait_slot(void);
/* Begin a headset frame and set the engine's view; 0 when it may be drawn. */
int  app_vr_begin_frame(void);
/* Face the video's front where the viewer looks now. */
void app_vr_recenter(void);

#ifdef __cplusplus
}
#endif
