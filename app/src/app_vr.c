/*
 * app_vr - PSVR2 output for the PS5VR build. See app_vr.h.
 *
 * Hmd2 and VrTracker2 are system modules: they are loaded with
 * sceSysmoduleLoadModule and their functions found through sceKernelDlsym, or,
 * where an app may not dlsym them, at the FW 13.60 export offsets after a
 * module-name and rodata check (vr_exports_1360.h). The frame setup is the one
 * Horizon Call of the Mountain uses (ps5vr/research/hmd2-notes.md): a type 0x10
 * layer whose two textures are the same side-by-side eye buffer, with per-eye
 * viewports, FOV tangents, the device pose and the predicted display time.
 */
#include "app_vr.h"

#include <ctype.h>
#include <string.h>

#ifdef APP_VR
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "eng_agc_runtime.h"
#include "eng_boot_trace.h"
#include "eng_vr_audio.h"
#include "vr_system.h"

typedef struct { uint64_t size, align; } SizeAlign;

#define HMD2_FUNCS(X)                                                                        \
    X(int, sceHmd2Initialize, (const void *param))                                          \
    X(int, sceHmd2Open, (int user_id, int type, int index, const void *param))               \
    X(int, sceHmd2GetDeviceInformation, (void *info))                                        \
    X(int, sceHmd2GetFieldOfViewWithoutHandle, (void *fov))                                  \
    X(SizeAlign, sceHmd2ReprojectionQueryBufferSizeAlign, (void))                            \
    X(SizeAlign, sceHmd2ReprojectionQueryDisplayBufferSizeAlign, (void))                     \
    X(int, sceHmd2ReprojectionInitialize, (const void *param, const void *reserved))         \
    X(int, sceHmd2ReprojectionEnableVrMode, (uint64_t mode))                                 \
    X(int, sceHmd2ReprojectionDisableVrMode, (void))                                         \
    X(int, sceHmd2ReprojectionBeginFrame, (uint64_t frame))                                  \
    X(int, sceHmd2ReprojectionGetPredictedDisplayTime, (uint64_t frame, uint64_t *time))     \
    X(int, sceHmd2ReprojectionSetParam, (uint32_t buffer, uint64_t frame, const void *param, \
                                         const void *trace))                               \
    X(int, sceHmd2ReprojectionSetRenderConfig, (uint64_t half_rate, uint64_t r0, uint64_t r1)) \
    X(SizeAlign, sceHmd2ReprojectionGetMirroringWorkMemorySizeAlign, (void))                \
    X(SizeAlign, sceHmd2ReprojectionGetMirroringDisplayBufferSizeAlign, (void))             \
    X(int, sceHmd2ReprojectionSetMirroringOption, (const void *opt))                       \
    X(int, sceHmd2ReprojectionEnableMirroring, (const void *param))                        \
    X(int, sceHmd2ReprojectionDisableMirroring, (void))                                    \
    X(int, sceHmd2ReprojectionTerminate, (void))                                           \
    X(int, sceHmd2Close, (int handle))                                                     \
    X(int, sceHmd2Terminate, (void))                                                       \
    X(int, sceHmd2InternalImageOpenBySlot, (int slot))                                     \
    X(int, sceHmd2InternalLedDetectorOpenBySlot, (int slot))                               \
    X(int, sceHmd2GazeGetResult, (const void *param, void *result))                        \
    X(int, sceHmd2SetVibration, (int handle, int strength))                        \
    X(int, sceHmd2InternalGazeStart, (void))                                               \
    X(int, sceHmd2InternalGazeStop, (void))
#define VRTRACKER2_FUNCS(X)                                                                  \
    X(int, sceVrTracker2QueryMemory, (const void *param, void *out))                         \
    X(int, sceVrTracker2Initialize, (const void *param))                                     \
    X(int, sceVrTracker2RegisterDevice, (int type, int handle))                              \
    X(int, sceVrTracker2GetResult, (const void *param, void *result))                        \
    X(int, sceVrTracker2UnregisterDevice, (int handle))                                      \
    X(int, sceVrTracker2Finalize, (void))
#define DECLARE_PTR(ret, name, args) static ret(*name) args;
HMD2_FUNCS(DECLARE_PTR)
VRTRACKER2_FUNCS(DECLARE_PTR)

#define SYSMODULE_HMD2       0x12a
#define SYSMODULE_VRTRACKER2 0x12b
#define VR_MODE_120HZ        0x2000f

#define VR_QUEUE_FULL        ((int)0x8A72001A)
#define VO_BUSY              ((int)0x80290009)
/* Per-eye buffer. It is spread evenly in tangent space, so the centre gets the
 * fewest texels per degree - where the lens puts the panel's most (20+ px/deg).
 * 2816 gave the centre ~15 px/deg after the overscan; 4096 gives ~22, a sharper
 * centre for 8K sources, for ~0.5 ms more GPU per frame (8K: 2.2-2.6 ms) and
 * 3 x 64 MB more direct memory. app_vr_set_options can change it. */
#define EYE_DEFAULT          4096
#define EYE_W                g.eye
#define EYE_H                g.eye
#define NUM_BUFFERS          3
/* Margin the reprojection turns into while the head moves between projections
 * (one video frame, up to ~33 ms). */
#define OVERSCAN             1.15f
/* 120 Hz output with half-rate rendering (Horizon's setup): every frame we
 * submit is shown for exactly two refreshes, so 60 fps video gets 2 refreshes
 * per frame and 30 fps gets 4 - no 90 Hz 1/2 alternation. Hmd2's BeginFrame
 * does not hold us back, so the 60 Hz cadence is ours. */
#define FRAME_US             16667

int sceSysmoduleLoadModule(uint32_t id);
const char *sceKernelGetFsSandboxRandomWord(void);
int sceKernelLoadStartModule(const char *path, size_t argc, const void *argv, uint32_t flags,
                             void *opt, int *res);
int sceKernelDlsym(int handle, const char *name, void **addr);
int sceKernelAllocateDirectMemory(int64_t start, int64_t end, size_t len, size_t align, int type,
                                  int64_t *offset);
int sceKernelMapDirectMemory(void **addr, size_t len, int prot, int flags, int64_t offset,
                             size_t align);
int sceKernelUsleep(unsigned usec);
uint64_t sceKernelGetProcessTime(void);

/* libSceVideoOut; on bus type 32 the port is Hmd2's "VR driver". */
typedef struct { void *data, *metadata, *reserved0, *reserved1; } vo_buffer_t;
typedef struct { uint8_t reserved[80]; } vo_attribute_t;
int sceVideoOutOpen(int user_id, int bus_type, int index, const void *param);
int sceVideoOutClose(int handle);
void sceVideoOutSetBufferAttribute2(void *attr, uint64_t format, uint32_t tiling, uint32_t width,
                                    uint32_t height, uint64_t option, uint32_t reserved0,
                                    uint64_t reserved1);
int sceVideoOutRegisterBuffers2(int handle, int set_index, int start_index, void *buffers,
                                int count, void *attr, int category, void *option);
int sceVideoOutUnregisterBuffers(int handle, int set_index);
int sceVideoOutSubmitFlip(int handle, int index, int flip_mode, int64_t flip_arg);
int sceVideoOutConfigureOutput(int handle, uint64_t mode, const void *options, const void *r0,
                               uint64_t r1);

/* -- module loading -------------------------------------------------------- */
typedef struct { const char *name; uint32_t offset; } export_sig_t;
typedef struct { uint32_t va; const char *text; } export_anchor_t;
#include "vr_exports_1360.h"

typedef struct { const char *name; void **slot; } slot_t;
#define SLOT(ret, name, args) {#name, (void **)&name},
static slot_t kHmd2Slots[] = {HMD2_FUNCS(SLOT)};
static slot_t kVrTracker2Slots[] = {VRTRACKER2_FUNCS(SLOT)};
#define COUNT(a) (sizeof(a) / sizeof *(a))

typedef struct {
    uint64_t size;
    char name[256];
    struct { uint64_t addr; uint32_t size; int32_t prot; } seg[4];
    uint32_t nseg;
    uint8_t fingerprint[20];
} module_info_t; /* 0x160 */
int sceKernelGetModuleInfo(int handle, module_info_t *info);

static int load_sysmodule(uint32_t id, const char *name)
{
    int rc = sceSysmoduleLoadModule(id);
    char path[128];
    snprintf(path, sizeof path, "/%s/common/lib/%s.sprx", sceKernelGetFsSandboxRandomWord(), name);
    int res = 0;
    int h = sceKernelLoadStartModule(path, 0, NULL, 0, NULL, &res);
    eng_bt("vr: sysmodule 0x%x %s -> 0x%08x, handle 0x%x", id, name, rc, h);
    return h;
}

static uintptr_t module_base(int h, const char *file, const export_anchor_t *anchor)
{
    module_info_t mi;
    memset(&mi, 0, sizeof mi);
    mi.size = sizeof mi;
    if (sceKernelGetModuleInfo(h, &mi) != 0 || strcmp(mi.name, file) != 0)
        return 0;
    uintptr_t base = 0;
    for (unsigned i = 0; i < mi.nseg && i < 4; i++)
        if (!base || mi.seg[i].addr < base)
            base = mi.seg[i].addr;
    const uintptr_t want = base + anchor->va;
    const size_t len = strlen(anchor->text) + 1;
    for (unsigned i = 0; i < mi.nseg && i < 4; i++) {
        const uintptr_t lo = mi.seg[i].addr, hi = lo + mi.seg[i].size;
        if (want >= lo && want + len <= hi && (mi.seg[i].prot & 1))
            return memcmp((const void *)want, anchor->text, len) == 0 ? base : 0;
    }
    return 0;
}

static int resolve_module(int h, const char *file, slot_t *slots, size_t n,
                          const export_sig_t *table, size_t table_n,
                          const export_anchor_t *anchor)
{
    uintptr_t base = 0;
    int tried_base = 0, ok = 1;
    for (size_t i = 0; i < n; i++) {
        if (sceKernelDlsym(h, slots[i].name, slots[i].slot) == 0 && *slots[i].slot)
            continue;
        if (!tried_base) {
            base = module_base(h, file, anchor);
            tried_base = 1;
            eng_bt("vr: %s by export offsets, base %p", file, (void *)base);
        }
        *slots[i].slot = NULL;
        for (size_t j = 0; base && j < table_n; j++)
            if (!strcmp(table[j].name, slots[i].name))
                *slots[i].slot = (void *)(base + table[j].offset);
        if (!*slots[i].slot) {
            eng_bt("vr: missing %s", slots[i].name);
            ok = 0;
        }
    }
    return ok ? 0 : -1;
}

static void *direct_alloc(size_t size, size_t align)
{
    size = (size + 0xffff) & ~(size_t)0xffff;
    int64_t off = 0;
    void *va = NULL;
    if (sceKernelAllocateDirectMemory(0, (int64_t)16 << 30, size, align, 12 /* WB_ONION */, &off) != 0 ||
        sceKernelMapDirectMemory(&va, size, 0x33, 0, off, align) != 0) {
        eng_bt("vr: direct alloc 0x%zx failed", size);
        return NULL;
    }
    memset(va, 0, size);
    return va;
}

/* -- state ----------------------------------------------------------------- */
typedef struct { void *layers; uint32_t count; uint32_t pad; } layer_set_t;

static struct {
    int loaded;            /* modules resolved, Hmd2 open, tracker up, reprojection ready */
    int user;              /* the user the session is for */
    int failed;            /* setup failed once: do not retry every video */
    int hmd;
    float fov[4];          /* out, in, top, bottom tangents */
    int active;
    int vh;
    app_vr_format fmt;
    uint8_t layers[NUM_BUFFERS][0x130] __attribute__((aligned(16)));
    layer_set_t sets[NUM_BUFFERS];
    uint64_t frame;
    uint64_t next_begin;   /* process time (us) the next frame may begin */
    int buf;
    float yaw_q[4];        /* tracker space -> video space */
    /* the frame begun by app_vr_begin_frame */
    uint64_t t;
    float pd[3], qd[4];
    int frame_open;
    unsigned submits, errors;
    uint32_t eye;
    app_vr_options opt;
    float head_yaw, head_pitch;     /* video space, degrees */
    int worn;
    uint64_t worn_checked;          /* process time (us) */
} g = { .vh = -1, .yaw_q = {0, 0, 0, 1}, .eye = EYE_DEFAULT, .opt = {-1, 10, 0, 0, 70.0f},
        .worn = 1 };

void app_vr_set_options(const app_vr_options *o)
{
    if (o)
        g.opt = *o;
}

void app_vr_get_options(app_vr_options *o)
{
    if (o)
        *o = g.opt;
}

void app_vr_set_format(const app_vr_format *fmt)
{
    if (fmt) {
        g.fmt = *fmt;
        eng_bt("vr: format projection %d stereo %d", fmt->projection, fmt->stereo);
    }
}

void app_vr_get_format(app_vr_format *fmt)
{
    if (fmt)
        *fmt = g.fmt;
}

uint64_t app_vr_frame_time(void)
{
    return g.t;
}

/* Eye tracking: Hmd2's gaze result (0x38 bytes) for the open headset. */
int app_vr_gaze(void *result)
{
    if (!g.loaded || !sceHmd2GazeGetResult)
        return -1;
    /* Eye tracking runs only when asked for: an ioctl on the Hmd2 device
     * (mode 1); without it the result stays "not valid". */
    static int s_started;
    if (!s_started && sceHmd2InternalGazeStart) {
        s_started = 1;
        eng_bt("vr: eye tracking start -> 0x%08x", sceHmd2InternalGazeStart());
    }
    uint8_t param[0x10];
    memset(param, 0, sizeof param);
    *(uint32_t *)param = sizeof param;
    *(int32_t *)(param + 4) = g.hmd;
    memset(result, 0, 0x38);
    *(uint32_t *)result = 0x38;
    return sceHmd2GazeGetResult(param, result);
}

static void qmul(const float a[4], const float b[4], float o[4]);

/* Where the eyes look, in the space the picture and panels are drawn in. The
 * gaze result: +4 status (0 ok), +0x10 1 = valid, +0x14 a unit direction in
 * headset space (-z forward). 1 when valid. */
int app_vr_gaze_dir(float dir[3])
{
    uint8_t r[0x38];
    if (app_vr_gaze(r) != 0 || *(int32_t *)(r + 4) != 0 || *(uint32_t *)(r + 0x10) != 1)
        return 0;
    const float *gd = (const float *)(r + 0x14);
    float head[4];
    qmul(g.yaw_q, g.qd, head);
    /* v' = v + 2w (q x v) + 2 q x (q x v) */
    const float x = head[0], y = head[1], z = head[2], w = head[3];
    const float tx = 2 * (y * gd[2] - z * gd[1]), ty = 2 * (z * gd[0] - x * gd[2]),
                tz = 2 * (x * gd[1] - y * gd[0]);
    dir[0] = gd[0] + w * tx + (y * tz - z * ty);
    dir[1] = gd[1] + w * ty + (z * tx - x * tz);
    dir[2] = gd[2] + w * tz + (x * ty - y * tx);
    return 1;
}

void app_vr_head_quat(float q[4])
{
    memcpy(q, g.qd, sizeof g.qd);
}

static void qmul(const float a[4], const float b[4], float o[4]);

/* -- the PS VR2 Sense controllers ------------------------------------------- */
/* As Horizon opens them: scePadOpen(user, 3 = left / 4 = right, 0, NULL), each
 * registered with the tracker: device 2 takes pad type 3, device 1 type 4
 * (its handle check; 0 is the headset). The pose
 * comes from VrTracker2GetResult with the pad handle; buttons from
 * scePadVrControllerReadState (0x68 bytes: +0 buttons, +4 stick x/y bytes,
 * +0x40 connected). */
int scePadOpen(int user, int type, int index, const void *param);
int scePadVrControllerReadState(int handle, void *state);

static struct {
    int handle[2];
    int registered[2];
    int opened;
    uint32_t buttons[2], last_logged[2];
    int connected[2];
} s_ctl = {{-1, -1}, {0, 0}, 0, {0, 0}, {0, 0}, {0, 0}};

static void controllers_open(void)
{
    if (s_ctl.opened || !g.loaded)
        return;
    s_ctl.opened = 1;
    for (int side = 0; side < 2; side++) {
        s_ctl.handle[side] = scePadOpen(g.user, side ? 4 : 3, 0, NULL);
        const int rc = s_ctl.handle[side] >= 0
                           ? sceVrTracker2RegisterDevice(side ? 1 : 2, s_ctl.handle[side]) : -1;
        s_ctl.registered[side] = rc == 0;
        eng_bt("vr: %s controller pad 0x%x, tracker register 0x%08x", side ? "right" : "left",
               s_ctl.handle[side], rc);
    }
}

/* The controller's pose, head-relative, in the space panels are drawn in:
 * position (m) and orientation. 1 when tracked. */
int app_vr_controller(int side, float pos[3], float q[4], uint32_t *buttons, float stick[2])
{
    if (side < 0 || side > 1 || !g.active)
        return 0;
    controllers_open();
    if (!s_ctl.registered[side])
        return 0;
    uint8_t st[0x68];
    memset(st, 0, sizeof st);
    s_ctl.connected[side] = scePadVrControllerReadState(s_ctl.handle[side], st) == 0 && st[0x40];
    s_ctl.buttons[side] = *(uint32_t *)st;
    if (buttons)
        *buttons = s_ctl.buttons[side];
    if (stick) {
        stick[0] = (st[4] - 128) / 127.0f;
        stick[1] = (st[5] - 128) / 127.0f;
    }
    static uint8_t res[0x400];
    uint8_t gp[0x20];
    memset(gp, 0, sizeof gp);
    *(uint32_t *)(gp + 0x00) = 0x20;
    *(int32_t *)(gp + 0x04) = s_ctl.handle[side];
    *(uint64_t *)(gp + 0x10) = g.t;
    memset(res, 0, sizeof res);
    const int rc = sceVrTracker2GetResult(gp, res);
    /* (logged while the layout is being confirmed) */
    static uint64_t s_next[2];
    const uint64_t now = sceKernelGetProcessTime();
    if (now >= s_next[side] || s_ctl.buttons[side] != s_ctl.last_logged[side]) {
        s_next[side] = now + 3000000;
        s_ctl.last_logged[side] = s_ctl.buttons[side];
        const uint32_t *w = (const uint32_t *)res;
        const float *f = (const float *)(res + 0x60);
        eng_bt("vr: ctl %d rc 0x%08x conn %d btn %08x stick %02x%02x %02x%02x | %08x %08x | %.3f %.3f %.3f q %.3f %.3f %.3f %.3f",
               side, rc, s_ctl.connected[side], s_ctl.buttons[side], st[4], st[5], st[6], st[7],
               w[0], w[1], (double)f[0], (double)f[1], (double)f[2], (double)f[4], (double)f[5],
               (double)f[6], (double)f[7]);
    }
    if (rc != 0 || !*(uint32_t *)(res + 4))
        return 0;
    float p[3], dq[4], t[3];
    memcpy(p, res + 0x60, 12);
    memcpy(dq, res + 0x70, 16);
    for (int k = 0; k < 3; k++)
        t[k] = p[k] - g.pd[k];
    /* into picture space: yaw_q rotation of the offset and the orientation */
    const float x = g.yaw_q[0], y = g.yaw_q[1], z = g.yaw_q[2], w = g.yaw_q[3];
    const float tx = 2 * (y * t[2] - z * t[1]), ty = 2 * (z * t[0] - x * t[2]), tz = 2 * (x * t[1] - y * t[0]);
    pos[0] = t[0] + w * tx + (y * tz - z * ty);
    pos[1] = t[1] + w * ty + (z * tx - x * tz);
    pos[2] = t[2] + w * tz + (x * ty - y * tx);
    qmul(g.yaw_q, dq, q);
    return 1;
}

void app_vr_head_pose(float pos[3], float yaw_q[4])
{
    memcpy(pos, g.pd, sizeof g.pd);
    memcpy(yaw_q, g.yaw_q, sizeof g.yaw_q);
}

/* A tracker-space point (hand joints) into the head-relative picture space. */
void app_vr_to_view(const float p[3], float out[3])
{
    const float t[3] = {p[0] - g.pd[0], p[1] - g.pd[1], p[2] - g.pd[2]};
    const float x = g.yaw_q[0], y = g.yaw_q[1], z = g.yaw_q[2], w = g.yaw_q[3];
    const float tx = 2 * (y * t[2] - z * t[1]), ty = 2 * (z * t[0] - x * t[2]), tz = 2 * (x * t[1] - y * t[0]);
    out[0] = t[0] + w * tx + (y * tz - z * ty);
    out[1] = t[1] + w * ty + (z * tx - x * tz);
    out[2] = t[2] + w * tz + (x * ty - y * tx);
}

/* The headset's rumble, 0 (off) to 25. */
void app_vr_vibrate(int strength)
{
    if (g.loaded && sceHmd2SetVibration)
        sceHmd2SetVibration(g.hmd, strength < 0 ? 0 : strength > 25 ? 25 : strength);
}

/* Load address of libSceVrTracker2 (from a bound export), 0 if not bound. */
uintptr_t app_vr_tracker_base(void)
{
    if (!sceVrTracker2GetResult)
        return 0;
    for (size_t i = 0; i < COUNT(klibSceVrTracker2Exports); i++)
        if (!strcmp(klibSceVrTracker2Exports[i].name, "sceVrTracker2GetResult"))
            return (uintptr_t)sceVrTracker2GetResult - klibSceVrTracker2Exports[i].offset;
    return 0;
}

/* Hand tracking's camera opens, as libSceVrHand makes them (both are
 * idempotent): which one fails, and how. */
void app_vr_camera_probe(void)
{
    if (!sceHmd2InternalImageOpenBySlot || !sceHmd2InternalLedDetectorOpenBySlot)
        return;
    const int r1 = sceHmd2InternalImageOpenBySlot(2);
    const int r2 = sceHmd2InternalLedDetectorOpenBySlot(1);
    eng_bt("vr: camera image slot 2 -> 0x%08x, LED detector slot 1 -> 0x%08x", r1, r2);
}

int app_vr_port(void)
{
    return g.active ? g.vh : -1;
}

void app_vr_head_dir(float *yaw, float *pitch)
{
    if (yaw) *yaw = g.head_yaw;
    if (pitch) *pitch = g.head_pitch;
}

/* The TV shows what the headset shows (Hmd2's mirror), the way Horizon Call
 * of the Mountain sets it up: option {1}, then its work and display memory. */
static int s_mirror_want, s_mirror_on;
static void *s_mirror_work, *s_mirror_disp;

static void mirror_apply(void)
{
    if (!g.loaded || s_mirror_want == s_mirror_on)
        return;
    if (s_mirror_want) {
        if (!s_mirror_work) {
            SizeAlign w = sceHmd2ReprojectionGetMirroringWorkMemorySizeAlign();
            SizeAlign d = sceHmd2ReprojectionGetMirroringDisplayBufferSizeAlign();
            s_mirror_work = direct_alloc(w.size, w.align > 0x10000 ? w.align : 0x10000);
            s_mirror_disp = direct_alloc(d.size, d.align > 0x10000 ? d.align : 0x10000);
            eng_bt("vr: mirror memory work 0x%lx display 0x%lx", (long)w.size, (long)d.size);
        }
        if (!s_mirror_work || !s_mirror_disp)
            return;
        uint32_t opt[8] = {1, 0, 0, 0, 0, 0, 0, 0};
        int r1 = sceHmd2ReprojectionSetMirroringOption(opt);
        void *param[2] = {s_mirror_work, s_mirror_disp};
        int r2 = sceHmd2ReprojectionEnableMirroring(param);
        eng_bt("vr: TV mirror on: option 0x%08x enable 0x%08x", r1, r2);
        s_mirror_on = r2 == 0;
    } else {
        int r = sceHmd2ReprojectionDisableMirroring();
        eng_bt("vr: TV mirror off: 0x%08x", r);
        s_mirror_on = 0;
    }
}

void app_vr_set_mirror(int on)
{
    s_mirror_want = on != 0;
    mirror_apply();
}

int app_vr_worn(void)
{
    if (!g.loaded)
        return 1;
    const uint64_t now = sceKernelGetProcessTime();
    if (now - g.worn_checked >= 500000) {
        g.worn_checked = now;
        uint8_t info[64];
        memset(info, 0, sizeof info);
        if (sceHmd2GetDeviceInformation(info) == 0) {
            const int w = info[20] != 0;
            if (w != g.worn)
                eng_bt("vr: headset %s", w ? "put on" : "taken off");
            g.worn = w;
        }
    }
    return g.worn;
}

/* Another system library (hand tracking; ShellCoreUtil's exit): loaded from the
 * sandbox's common/lib and bound by this firmware's export offsets. */
int app_vr_bind(const char *lib, const char *const *names, void **slots, int n)
{
    static const struct {
        const char *lib;
        const export_sig_t *table;
        size_t count;
        const export_anchor_t *anchor;
    } kLibs[] = {
        {"libSceVrHand", klibSceVrHandExports, COUNT(klibSceVrHandExports), &klibSceVrHandAnchor},
        {"libSceSystemService", klibSceSystemServiceExports, COUNT(klibSceSystemServiceExports),
         &klibSceSystemServiceAnchor},
    };
    for (size_t k = 0; k < COUNT(kLibs); k++) {
        if (strcmp(kLibs[k].lib, lib))
            continue;
        char path[128], file[64];
        snprintf(path, sizeof path, "/%s/common/lib/%s.sprx", sceKernelGetFsSandboxRandomWord(), lib);
        snprintf(file, sizeof file, "%s.sprx", lib);
        int res = 0;
        const int h = sceKernelLoadStartModule(path, 0, NULL, 0, NULL, &res);
        eng_bt("vr: load %s -> 0x%x", lib, h);
        if (h < 0)
            return -1;
        slot_t sl[16];
        if (n > 16)
            return -1;
        for (int i = 0; i < n; i++) {
            sl[i].name = names[i];
            sl[i].slot = &slots[i];
        }
        return resolve_module(h, file, sl, (size_t)n, kLibs[k].table, kLibs[k].count, kLibs[k].anchor);
    }
    return -1;
}

/* The modules load through the sandbox's /<random word>/common/lib path, so
 * this has to run before the process is given the real root. */
int app_vr_preload(void)
{
    static int done = 0;
    if (done)
        return done > 0 ? 0 : -1;
    done = -1;
    int h = load_sysmodule(SYSMODULE_HMD2, "libSceHmd2");
    if (h < 0 || resolve_module(h, "libSceHmd2.sprx", kHmd2Slots, COUNT(kHmd2Slots),
                                klibSceHmd2Exports, COUNT(klibSceHmd2Exports), &klibSceHmd2Anchor))
        return -1;
    h = load_sysmodule(SYSMODULE_VRTRACKER2, "libSceVrTracker2");
    if (h < 0 || resolve_module(h, "libSceVrTracker2.sprx", kVrTracker2Slots,
                                COUNT(kVrTracker2Slots), klibSceVrTracker2Exports,
                                COUNT(klibSceVrTracker2Exports), &klibSceVrTracker2Anchor))
        return -1;
    done = 1;
    vr_system_preload();
    return 0;
}

static int setup(int user_id)
{
    if (g.loaded)
        return 0;
    if (g.failed)
        return -1;
    g.failed = 1;
    if (app_vr_preload() != 0)
        return -1;

    uint8_t init[16] = {0};
    int rc = sceHmd2Initialize(init);
    eng_bt("vr: sceHmd2Initialize -> 0x%08x", rc);
    if (rc != 0)
        return -1;
    float fov[16];
    memset(fov, 0, sizeof fov);
    rc = sceHmd2GetFieldOfViewWithoutHandle(fov);
    memcpy(g.fov, fov, sizeof g.fov);
    eng_bt("vr: FOV rc 0x%08x out %.4f in %.4f top %.4f bottom %.4f", rc, (double)fov[0],
           (double)fov[1], (double)fov[2], (double)fov[3]);
    if (rc != 0 || g.fov[0] <= 0.0f || g.fov[2] <= 0.0f)
        return -1;
    g.hmd = sceHmd2Open(user_id, 0, 0, NULL);
    uint8_t info[64];
    memset(info, 0, sizeof info);
    rc = sceHmd2GetDeviceInformation(info);
    eng_bt("vr: sceHmd2Open -> 0x%08x; device status %u worn %u lens %u mm", g.hmd,
           *(uint32_t *)info, info[20], info[21]);
    if (g.hmd < 0)
        return -1;

    /* Tracker (profile 0, the only one open to apps). */
    uint32_t qparam[8] = {0x20, 0};
    uint32_t qout[8] = {0};
    if (sceVrTracker2QueryMemory(qparam, qout) != 0)
        return -1;
    void *twork = direct_alloc(qout[1], qout[2] > 0x10000 ? qout[2] : 0x10000);
    if (!twork)
        return -1;
    uint8_t ip[0x48];
    memset(ip, 0, sizeof ip);
    *(uint32_t *)(ip + 0x00) = 0x48;
    *(int32_t *)(ip + 0x08) = 700;
    *(int32_t *)(ip + 0x0c) = 700;
    *(uint64_t *)(ip + 0x10) = 0x1f00;
    *(uint64_t *)(ip + 0x18) = 0x1f00;
    *(void **)(ip + 0x20) = twork;
    *(uint32_t *)(ip + 0x28) = qout[1];
    *(uint32_t *)(ip + 0x2c) = qout[2];
    rc = sceVrTracker2Initialize(ip);
    int rc2 = rc == 0 ? sceVrTracker2RegisterDevice(0, g.hmd) : -1;
    eng_bt("vr: tracker init 0x%08x register 0x%08x", rc, rc2);
    if (rc != 0 || rc2 != 0)
        return -1;

    /* Reprojection (once per process). */
    SizeAlign wb = sceHmd2ReprojectionQueryBufferSizeAlign();
    SizeAlign db = sceHmd2ReprojectionQueryDisplayBufferSizeAlign();
    void *work = direct_alloc(wb.size, 0x10000);
    void *disp = direct_alloc(db.size, 0x10000);
    if (!work || !disp)
        return -1;
    uint8_t rp[72];
    memset(rp, 0, sizeof rp);
    *(void **)(rp + 0x00) = work;
    *(void **)(rp + 0x08) = disp;
    *(uint32_t *)(rp + 0x10) = 256;
    *(int32_t *)(rp + 0x14) = 0x1f00;
    *(uint32_t *)(rp + 0x18) = 1;
    rc = sceHmd2ReprojectionInitialize(rp, NULL);
    eng_bt("vr: sceHmd2ReprojectionInitialize -> 0x%08x", rc);
    if (rc != 0)
        return -1;
    g.loaded = 1;
    g.failed = 0;
    return 0;
}

/* Tracker pose at `time`: device position + quat, eye quats (x,y,z,w). */
static int pose(uint64_t time, float dp[3], float dq[4], float lq[4], float rq[4])
{
    static uint8_t res[0x400];
    uint8_t gp[0x20];
    memset(gp, 0, sizeof gp);
    *(uint32_t *)(gp + 0x00) = 0x20;
    *(int32_t *)(gp + 0x04) = g.hmd;
    *(uint64_t *)(gp + 0x10) = time;
    memset(res, 0, sizeof res);
    if (sceVrTracker2GetResult(gp, res) != 0)
        return 0;
    memcpy(dp, res + 0x60, 12);
    memcpy(dq, res + 0x70, 16);
    if (lq)
        memcpy(lq, res + 0xb0, 16);
    if (rq)
        memcpy(rq, res + 0xf0, 16);
    return *(uint32_t *)(res + 4) != 0;
}

static void qmul(const float a[4], const float b[4], float o[4])
{
    float r[4] = {
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    };
    memcpy(o, r, sizeof r);
}

void app_vr_recenter(void)
{
    if (!g.loaded)
        return;
    float dp[3], dq[4];
    if (!pose(0, dp, dq, NULL, NULL))
        return;
    /* forward = R(q) * (0,0,-1); yaw a to the right; video space = Ry(a) * tracker. */
    const float x = dq[0], y = dq[1], z = dq[2], w = dq[3];
    const float fx = -(2 * (x * z + y * w)), fz = -(1 - 2 * (x * x + y * y));
    const float a = atan2f(fx, -fz);
    g.yaw_q[0] = 0.0f;
    g.yaw_q[1] = sinf(a * 0.5f);
    g.yaw_q[2] = 0.0f;
    g.yaw_q[3] = cosf(a * 0.5f);
    eng_bt("vr: recentred, yaw %.1f deg", (double)(a * 57.29578f));
}

/* Hmd2 SetParam + flip, called by the engine after the frame's GPU fence. */
static void submit(void)
{
    if (!g.frame_open)
        return;
    g.frame_open = 0;
    static uint8_t lp[0x148] __attribute__((aligned(16)));
    struct { void *layers; int32_t count; int32_t pad; void *ext; } param = {lp, 1, 0, NULL};
    const uint32_t tw = 2 * EYE_W, th = EYE_H;
    float tl[4] = {g.fov[0] * OVERSCAN, g.fov[1] * OVERSCAN, g.fov[2] * OVERSCAN, g.fov[3] * OVERSCAN};
    float tr[4] = {tl[1], tl[0], tl[2], tl[3]};
    memset(lp, 0, sizeof lp);
    uint32_t *u = (uint32_t *)lp;
    u[0x00 / 4] = 0x10;
    u[0x08 / 4] = tw, u[0x0c / 4] = th, u[0x10 / 4] = tw, u[0x14 / 4] = th;
    u[0x18 / 4] = 0x92, u[0x1c / 4] = 0x00fff000, u[0x20 / 4] = 0x05500000;   /* sampler */
    u[0x28 / 4] = EYE_W, u[0x2c / 4] = EYE_H;                                   /* left half */
    u[0x38 / 4] = EYE_W, u[0x3c / 4] = EYE_H, u[0x40 / 4] = EYE_W;             /* right half */
    memcpy(lp + 0x48, tl, 16);
    memcpy(lp + 0x58, tr, 16);
    memcpy(lp + 0x68, g.pd, 12);
    memcpy(lp + 0x74, g.qd, 16);
    *(uint64_t *)(lp + 0x88) = g.t;
    *(uint64_t *)(lp + 0x90) = g.frame;
    u[0x98 / 4] = 100;
    u[0xa0 / 4] = EYE_W, u[0xa4 / 4] = EYE_H, u[0xa8 / 4] = EYE_W, u[0xac / 4] = EYE_H;
    int r1 = sceHmd2ReprojectionSetParam((uint32_t)g.buf, g.frame, &param, NULL);
    int r2 = sceVideoOutSubmitFlip(g.vh, g.buf, 2, (int64_t)g.frame);
    if ((r1 || r2) && g.errors++ < 20)
        eng_bt("vr: frame %lu setparam 0x%08x flip 0x%08x", (unsigned long)g.frame, r1, r2);
    if (g.submits++ % 900 == 0)
        eng_bt("vr: frame %lu submitted (buf %d)", (unsigned long)g.frame, g.buf);
    g.buf = (g.buf + 1) % NUM_BUFFERS;
}

int app_vr_start(const app_vr_format *fmt, int user_id)
{
    if (g.active)
        return 0;
    if (setup(user_id) != 0) {
        eng_bt("vr: headset setup failed");
        return -1;
    }
    g.eye = (g.opt.eye_px >= 1024 && g.opt.eye_px <= 4096) ? ((uint32_t)g.opt.eye_px + 127u) & ~127u
                                                            : EYE_DEFAULT;
    uint32_t tsharp[NUM_BUFFERS][8];
    g.user = user_id;
    /* bits: 8, 10, or 16 = RGBA16F (HDR, experimental) */
    if (eng_agc_vr_targets(EYE_W, EYE_H, NUM_BUFFERS, tsharp,
                           g.opt.bits == 16 ? 2 : g.opt.bits != 8) != 0) {
        eng_bt("vr: eye buffers failed");
        return -1;
    }
    mirror_apply();                         /* before VR mode, as Horizon does */
    int rc = sceHmd2ReprojectionEnableVrMode(VR_MODE_120HZ);
    g.vh = sceVideoOutOpen(0xff, 32, 0, NULL);
    eng_bt("vr: EnableVrMode -> 0x%08x, VR port 0x%08x", rc, g.vh);
    if (g.vh < 0) {
        sceHmd2ReprojectionDisableVrMode();
        return -1;
    }
    vo_buffer_t bufs[NUM_BUFFERS];
    memset(bufs, 0, sizeof bufs);
    memset(g.layers, 0, sizeof g.layers);
    for (int b = 0; b < NUM_BUFFERS; b++) {
        *(uint32_t *)g.layers[b] = 0x10;                 /* same texture twice */
        memcpy(g.layers[b] + 0x04, tsharp[b], 32);
        memcpy(g.layers[b] + 0x24, tsharp[b], 32);
        g.sets[b].layers = g.layers[b];
        g.sets[b].count = 1;
        bufs[b].data = &g.sets[b];
    }
    vo_attribute_t attr;
    memset(&attr, 0, sizeof attr);
    sceVideoOutSetBufferAttribute2(&attr, 0x8000000000000000ull, 0, 2 * EYE_W, EYE_H, 0, 0, 0);
    rc = sceVideoOutRegisterBuffers2(g.vh, 0, 0, bufs, NUM_BUFFERS, &attr, 0x10, NULL);
    eng_bt("vr: RegisterBuffers2 -> 0x%08x", rc);
    if (rc == 0) {
        int c1 = sceVideoOutConfigureOutput(g.vh, VR_MODE_120HZ, NULL, NULL, 0);
        int c2 = sceHmd2ReprojectionSetRenderConfig(1, 0, 0);
        eng_bt("vr: 120 Hz output 0x%08x, half-rate render config 0x%08x", c1, c2);
    }
    if (rc != 0) {
        sceVideoOutClose(g.vh);
        g.vh = -1;
        sceHmd2ReprojectionDisableVrMode();
        return -1;
    }
    if (fmt)
        g.fmt = *fmt;
    else
        g.fmt.projection = g.fmt.stereo = 0;
    g.buf = 0;
    g.errors = 0;
    g.frame_open = 0;
    eng_agc_vr_reset_picture();
    eng_agc_vr_set_submit(submit);
    /* The video's front faces the tracking space's forward (-z), which is where
     * the system's own VR interface is centred (home menu, cinematic screen).
     * No recentring on the head here: the tracker's first poses after it
     * registers are a stale default (always yaw 90.5 deg), not the head. */
    g.yaw_q[0] = g.yaw_q[1] = g.yaw_q[2] = 0.0f;
    g.yaw_q[3] = 1.0f;
    g.active = 1;
    eng_bt("vr: started, projection %d stereo %d, eye %u px %d-bit, sharpen %d",
           g.fmt.projection, g.fmt.stereo, g.eye, g.opt.bits == 16 ? 16 : g.opt.bits != 8 ? 10 : 8,
           g.opt.sharpen);
    return 0;
}

void app_vr_stop(void)
{
    if (!g.active)
        return;
    g.active = 0;
    g.frame_open = 0;
    eng_agc_vr_set_view(NULL);
    {
        static const float ident[4] = {0, 0, 0, 1};
        eng_vr_audio_set_head(ident, 0);
    }
    /* Leave VR mode first: the reprojection keeps sampling the last buffer, so
     * the buffers stay busy until it stops. */
    int rc3 = sceHmd2ReprojectionDisableVrMode();
    int rc = 0;
    for (int i = 0; i < 50; i++) {
        rc = sceVideoOutUnregisterBuffers(g.vh, 0);
        if (rc != VO_BUSY)
            break;
        sceKernelUsleep(10000);
    }
    int rc2 = sceVideoOutClose(g.vh);
    g.vh = -1;
    g.next_begin = 0;
    eng_bt("vr: stopped after %u frames (unregister 0x%08x close 0x%08x disable 0x%08x)",
           g.submits, rc, rc2, rc3);
}

/* Before the app ends: the libraries' own exit code faults on a live tracker
 * (SIGBUS in libSceVrTracker2's destructor), so take everything down first. */
void app_vr_shutdown(void)
{
    app_vr_stop();
    vr_system_shutdown();
    if (!g.loaded)
        return;
    const int r1 = sceVrTracker2UnregisterDevice(g.hmd);
    const int r2 = sceVrTracker2Finalize();
    const int r3 = sceHmd2ReprojectionTerminate();
    const int r4 = sceHmd2Close(g.hmd);
    const int r5 = sceHmd2Terminate();
    g.loaded = 0;
    eng_bt("vr: shut down (tracker 0x%08x 0x%08x, reprojection 0x%08x, hmd 0x%08x 0x%08x)", r1, r2,
           r3, r4, r5);
}

/* New eye buffers (their format follows the options): the session again. */
int app_vr_restart(void)
{
    if (!g.active)
        return -1;
    const app_vr_format f = g.fmt;
    app_vr_stop();
    return app_vr_start(&f, g.user);
}

int app_vr_active(void)
{
    return g.active;
}

void app_vr_wait_slot(void)
{
    if (!g.active)
        return;
    /* A fixed schedule, so the rate does not drift below 60 Hz; after a stall
     * it restarts from now. */
    uint64_t now = sceKernelGetProcessTime();
    if (g.next_begin > now && g.next_begin - now <= FRAME_US) {
        sceKernelUsleep((unsigned)(g.next_begin - now));
        now = g.next_begin;
    }
    g.next_begin = (g.next_begin && now < g.next_begin + FRAME_US) ? g.next_begin + FRAME_US
                                                                   : now + FRAME_US;
}

int app_vr_begin_frame(void)
{
    if (!g.active)
        return -1;
    vr_system_tick();
    if (g.frame_open)       /* the last frame was never submitted: reuse it */
        return 0;
    const uint64_t frame = g.frame + 1;
    int rc = sceHmd2ReprojectionBeginFrame(frame);
    for (int tries = 0; rc == VR_QUEUE_FULL && tries < 40; tries++) {
        sceKernelUsleep(1000);
        rc = sceHmd2ReprojectionBeginFrame(frame);
    }
    if (rc != 0) {
        if (g.errors++ < 20)
            eng_bt("vr: BeginFrame(%lu) -> 0x%08x", (unsigned long)frame, rc);
        return -1;
    }
    g.frame = frame;
    g.t = 0;
    sceHmd2ReprojectionGetPredictedDisplayTime(frame, &g.t);
    float lq[4] = {0, 0, 0, 1}, rq[4] = {0, 0, 0, 1};
    g.qd[0] = g.qd[1] = g.qd[2] = 0.0f;
    g.qd[3] = 1.0f;
    g.pd[0] = g.pd[1] = g.pd[2] = 0.0f;
    const int tracked = pose(g.t, g.pd, g.qd, lq, rq);
    {
        float head[4];
        qmul(g.yaw_q, g.qd, head);
        eng_vr_audio_set_head(head, tracked);     /* the sound turns with the picture */
        /* forward = R(head) * (0,0,-1): yaw clockwise, pitch up */
        const float x = head[0], y = head[1], z = head[2], w = head[3];
        const float fx = -2 * (x * z + y * w), fy = -2 * (y * z - x * w), fz = -(1 - 2 * (x * x + y * y));
        g.head_yaw = atan2f(fx, -fz) * 57.29578f;
        g.head_pitch = asinf(fy < -1 ? -1 : fy > 1 ? 1 : fy) * 57.29578f;
    }

    eng_agc_vr_view_t v;
    memset(&v, 0, sizeof v);
    v.buffer = g.buf;
    qmul(g.yaw_q, lq, v.q[0]);
    qmul(g.yaw_q, rq, v.q[1]);
    for (int i = 0; i < 4; i++) {
        v.tan[0][i] = g.fov[i] * OVERSCAN;
        v.tan[1][i] = g.fov[i] * OVERSCAN;
    }
    v.tan[1][0] = g.fov[1] * OVERSCAN;     /* right eye: in to the left, out to the right */
    v.tan[1][1] = g.fov[0] * OVERSCAN;
    v.projection = g.fmt.projection;
    v.stereo = g.fmt.stereo;
    v.screen_deg = g.opt.screen_deg > 10.0f ? g.opt.screen_deg : 70.0f;
    v.sharpen = g.opt.sharpen;
    v.swap = g.opt.swap;
    eng_agc_vr_set_view(&v);
    g.frame_open = 1;
    return 0;
}

#else /* !APP_VR */

int app_vr_preload(void) { return -1; }
int app_vr_start(const app_vr_format *fmt, int user_id) { (void)fmt; (void)user_id; return -1; }
void app_vr_stop(void) {}
void app_vr_shutdown(void) {}
int app_vr_restart(void) { return -1; }
void app_vr_set_options(const app_vr_options *o) { (void)o; }
void app_vr_get_options(app_vr_options *o) { (void)o; }
void app_vr_set_format(const app_vr_format *f) { (void)f; }
void app_vr_get_format(app_vr_format *f) { (void)f; }
void app_vr_head_dir(float *y, float *p) { if (y) *y = 0; if (p) *p = 0; }
uint64_t app_vr_frame_time(void) { return 0; }
void app_vr_camera_probe(void) {}
uintptr_t app_vr_tracker_base(void) { return 0; }
int app_vr_gaze(void *result) { (void)result; return -1; }
void app_vr_to_view(const float p[3], float out[3]) { out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; }
void app_vr_vibrate(int strength) { (void)strength; }
int app_vr_gaze_dir(float dir[3]) { (void)dir; return 0; }
void app_vr_head_quat(float q[4]) { q[0] = q[1] = q[2] = 0; q[3] = 1; }
int app_vr_controller(int side, float pos[3], float q[4], uint32_t *buttons, float stick[2]) { (void)side; (void)pos; (void)q; (void)buttons; (void)stick; return 0; }
void app_vr_head_pose(float pos[3], float yaw_q[4]) { pos[0] = pos[1] = pos[2] = 0; yaw_q[0] = yaw_q[1] = yaw_q[2] = 0; yaw_q[3] = 1; }
int app_vr_port(void) { return -1; }
int app_vr_worn(void) { return 1; }
void app_vr_set_mirror(int on) { (void)on; }
void app_vr_wait_slot(void) {}
int app_vr_active(void) { return 0; }
int app_vr_begin_frame(void) { return -1; }
void app_vr_recenter(void) {}

#endif /* APP_VR */

/* -- format names (both builds: harmless, and testable on the Mac) ---------- */
/* Own case-insensitive helpers: the console does not export strcasestr. */
static int ieq(const char *a, const char *b, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i]))
            return 0;
    return 1;
}

static int has_token(const char *s, const char *tok)
{
    /* tokens are runs of letters/digits; "360tb" also matches "360" + "tb" below */
    const size_t n = strlen(tok);
    for (const char *p = s; *p; ) {
        while (*p && !isalnum((unsigned char)*p))
            p++;
        const char *q = p;
        while (*q && isalnum((unsigned char)*q))
            q++;
        if ((size_t)(q - p) == n && ieq(p, tok, n))
            return 1;
        p = q;
    }
    return 0;
}

static int contains(const char *s, const char *sub)
{
    const size_t n = strlen(sub);
    for (; *s; s++)
        if (ieq(s, sub, n))
            return 1;
    return 0;
}

static int stereo_of(const char *s)
{
    if (has_token(s, "tb") || has_token(s, "ou") || has_token(s, "3dv") || has_token(s, "hou") ||
        has_token(s, "fou") || has_token(s, "tab") || has_token(s, "htab") ||
        contains(s, "half-ou") || contains(s, "half.ou") || contains(s, "h-ou") ||
        contains(s, "topbottom") || contains(s, "top_bottom") || contains(s, "overunder") ||
        contains(s, "360tb") || contains(s, "180tb"))
        return 2;
    if (has_token(s, "sbs") || has_token(s, "lr") || has_token(s, "3dh") || has_token(s, "hsbs") ||
        has_token(s, "fsbs") || contains(s, "half-sbs") || contains(s, "half.sbs") ||
        contains(s, "h-sbs") ||
        contains(s, "sidebyside") || contains(s, "side_by_side") || contains(s, "360sbs") ||
        contains(s, "180sbs") || contains(s, "360lr") || contains(s, "180lr"))
        return 1;
    return 0;
}

int app_vr_parse(const char *s, app_vr_format *out)
{
    if (!s || !*s || !out)
        return 0;
    app_vr_format f = {0, 0};
    if (contains(s, "360"))
        f.projection = 360;
    else if (contains(s, "180"))
        f.projection = 180;
    else if (contains(s, "flat") || contains(s, "screen") || contains(s, "3d"))
        f.projection = 0;
    else
        return 0;
    f.stereo = stereo_of(s);
    *out = f;
    return 1;
}

void app_vr_fit_frame(app_vr_format *f, int w, int h)
{
    if (!f || w <= 0 || h <= 0 || (f->projection != 360 && f->projection != 180))
        return;
    const float a = (float)w / (float)h;
    /* frame aspect of each layout: [mono, sbs, tb] (full-size eyes; half-size
     * SBS/TB packs land on the mono shape and are handled by the name) */
    const float k360[3] = {2.0f, 4.0f, 1.0f}, k180[3] = {1.0f, 2.0f, 0.5f};
    const float *k = f->projection == 360 ? k360 : k180;
    int best = -1;
    float err = 1e9f;
    for (int i = 0; i < 3; i++) {
        const float e = a > k[i] ? a / k[i] : k[i] / a;
        if (e < err) {
            err = e;
            best = i;
        }
    }
    const int st = f->stereo >= 0 && f->stereo <= 2 ? f->stereo : 0;
    const float named = a > k[st] ? a / k[st] : k[st] / a;
    if (named <= 1.15f)                     /* the name's layout fits: keep it */
        return;
    if (err <= 1.15f) {                     /* another layout fits */
        f->stereo = best;
    } else {                                /* nothing equirect-shaped */
        f->projection = 0;
        f->stereo = 0;
    }
}

int app_vr_guess(const char *path, app_vr_format *out)
{
    if (!path || !*path || !out)
        return 0;
    /* A URL: only its file name counts (not the folders, host or query). */
    char base[512];
    const char *name = path;
    if (strstr(path, "://")) {
        size_t n = strcspn(path, "?#");
        const char *end = path + n, *b = end;
        while (b > path && b[-1] != '/')
            b--;
        n = (size_t)(end - b) < sizeof base - 1 ? (size_t)(end - b) : sizeof base - 1;
        memcpy(base, b, n);
        base[n] = 0;
        name = base;
    }
    app_vr_format f = {0, 0};
    if (has_token(name, "360") || has_token(name, "vr360") || contains(name, "360tb") ||
        contains(name, "360sbs") || contains(name, "360lr") || contains(name, "_360_"))
        f.projection = 360;
    else if (has_token(name, "180") || has_token(name, "vr180") || contains(name, "180sbs") ||
             contains(name, "180lr") || contains(name, "180x180"))
        f.projection = 180;
    else if (has_token(name, "oculusrift") || has_token(name, "oculus") || has_token(name, "gearvr") ||
             has_token(name, "psvr") || has_token(name, "vr") || has_token(name, "3dvr"))
        f.projection = 180;      /* VR releases named for a headset are 180 SBS */
    else if (has_token(name, "3d") || stereo_of(name)) {
        f.projection = 0;        /* a 3D movie: flat screen, side by side unless it says */
        f.stereo = stereo_of(name);
        if (!f.stereo)
            f.stereo = 1;
        *out = f;
        return 1;
    } else
        return 0;
    f.stereo = stereo_of(name);
    if (f.projection == 180 && f.stereo == 0 && !has_token(name, "mono"))
        f.stereo = 1;            /* 180 video is stereo side by side unless it says */
    *out = f;
    return 1;
}
