/*
 * PS5VR (PPSA99177) - a video player for PlayStation VR2.
 *
 * Plays 180/360 VR video, 3D movies (side by side, over-under, MVC Blu-ray,
 * MV-HEVC spatial video) and flat video on a virtual screen, in the headset.
 * Its interface is its own; while it is being built, videos are queued
 * through the PS5 service's player mailbox (ps5vr/tools/vrtest.sh play).
 * The playback engine is the engine's (engine/, see engine/LICENSE.the engine).
 */
#include "eng_jailbreak.h"
#include "app_bridge.h"
#include "app_control.h"
#include "app_input.h"
#include "app_player.h"
#include "app_vr.h"
#include "ui_canvas.h"
#include "ui_text.h"
#include "vr_ui.h"
#include "vr_system.h"
#include "donate_qr.h"

#include <string>
#include <ctime>

#include "eng_adec.h"
#include "eng_agc_runtime.h"
#include "eng_boot_log.h"
#include "eng_boot_trace.h"
#include "eng_direct_mem.h"
#include "eng_hw.h"
#include "eng_vdec.h"
#include "pp_playback.h"

#include <arpa/inet.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <signal.h>
#include <sys/ucontext.h>
#include <sys/socket.h>
#include <unistd.h>

extern "C" {
#include <libavutil/cpu.h>
#include <libavutil/log.h>

int sceUserServiceInitialize(void *params);
int sceUserServiceGetLoginUserIdList(int user_ids[4]);
int scePadInit(void);
int sceKernelSendNotificationRequest(int, void *, unsigned long, int);
int sceSystemServiceHideSplashScreen(void);
extern int g_ps5_user_id;
extern pp_playback g_pp_pb;
}

namespace engine {
extern int DisplayWidth;
extern int DisplayHeight;
}

namespace {

/* 128 MiB of direct memory for decode surfaces, as the engine sizes it. */
constexpr size_t kDirectMemPool = 128u * 1024u * 1024u;

void notify(const char *text)
{
    struct { char pad[45]; char msg[3075]; } n;
    std::memset(&n, 0, sizeof n);
    std::snprintf(n.msg, sizeof n.msg, "%s", text);
    sceKernelSendNotificationRequest(0, &n, sizeof n, 0);
}

void av_log_to_boot_log(void *, int level, const char *fmt, va_list vl)
{
    if (level > AV_LOG_WARNING)
        return;
    char line[512];
    int n = std::vsnprintf(line, sizeof line, fmt, vl);
    if (n <= 0)
        return;
    if (n >= (int)sizeof line)
        n = (int)sizeof line - 1;
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r'))
        line[--n] = '\0';
    if (n)
        eng_bt("ffmpeg[%d]: %s", level, line);
}

/*
 * A crash report straight to the payload's log (GET /api/logs): most
 * consoles have no USB stick for engine.log. Only async-signal-safe calls.
 */
void crash_handler(int sig, siginfo_t *si, void *)
{
    char body[160];
    int len = std::snprintf(body, sizeof body, "[engine] CRASH signal=%d addr=%p\n", sig,
                            si ? si->si_addr : nullptr);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd >= 0 && len > 0) {
        struct sockaddr_in a;
        std::memset(&a, 0, sizeof a);
        a.sin_family = AF_INET;
        a.sin_port = htons(APP_SERVICE_PORT);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, (struct sockaddr *)&a, sizeof a) == 0) {
            char head[128];
            int hl = std::snprintf(head, sizeof head,
                                   "POST /api/log HTTP/1.0\r\nContent-Length: %d\r\n\r\n", len);
            (void)write(fd, head, (size_t)hl);
            (void)write(fd, body, (size_t)len);
        }
        close(fd);
    }
    /* Not _exit(): that faults in an app process (SIGSYS) and the kernel then
     * reports the exit, not the crash. With the default action back, the
     * faulting instruction runs again and the kernel's crash report (klog:
     * registers, backtrace, libraries) describes the real fault. */
    signal(sig, SIG_DFL);
}

extern "C" __attribute__((noinline, used)) void main_entry_marker(void) {}

void install_crash_handler()
{
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
}

/* Display, decoders and controller, in the engine's order: the decoder probes
 * and module loads run before anything else touches the process. */
int init_hardware()
{
    eng_vdec_probe();
    eng_adec_native_probe();
    eng_hw_probe();

    if (eng_agc_runtime_init(engine::DisplayWidth, engine::DisplayHeight, 0) != 0) {
        eng_bt("ps5vr: display init failed");
        return -1;
    }
    eng_agc_runtime_get_size(&engine::DisplayWidth, &engine::DisplayHeight);
    eng_agc_runtime_frame_begin();
    eng_agc_runtime_present();
    eng_bt("ps5vr: display %dx%d", engine::DisplayWidth, engine::DisplayHeight);

    /* A game-category app keeps the system's launch splash (sce_sys/pic0,
     * the PS5VR art) over everything - the browser dialog included - until it
     * says it is ready. Once a frame is up, it is. */
    eng_bt("ps5vr: hide splash -> %d", sceSystemServiceHideSplashScreen());

    av_log_set_callback(av_log_to_boot_log);
    av_log_set_level(AV_LOG_WARNING);
    av_force_cpu_flags(0);
    eng_direct_mem_init(kDirectMemPool);

    sceUserServiceInitialize(nullptr);
    scePadInit();
    int users[4] = {0};
    sceUserServiceGetLoginUserIdList(users);
    g_ps5_user_id = users[0];
    eng_bt("ps5vr: user %d", users[0]);
    app_player_init(users[0]);
    return 0;
}

/* The PS5VR PS5 service is a development aid for PS5VR (logs, test requests):
 * look for it briefly, never wait on it. */
void wait_for_service()
{
    for (int i = 0; i < 20 && !app_service_up(); i++)
        usleep(100 * 1000);
    eng_bt("ps5vr: dev service %s", app_service_up() ? "found" : "absent");
    return;
    bool warned = false;
    while (!app_service_up()) {
        if (!warned) {
            warned = true;
            notify("PS5VR: waiting for the PS5 service\n"
                   "Load the dev payload, or add it to your autoloader");
        }
        usleep(1 * 1000 * 1000);
    }
}

/* A "quit" command: leave the way the engine's QUIT does - playback already
 * stopped, the GPU drained and VideoOut closed before the process ends, so
 * the kernel never reclaims memory the GPU might still be using. */
/* A "quit": playback stopped, the headset libraries down, the GPU drained and
 * VideoOut closed - then the system closes the app as if from the PS menu.
 * Returning from main instead runs the libraries' exit code, which faults. */
int shutdown_app()
{
    eng_bt("ps5vr: quit");
    pp_playback_shutdown(&g_pp_pb);
    app_vr_shutdown();
    eng_agc_runtime_shutdown();
    eng_bt("ps5vr: shut down");
    eng_boot_log_flush();
    usleep(1000 * 1000);   /* the log thread posts every 500 ms */
    const int rc = vr_system_exit_app();
    eng_bt("ps5vr: exit app -> 0x%08x", rc);
    eng_boot_log_flush();
    for (int i = 0; rc == 0 && i < 100; i++)
        usleep(100 * 1000);                     /* ShellUI ends the process */
    /* Not answered: end without the exit code (everything is down already). */
    kill(getpid(), SIGKILL);
    return 0;
}

/* Imports the console left NULL (see scripts/build.sh). The weak stand-ins
 * cover the first link pass, before the generated table exists. */
extern "C" __attribute__((weak)) int app_import_count(void) { return 0; }
extern "C" __attribute__((weak)) const char *app_import_null(int) { return nullptr; }

void check_imports()
{
    int missing = 0;
    for (int i = 0; i < app_import_count(); i++)
        if (const char *name = app_import_null(i)) {
            eng_bt("ps5vr: import %s is NULL on this console", name);
            missing++;
        }
    eng_bt("ps5vr: %d imports, %d NULL", app_import_count(), missing);
}

std::string json_escape(const std::string &in)
{
    std::string o;
    for (char ch : in) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += ch; }
        else if ((unsigned char)ch < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
        else o += ch;
    }
    return o;
}

/* The TV while nothing plays: what PS5VR is and what to do. */
void draw_home(const char *line)
{
    static ui_canvas c;
    static bool ready = c.px || (ui_canvas_init(&c, 1920, 1080) == 0 && ui_text_init() == 0);
    if (!ready)
        return;
    const float stops[2] = {0.0f, 1.0f};
    const ui_color cols[2] = {UI_RGBA(10, 14, 28, 255), UI_RGBA(4, 6, 12, 255)};
    ui_fill_vgradient(&c, 0, 0, 1920, 1080, stops, cols, 2);
    /* the logo: a PlayStation-blue block, "VR" in white (sce_sys/icon0.png) */
    ui_fill_rrect(&c, 960 - 130, 230, 260, 260, 26, UI_RGBA(0x00, 0x70, 0xd1, 255));
    ui_text_draw_aligned(&c, UI_BOLD, 136.0f, 960, 410, 1, UI_WHITE, "VR", 0);
    ui_text_draw_aligned(&c, UI_BOLD, 64.0f, 960, 600, 1, UI_WHITE, "PS5VR", 0);
    ui_text_draw_aligned(&c, UI_MEDIUM, 36.0f, 960, 760, 1, UI_RGBA(120, 200, 255, 255), line, 0);
    /* Optional support, out of the way: a small, dimmed QR code in the corner
     * (buymeacoffee.com/theghostonline). Light card so phones can read it. */
    {
        const int m = 4, q = 2;                     /* module px, quiet modules */
        const int side = (kDonateQrSize + 2 * q) * m;
        const int x0 = 1920 - 72 - side, y0 = 1080 - 112 - side;
        ui_fill_rrect(&c, x0, y0, side, side, 8, UI_RGBA(200, 208, 222, 255));
        for (int y = 0; y < kDonateQrSize; y++)
            for (int x = 0; x < kDonateQrSize; x++)
                if (kDonateQr[y][x] == '1')
                    ui_fill_rect(&c, x0 + (q + x) * m, y0 + (q + y) * m, m, m, UI_RGBA(10, 14, 28, 255));
        ui_text_draw_aligned(&c, UI_MEDIUM, 24.0f, x0 + side / 2.0f, y0 + side + 36, 1,
                             UI_RGBA(120, 132, 155, 255), "Support PS5VR", 0);
    }
    eng_agc_runtime_frame_begin();
    eng_agc_runtime_clear_black();
    eng_agc_composite_overlay(1, c.px, 1920, 1080, 1, 1.0f);
    eng_agc_runtime_present();
}

} // namespace

int main()
{
    install_crash_handler();
    eng_bt("ps5vr: app start");
    check_imports();
    if (init_hardware() != 0) {
        notify("PS5VR: the display could not be started");
        for (;;)
            usleep(1 * 1000 * 1000);
    }
    /* /data and USB drives: ps5vr-helper (or etaHEN's jailbreak daemon) opens
     * the sandbox - after every system module is loaded, because they load
     * through the sandbox's own path. */
    app_vr_preload();
    eng_bt("ps5vr: sandbox %s", eng_jailbreak_self() ? "open" : "closed (retrying)");
    wait_for_service();
    app_control_start();
    vr_settings_server_start();                     /* http://<console>:8090/ */

    static ui_canvas lib_canvas;
    ui_canvas_init(&lib_canvas, VR_UI_W, VR_UI_H);
    VrLibrary library;
    library.open();
    VrHandPointer hand;
    bool hands_asked = false;
    vr_settings().control = vr_prefs_control();
    vr_settings().tv_mirror = vr_prefs_mirror();
    app_vr_set_mirror(vr_settings().tv_mirror);        /* applied when the session starts */
    if (vr_prefs_hdr()) {                           /* HDR (experimental): float eye buffers */
        app_vr_options o;
        app_vr_get_options(&o);
        o.bits = 16;
        app_vr_set_options(&o);
    }
    for (;;) {
        draw_home("Put on your PlayStation VR2");
        char *req = nullptr;
        std::string path, title, vr;
        int got = 0, poll = 0;
        double next_vr_try = 0;
        VrPanelPlace place;
        app_input_open(g_ps5_user_id);
        while (!app_control_quit_requested()) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            const double now = (double)ts.tv_sec + ts.tv_nsec / 1e9;
            if (eng_jailbreak_poll())
                library.open();                     /* /data and USB just opened */
            /* The headset: the session runs for the whole app, not per video. */
            if (!app_vr_active() && now >= next_vr_try && app_vr_start(nullptr, g_ps5_user_id) != 0)
                next_vr_try = now + 5.0;
            app_input_state in;
            app_input_poll(&in);
            /* hands: hand tracking runs with the headset when the setting is on */
            if (app_vr_active() && vr_settings().hands_needed() && !hands_asked) {
                vr_hands_request(1);
                hands_asked = true;
            }
            hand.update(app_vr_active() && place.shown ? &place : nullptr, 2.2f, 2.2f,
                        (float)VR_UI_H / (float)VR_UI_W);
            /* Sense controller buttons act like the DualSense's */
            in.pressed |= hand.buttons_pressed &
                          (APP_BTN_CIRCLE | APP_BTN_TRIANGLE | APP_BTN_OPTIONS | APP_BTN_R3 | APP_BTN_L3);
            if (in.pressed & APP_BTN_L3)
                in.pressed |= APP_BTN_R3;
            library.point(hand, in);
            eng_agc_vr_panel_cursor(hand.valid && !hand.gaze, hand.u, hand.v, hand.pinching);
            if (in.pressed & APP_BTN_R3) {
                app_vr_recenter();
                place.shown = false;
            }
            if (library.input(in, path, title, vr) == 1)
                break;
            if ((poll++ % 6) == 0 && (got = app_bridge_next(&req)) == 1)
                break;
            std::string posted;
            if (vr_settings_take_play(posted)) {
                req = strdup(posted.c_str());
                got = 1;
                break;
            }
            if (app_vr_active()) {
                app_vr_wait_slot();
                if (app_vr_begin_frame() == 0) {
                    const bool changed = library.render(lib_canvas);
                    place.update(true);
                    eng_agc_vr_panel(lib_canvas.px, VR_UI_W, VR_UI_H, changed || !place.shown, 1,
                                     place.yaw, place.pitch, 2.2f, 2.2f, 1.0f);
                    eng_agc_vr_draw();
                    eng_agc_runtime_present();
                }
            } else {
                usleep(16 * 1000);
            }
        }
        app_input_close();
        eng_agc_vr_panel(nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        if (app_control_quit_requested()) {
            free(req);
            return shutdown_app();
        }
        if (got != 1) {                             /* chosen in the library */
            std::string json = "{\"url\":\"" + json_escape(path) + "\",\"title\":\"" +
                               json_escape(title) + "\"" +
                               (vr.empty() ? "" : ",\"vr\":\"" + json_escape(vr) + "\"") + "}";
            app_player_run(json.c_str());
        } else {
            app_player_run(req);
        }
        free(req);
        if (app_control_quit_requested())
            return shutdown_app();
        eng_boot_log_flush();
    }
}
