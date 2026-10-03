/*
 * vr_ui_preview - draw PS5VR's library and VR menu on the Mac, to PNG.
 *   tools/preview/build.sh && tools/preview/vr_ui_preview <out-dir>
 */
#include "vr_ui.h"
#include "app_vr.h"
#include "vr_system.h"
#include "ui_text.h"

#include <cstdio>
#include <cstring>
#include <png.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

/* the headset side, stubbed */
static app_vr_options s_opt = {-1, 10, 0, 0, 70.0f};
extern "C" {
void app_vr_head_dir(float *y, float *p) { *y = 0; *p = 0; }
void app_vr_get_options(app_vr_options *o) { *o = s_opt; }
void app_vr_set_options(const app_vr_options *o) { s_opt = *o; }
void app_vr_set_format(const app_vr_format *) {}
void app_vr_set_mirror(int) {}
void app_vr_recenter(void) {}
void eng_vr_audio_set_enabled(int) {}
void app_vr_head_pose(float pos[3], float q[4]) { pos[0] = pos[1] = pos[2] = 0; q[0] = q[1] = q[2] = 0; q[3] = 1; }
void vr_hands_request(int) {}
int app_vr_controller(int, float *, float *, uint32_t *, float *) { return 0; }
void app_vr_to_view(const float p[3], float o[3]) { o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; }
void app_vr_vibrate(int) {}
int app_vr_worn(void) { return 1; }
void eng_agc_vr_overlay_clear(void) {}
void eng_agc_vr_overlay_quad(const float (*)[3], const uint32_t *) {}
void eng_agc_vr_set_framegen(int) {}
int app_vr_restart(void) { return 0; }
int app_vr_gaze_dir(float d[3]) { (void)d; return 0; }
int vr_hands_get(int, vr_hand_t *) { return 0; }
}

/* the sources, stubbed with what the console's root list holds */
bool vr_src_list(const VrSrc &where, std::vector<VrSrc> &out, std::string &)
{
    auto add = [&](VrSrcKind k, const char *n, const char *d) {
        VrSrc s;
        s.kind = k;
        s.name = n;
        s.detail = d;
        out.push_back(s);
    };
    out.clear();
    if (where.kind == SRC_ROOT) {
        add(SRC_DIR, "This PS5", "/data/ps5vr/videos");
        add(SRC_DIR, "USB drive 1", "/mnt/usb0");
        add(SRC_DLNA_LIST, "Media servers (DLNA)", "");
        add(SRC_RSS_LIST, "RSS feeds", "2");
        add(SRC_SITE, "My Stash", "DeoVR");
        add(SRC_INFO, "Settings", "http://192.168.1.20:8090");
    } else {
        add(SRC_PLAY, "Sample.180.SBS.4k.mp4", "33.5 MB");
        add(SRC_PLAY, "sample_180_sbs_8k60.mp4", "866.3 MB");
        add(SRC_PLAY, "dd_360_tb_4k60.mp4", "48.6 MB");
    }
    return true;
}
bool vr_src_resolve(const VrSrc &, VrSrc &, std::string &) { return false; }
int vr_prefs_control(void) { return 0; }
const char *vr_control_name(int) { return "Off"; }
bool vr_prefs_hdr(void) { return false; }
bool vr_prefs_mirror(void) { return false; }
void vr_prefs_set_mirror(bool) {}
void vr_prefs_set_hdr(bool) {}
void vr_prefs_set_control(int) {}

static void save(const ui_canvas &c, const std::string &path)
{
    ui_canvas bg;
    ui_canvas_init(&bg, c.w + 160, c.h + 160);
    ui_fill_rect(&bg, 0, 0, bg.w, bg.h, UI_RGBA(20, 28, 50, 255));   /* the space behind */
    ui_image im = {c.px, c.w, c.h};
    ui_draw_image(&bg, &im, 80, 80, c.w, c.h, 1.0f);
    png_image pi;
    std::memset(&pi, 0, sizeof pi);
    pi.version = PNG_IMAGE_VERSION;
    pi.width = bg.w;
    pi.height = bg.h;
    pi.format = PNG_FORMAT_RGBA;
    png_image_write_to_file(&pi, path.c_str(), 0, bg.px, bg.w * 4, nullptr);
    std::printf("wrote %s\n", path.c_str());
    ui_canvas_free(&bg);
}

static app_input_state press(uint32_t b)
{
    app_input_state s;
    std::memset(&s, 0, sizeof s);
    s.pressed = b;
    return s;
}

int main(int argc, char **argv)
{
    const std::string out = argc > 1 ? argv[1] : ".";
    ui_text_init();
    ui_canvas c;
    ui_canvas_init(&c, VR_UI_W, VR_UI_H);

    /* a library with a few videos in "This PS5" (paths under /tmp on the Mac) */
    VrLibrary lib;
    lib.open();
    usleep(200 * 1000);                       /* the worker thread lists */
    lib.render(c);
    save(c, out + "/library_sources.png");
    std::string u, t, v;
    lib.input(press(APP_BTN_CROSS), u, t, v);
    usleep(200 * 1000);
    lib.render(c);
    save(c, out + "/library_folder.png");

    {
        ui_canvas cc;
        ui_canvas_init(&cc, VR_CTRL_W, VR_CTRL_H);
        VrControls ctl;
        ctl.open();
        ctl.render(cc, "Sample 180 3D (4K)", 754, 5400, false);
        save(cc, out + "/vr_controls.png");
    }
    VrMenu menu;
    menu.open();
    menu.input(press(APP_BTN_DOWN));
    menu.input(press(APP_BTN_RIGHT));
    menu.render(c);
    save(c, out + "/vr_menu.png");
    return 0;
}
