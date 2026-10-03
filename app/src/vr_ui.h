/*
 * vr_ui - PS5VR's own interface in the headset: the library (where videos are
 * picked) and the VR menu (how the playing video is shown). Both draw into a
 * ui_canvas that the engine shows as a floating panel (eng_agc_vr_panel).
 */
#pragma once

#include "app_input.h"
#include "ui_canvas.h"
#include "vr_sources.h"

#include <cstdint>
#include <string>
#include <vector>

constexpr int VR_UI_W = 1600, VR_UI_H = 1000;   /* panel canvas */

/* Where a panel sits: world-locked where the viewer looked when it opened. */
struct VrPanelPlace {
    bool shown = false;
    float yaw = 0, pitch = 0;
    int away_frames = 0;
    bool moved = false;               /* it just followed the gaze */
    /* Call every frame (60 Hz) with whether the panel is up. */
    void update(bool show);
    void reset_front() { yaw = 0; pitch = -6; }
    float below = 6.0f;               /* degrees under the gaze it opens at */
};

/* Pointing at a panel: by default with the eyes (look, then pinch with either
 * hand anywhere - the pinch itself moves nothing); with a hand ray (from about
 * the shoulder through the wrist and palm) when the eyes are not tracked. */
struct VrHandPointer {
    bool valid = false;               /* something points at the panel */
    bool gaze = false;                /* ... the eyes */
    bool laser = false;               /* ... a controller (else a hand ray) */
    float u = 0, v = 0;               /* where, 0..1 across and down (smoothed) */
    bool pinching = false;            /* a pinch (or trigger) is held */
    bool pressed = false;             /* it started this frame ... */
    bool released = false;            /* ... ended this frame */
    float press_u = 0, press_v = 0;   /* where it pointed just before the press */
    float drag = 0;                   /* how far the press moved since, in panel heights (down +) */
    bool tapped = false;              /* a short pinch with no panel under it */
    bool held_long = false;           /* a pinch was held 0.8 s (once per pinch) */
    uint32_t buttons_pressed = 0;     /* controller buttons that went down (pad bits) */
    /* Once per frame. panel = the panel's place, distance and width (metres);
     * aspect = height / width. With no panel shown only the pinches count.
     * Also draws the controllers, their lasers and the tracked hands. */
    void update(const VrPanelPlace *panel, float dist, float width, float aspect);

private:
    void smooth(float ru, float rv, bool eyes);
    bool m_pinch[2] = {false, false};
    int m_active = -1;                /* the hand that pinched last */
    float m_pinch_y0 = 0;             /* its pinch height when it began (metres) */
    int m_pinch_frames = 0;
    bool m_long_done = false;
    bool m_any_pinch = false;
    bool m_tap_on_panel = false;
    /* one-euro filter on (u, v), and the last frames for press_u/v */
    bool m_have = false;
    float m_fu = 0, m_fv = 0, m_du = 0, m_dv = 0;
    float m_hist_u[8] = {}, m_hist_v[8] = {};
    int m_hist_n = 0;
    /* controllers */
    uint32_t m_btn[2] = {0, 0};
    bool m_trigger[2] = {false, false};
    int m_laser_side = -1;
    float m_press_v_ctl = 0;
    /* blink to confirm */
    int m_closed_frames = 0;          /* eyes closed (gaze lost while worn) */
    bool m_blink_armed = false;       /* closed long enough: opening confirms */
    int m_buzz_frames = 0;
    float m_open_u = 0, m_open_v = 0; /* where the eyes rested before closing */
    int m_flash = 0;
};

class VrLibrary {
public:
    /* A hand pointer over the panel: focus follows it, pinch-and-release opens,
     * pinch-and-drag scrolls the list, the Back button goes back. */
    void point(const VrHandPointer &p, app_input_state &in);
    void open();                      /* back to the list of sources */
    /* 1: a video was chosen (url, title, vr format hint), 0: nothing yet. */
    int input(const app_input_state &in, std::string &url, std::string &title, std::string &vr);
    bool render(ui_canvas &c);        /* true when the pixels changed */
    void note(const std::string &msg) { m_note = msg; m_dirty = true; }
    bool busy() const;                /* a list (or a video) is loading */
    struct Job;                       /* (the worker thread's) */

private:
    void start(const VrSrc &where, int mode);   /* mode 0: list, 1: resolve a video */
    void poll();
    std::vector<VrSrc> m_items;
    std::vector<VrSrc> m_stack;       /* where we are, root first */
    std::string m_note;
    int m_focus = 0, m_scroll = 0;
    bool m_dirty = true;
    Job *m_job = nullptr;
    VrSrc m_chosen;                   /* a resolved video waiting to be played */
    bool m_have_chosen = false;
    /* hand pointer: what a pinch started on, and a drag of the list */
    enum { PRESS_NONE, PRESS_BACK, PRESS_LIST } m_press = PRESS_NONE;
    float m_press_y = 0;
    int m_press_scroll = 0, m_press_item = 0;
    bool m_drag = false, m_hover_back = false;
};

class VrMenu {
public:
    bool is_open() const { return m_open; }
    void open();
    void close() { m_open = false; m_dirty = true; }
    /* Applies changes directly (format, eyes, screen, sharpening, audio...). */
    void input(const app_input_state &in);
    /* A hand pointer: the row under it takes the focus, a pinch changes it,
     * Done closes the menu. */
    void point(const VrHandPointer &p, app_input_state &in);
    bool render(ui_canvas &c);

private:
    bool m_open = false, m_dirty = true;
    int m_row = 0;
    int m_press = -1;                 /* hand pointer: the row (or -2 Done) a pinch began on */
    bool m_hover_done = false;
};

/* Playback controls for hands: a bar under the picture, opened with a short
 * pinch. Rewind / play-pause / forward, the View menu, back to the library,
 * and a progress bar to pinch a place in. */
constexpr int VR_CTRL_W = 1600, VR_CTRL_H = 420;
class VrControls {
public:
    enum Action { NONE, PAUSE, BACK10, FWD10, SEEK, VIEW, LIBRARY };
    bool is_open() const { return m_open; }
    void open();
    void close() { m_open = false; }
    /* The pointer over the bar: an action when a pinch is let go on a control;
     * *seek_to = seconds for SEEK. */
    Action point(const VrHandPointer &p, double duration, double *seek_to);
    /* Closes after 6 s with nothing pointed at it. Every frame. */
    void tick();
    bool render(ui_canvas &c, const std::string &title, double position, double duration, bool paused);

private:
    bool m_open = false, m_dirty = true;
    int m_hover = -1, m_press = -1;   /* 0..4 the buttons, 5 the progress bar */
    float m_hover_u = 0;
    int m_idle = 0, m_last_sec = -1;
    bool m_last_paused = false;
};

/* Settings that outlive one video: kept for the session. */
struct VrSettings {
    int projection = -1;     /* -1 auto (from the file), 360, 180, 0 flat */
    int stereo = -1;         /* -1 auto, 0 mono, 1 sbs, 2 tb */
    bool spatial_audio = true;
    bool smooth_motion = true;        /* frame generation for 30 fps pictures */
    bool tv_mirror = false;
    int control = 0;                  /* hands & eyes: 0 off, 1 hands, 2 eyes + pinch, 3 eyes + blink */
    bool hands_needed() const { return control == 1 || control == 2; }
};
VrSettings &vr_settings();
/* The format of the playing video as detected, before the viewer's overrides. */
void vr_ui_set_detected(int projection, int stereo);
/* Apply the overrides on top of the detected format (VR menu, a new video). */
void vr_ui_apply_format();
