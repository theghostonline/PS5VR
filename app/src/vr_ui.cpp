/*
 * vr_ui - PS5VR's library and VR menu, drawn for the headset panel.
 */
#include "vr_ui.h"

#include "app_vr.h"
#include "ui_icons.h"
#include "ui_text.h"

#include "eng_agc_runtime.h"
#include "eng_boot_trace.h"
#include "eng_vr_audio.h"
#include "vr_system.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <strings.h>
#include <sys/stat.h>
#include <pthread.h>

namespace {

const ui_color kBlue = UI_RGBA(0x00, 0x70, 0xd1, 255);
const ui_color kPanel = UI_RGBA(12, 16, 28, 236);
const ui_color kRow = UI_RGBA(255, 255, 255, 14);
const ui_color kDim = UI_RGBA(150, 165, 190, 255);
const ui_color kText = UI_RGBA(235, 240, 248, 255);

/* The panel's frame: rounded dark card, the VR block, a title and a hint line. */
void draw_frame(ui_canvas &c, const char *title, const char *hints)
{
    ui_canvas_clear(&c);
    ui_fill_rrect(&c, 0, 0, VR_UI_W, VR_UI_H, 36, kPanel);
    ui_fill_rrect(&c, 48, 40, 96, 96, 12, kBlue);
    ui_text_draw_aligned(&c, UI_BOLD, 48, 96, 106, 1, UI_WHITE, "VR", 0);
    ui_text_draw(&c, UI_BOLD, 48, 172, 104, kText, title, VR_UI_W - 240);
    ui_fill_rect(&c, 48, 160, VR_UI_W - 96, 2, UI_RGBA(255, 255, 255, 30));
    if (hints && *hints)
        ui_text_draw_aligned(&c, UI_MEDIUM, 28, VR_UI_W / 2.0f, VR_UI_H - 36, 1, kDim, hints, 0);
}

} // namespace

/* -- panel placement ------------------------------------------------------- */
void VrPanelPlace::update(bool show)
{
    float y, p;
    app_vr_head_dir(&y, &p);
    const float py = std::max(-70.0f, std::min(50.0f, p)) - below;  /* a little below the gaze */
    if (show && !shown) {
        yaw = y;
        pitch = py;
        away_frames = 0;
    } else if (show) {
        /* Lazy follow: a panel left well outside the view for a second comes
         * back in front (also covers the stale first tracker poses). */
        float dy = std::fabs(std::remainder(y - yaw, 360.0f));
        float dp = std::fabs(py - pitch);
        if (dy > 45.0f || dp > 40.0f) {
            if (++away_frames >= 60) {
                yaw = y;
                pitch = py;
                away_frames = 0;
                moved = true;
            }
        } else {
            away_frames = 0;
        }
    }
    shown = show;
}

/* -- hand pointer --------------------------------------------------------------- */
namespace {
void qrot(const float q[4], const float v[3], float o[3])
{
    /* v' = v + 2w (q x v) + 2 q x (q x v) */
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    o[0] = v[0] + w * tx + (y * tz - z * ty);
    o[1] = v[1] + w * ty + (z * tx - x * tz);
    o[2] = v[2] + w * tz + (x * ty - y * tx);
}
float dot3(const float a[3], const float b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
} // namespace

namespace {
/* Where a ray from o (head-relative) along unit dir meets the panel: u, v. */
bool panel_hit(const VrPanelPlace *panel, float dist, float width, float aspect, const float o[3],
               const float dir_in[3], float *u, float *v)
{
    const float d2r = 3.14159265f / 180.0f;
    const float ya = panel->yaw * d2r, pa = panel->pitch * d2r;
    const float dir[3] = {sinf(ya) * cosf(pa), sinf(pa), -cosf(ya) * cosf(pa)};
    const float right[3] = {cosf(ya), 0.0f, sinf(ya)};
    const float up[3] = {-sinf(ya) * sinf(pa), cosf(pa), cosf(ya) * sinf(pa)};
    const float denom = dot3(dir_in, dir);
    if (denom < 0.2f)
        return false;
    const float t = (dist - dot3(o, dir)) / denom;
    if (t < 0.0f)
        return false;
    const float p[3] = {o[0] + t * dir_in[0], o[1] + t * dir_in[1], o[2] + t * dir_in[2]};
    *u = 0.5f + dot3(p, right) / width;
    *v = 0.5f - dot3(p, up) / (width * aspect);
    return *u > -0.08f && *u < 1.08f && *v > -0.08f && *v < 1.08f;
}
} // namespace

namespace {
/* (overlay colours are premultiplied, 0xAABBGGRR) */
uint32_t ov_rgba(int r, int g, int b, int a)
{
    return ((uint32_t)a << 24) | ((uint32_t)(b * a / 255) << 16) | ((uint32_t)(g * a / 255) << 8) |
           (uint32_t)(r * a / 255);
}

void cross3(const float a[3], const float b[3], float o[3])
{
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

void norm3(float v[3])
{
    const float l = sqrtf(dot3(v, v));
    if (l > 1e-6f)
        for (int k = 0; k < 3; k++)
            v[k] /= l;
}

/* A ribbon from a to b, width w, turned to face the head (at the origin). */
void ov_ribbon(const float a[3], const float b[3], float w, uint32_t ca, uint32_t cb)
{
    float d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, m[3], side[3];
    for (int k = 0; k < 3; k++)
        m[k] = 0.5f * (a[k] + b[k]);
    cross3(d, m, side);
    norm3(side);
    float p[4][3];
    for (int k = 0; k < 3; k++) {
        p[0][k] = a[k] - side[k] * w * 0.5f;
        p[1][k] = a[k] + side[k] * w * 0.5f;
        p[2][k] = b[k] + side[k] * w * 0.5f;
        p[3][k] = b[k] - side[k] * w * 0.5f;
    }
    const uint32_t c[4] = {ca, ca, cb, cb};
    eng_agc_vr_overlay_quad(p, c);
}

/* A small square at p facing the head. */
void ov_dot(const float p[3], float r, uint32_t col)
{
    const float up0[3] = {0, 1, 0};
    float right[3], up[3], f[3] = {p[0], p[1], p[2]};
    norm3(f);
    cross3(up0, f, right);
    norm3(right);
    cross3(f, right, up);
    float q[4][3];
    for (int k = 0; k < 3; k++) {
        q[0][k] = p[k] - right[k] * r - up[k] * r;
        q[1][k] = p[k] + right[k] * r - up[k] * r;
        q[2][k] = p[k] + right[k] * r + up[k] * r;
        q[3][k] = p[k] - right[k] * r + up[k] * r;
    }
    const uint32_t c[4] = {col, col, col, col};
    eng_agc_vr_overlay_quad(q, c);
}

/* A Sense controller: a rounded-off box along its forward axis, lit from above. */
void ov_controller(const float pos[3], const float q[4])
{
    const float ax[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float a[3][3];
    for (int i = 0; i < 3; i++)
        qrot(q, ax[i], a[i]);              /* the controller's right, up, back */
    const float half[3] = {0.022f, 0.018f, 0.065f};
    /* centred a little behind the tracked point (the ring sits at the front) */
    float c[3];
    for (int k = 0; k < 3; k++)
        c[k] = pos[k] + a[2][k] * 0.04f;
    for (int face = 0; face < 6; face++) {
        const int axis = face / 2;
        const float sgn = (face & 1) ? 1.0f : -1.0f;
        const int u = (axis + 1) % 3, w = (axis + 2) % 3;
        float p[4][3];
        const float su[4] = {-1, 1, 1, -1}, sw[4] = {-1, -1, 1, 1};
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 3; k++)
                p[i][k] = c[k] + a[axis][k] * half[axis] * sgn + a[u][k] * half[u] * su[i] +
                          a[w][k] * half[w] * sw[i];
        /* light: brighter when the face points up */
        const float n = a[axis][1] * sgn;
        const int g = (int)(70 + 70 * (n > 0 ? n : 0) + 20 * (n < 0 ? -n : 0));
        const uint32_t col = ov_rgba(g, g + 4, g + 12, 255);
        const uint32_t cc[4] = {col, col, col, col};
        eng_agc_vr_overlay_quad(p, cc);
    }
}

/* A tracked hand: its bones as soft ribbons, the finger tips as dots. */
void ov_hand(const vr_hand_t &hd, bool pinch)
{
    float v[VR_HAND_JOINTS][3];
    for (int j = 0; j < VR_HAND_JOINTS; j++)
        app_vr_to_view(hd.joint[j], v[j]);
    const uint32_t col = pinch ? ov_rgba(0x40, 0xa0, 0xff, 210) : ov_rgba(230, 236, 245, 150);
    /* OpenXR: wrist 1; thumb 2-5; index 6-10; middle 11-15; ring 16-20; little 21-25 */
    static const int kChains[5][2] = {{2, 5}, {6, 10}, {11, 15}, {16, 20}, {21, 25}};
    for (const auto &ch : kChains) {
        ov_ribbon(v[1], v[ch[0]], 0.012f, col, col);
        for (int j = ch[0]; j < ch[1]; j++)
            ov_ribbon(v[j], v[j + 1], j + 1 == ch[1] ? 0.009f : 0.012f, col, col);
        ov_dot(v[ch[1]], 0.006f, col);
    }
}
} // namespace

void VrHandPointer::smooth(float ru, float rv, bool eyes)
{
    /* one-euro filter: heavy smoothing while still, light while moving
     * (the eyes jump between fixations, so they get less lag) */
    const float rate = 60.0f, dcut = 1.0f, mincut = eyes ? 1.5f : 0.9f, beta = eyes ? 10.0f : 6.0f;
    auto alpha = [&](float cutoff) {
        const float tau = 1.0f / (2.0f * 3.14159265f * cutoff);
        return 1.0f / (1.0f + tau * rate);
    };
    if (!m_have) {
        m_fu = ru;
        m_fv = rv;
        m_du = m_dv = 0.0f;
        m_have = true;
    } else {
        const float ad = alpha(dcut);
        m_du += ad * ((ru - m_fu) * rate - m_du);
        m_dv += ad * ((rv - m_fv) * rate - m_dv);
        const float a = alpha(mincut + beta * sqrtf(m_du * m_du + m_dv * m_dv));
        m_fu += a * (ru - m_fu);
        m_fv += a * (rv - m_fv);
    }
    u = std::max(0.0f, std::min(1.0f, m_fu));
    v = std::max(0.0f, std::min(1.0f, m_fv));
}

void VrHandPointer::update(const VrPanelPlace *panel, float dist, float width, float aspect)
{
    valid = gaze = laser = pressed = released = tapped = held_long = false;
    buttons_pressed = 0;
    eng_agc_vr_overlay_clear();
    const int mode = vr_settings().control;

    /* -- the controllers: always on when tracked; a laser at the panel ------- */
    int ctl_hit = -1;
    float cu[2] = {0, 0}, cv[2] = {0, 0}, beam_end[2][3], tip[2][3];
    bool tracked[2] = {false, false}, trig_edge[2] = {false, false}, trig_up[2] = {false, false};
    for (int side = 0; side < 2; side++) {
        float pos[3], q[4], stick[2];
        uint32_t btn = 0;
        tracked[side] = app_vr_controller(side, pos, q, &btn, stick) != 0;
        buttons_pressed |= btn & ~m_btn[side];
        m_btn[side] = btn;
        /* select: the trigger (L2 / R2), or cross / square */
        const bool trig = (btn & (APP_BTN_L2 | APP_BTN_R2 | APP_BTN_CROSS | APP_BTN_SQUARE)) != 0;
        trig_edge[side] = trig && !m_trigger[side];
        trig_up[side] = !trig && m_trigger[side];
        m_trigger[side] = trig;
        if (!tracked[side])
            continue;
        const float back[3] = {0, 0, -1};
        float fwd[3];
        qrot(q, back, fwd);
        for (int k = 0; k < 3; k++) {
            tip[side][k] = pos[k] + fwd[k] * 0.02f;
            beam_end[side][k] = pos[k] + fwd[k] * 1.2f;
        }
        ov_controller(pos, q);
        if (panel && panel_hit(panel, dist, width, aspect, pos, fwd, &cu[side], &cv[side])) {
            /* the beam stops on the panel */
            const float d2r = 3.14159265f / 180.0f;
            const float ya = panel->yaw * d2r, pa = panel->pitch * d2r;
            const float pdir[3] = {sinf(ya) * cosf(pa), sinf(pa), -cosf(ya) * cosf(pa)};
            const float t = (dist - dot3(pos, pdir)) / dot3(fwd, pdir);
            for (int k = 0; k < 3; k++)
                beam_end[side][k] = pos[k] + fwd[k] * t;
            if (ctl_hit < 0 || side == m_laser_side)
                ctl_hit = side;
        }
        if (panel) {
            const bool on = ctl_hit == side;
            ov_ribbon(tip[side], beam_end[side], 0.004f,
                      on && trig ? ov_rgba(0x30, 0x90, 0xff, 230) : ov_rgba(255, 255, 255, on ? 200 : 90),
                      ov_rgba(255, 255, 255, on ? 120 : 0));
        }
    }
    if (ctl_hit >= 0) {
        m_laser_side = ctl_hit;
        laser = valid = true;
        smooth(cu[ctl_hit], cv[ctl_hit], false);
        pinching = m_trigger[ctl_hit];
        pressed = trig_edge[ctl_hit];
        released = trig_up[ctl_hit];
        if (pressed) {
            press_u = u;
            press_v = v;
        }
        drag = pinching ? v - press_v : 0;
        return;
    }
    if (m_laser_side >= 0 && (trig_up[0] || trig_up[1]))
        released = true;                   /* let go after moving off the panel */
    else if ((tracked[0] && trig_edge[0]) || (tracked[1] && trig_edge[1]))
        tapped = true;                     /* a trigger away from any panel: like a quick pinch */
    m_laser_side = -1;

    if (mode == 0) {
        m_have = false;
        pinching = false;
        drag = 0;
        return;
    }

    float head[3], yq[4], head_yaw = 0, head_pitch = 0;
    app_vr_head_pose(head, yq);
    app_vr_head_dir(&head_yaw, &head_pitch);
    const float hy = head_yaw * 3.14159265f / 180.0f;
    const float body_right[3] = {cosf(hy), 0.0f, sinf(hy)};

    /* -- hands: pinches (and their picture) -------------------------------- */
    bool edge_on[2] = {false, false}, edge_off[2] = {false, false}, hand_hit[2] = {false, false};
    float hu[2] = {0, 0}, hv[2] = {0, 0}, pinch_y[2] = {0, 0};
    vr_hand_t hd[2];
    bool have[2] = {false, false};
    bool any_pinch = false;
    for (int h = 0; h < 2; h++) {
        have[h] = vr_settings().hands_needed() && vr_hands_get(h, &hd[h]) != 0;
        if (!have[h]) {
            if (m_pinch[h])
                edge_off[h] = true;
            m_pinch[h] = false;
            continue;
        }
        float d[3];
        for (int k = 0; k < 3; k++)
            d[k] = hd[h].joint[VR_HAND_THUMB_TIP][k] - hd[h].joint[VR_HAND_INDEX_TIP][k];
        /* measured: a pinch reads 0.6-1.3 cm between the tip joints, a
         * relaxed hand between pinches 2.1-3 cm */
        const float gap = sqrtf(dot3(d, d));
        const bool was = m_pinch[h];
        m_pinch[h] = was ? gap < 0.020f : gap < 0.015f;
        pinch_y[h] = 0.5f * (hd[h].joint[VR_HAND_THUMB_TIP][1] + hd[h].joint[VR_HAND_INDEX_TIP][1]);
        if (m_pinch[h] && !was) {
            edge_on[h] = true;
            m_active = h;
            m_pinch_y0 = pinch_y[h];
        } else if (!m_pinch[h] && was) {
            edge_off[h] = true;
        }
        any_pinch = any_pinch || m_pinch[h];
        ov_hand(hd[h], m_pinch[h]);
    }

    /* -- what points: the eyes (modes 2, 3), else a hand ray (mode 1) ------- */
    float gu = 0, gv = 0;
    float gdir[3];
    const bool eye_mode = mode >= 2;
    const bool eyes_tracked = eye_mode && app_vr_gaze_dir(gdir);
    const bool eyes = eyes_tracked && panel &&
                      panel_hit(panel, dist, width, aspect, (const float[3]){0, 0, 0}, gdir, &gu, &gv);
    gaze = eyes;
    int h = -1;
    if (mode == 1 && panel) {
        for (int i = 0; i < 2; i++) {
            if (!have[i])
                continue;
            float o[3], palm[3];
            for (int k = 0; k < 3; k++)
                palm[k] = 0.5f * (hd[i].joint[VR_HAND_PALM][k] + hd[i].joint[VR_HAND_WRIST][k]);
            app_vr_to_view(palm, o);
            const float side = dot3(o, body_right) >= 0.0f ? 1.0f : -1.0f;
            const float sh[3] = {side * 0.17f * body_right[0], -0.12f, side * 0.17f * body_right[2]};
            float ray[3] = {o[0] - sh[0], o[1] - sh[1], o[2] - sh[2]};
            norm3(ray);
            hand_hit[i] = panel_hit(panel, dist, width, aspect, o, ray, &hu[i], &hv[i]);
        }
        const int pref = m_active >= 0 ? m_active : 0;
        h = hand_hit[pref] ? pref : hand_hit[0] ? 0 : hand_hit[1] ? 1 : -1;
    }
    const bool pointing = eyes || h >= 0;
    if (pointing) {
        smooth(eyes ? gu : hu[h], eyes ? gv : hv[h], eyes);
        valid = true;
        m_hist_u[m_hist_n % 8] = u;
        m_hist_v[m_hist_n % 8] = v;
        m_hist_n++;
    } else if (!(mode == 3 && m_closed_frames)) {
        m_have = false;
        m_hist_n = 0;
    }
    auto before = [&](int frames, float *pu, float *pv) {
        const int back = std::min(m_hist_n, frames);
        const int i = ((m_hist_n - 1 - back) % 8 + 8) % 8;
        *pu = m_hist_n > back ? m_hist_u[i] : u;
        *pv = m_hist_n > back ? m_hist_v[i] : v;
    };

    if (mode == 3) {
        /* -- eyes + blink: hold the eyes closed ~0.6 s; a short buzz says
         * "open now", and opening selects where they rested before. A natural
         * blink (0.1-0.4 s) never reaches the buzz. ------------------------- */
        const bool worn = app_vr_worn() != 0;
        if (m_buzz_frames && --m_buzz_frames == 0)
            app_vr_vibrate(0);
        if (!eyes_tracked && worn && (m_closed_frames || m_have)) {
            if (m_closed_frames == 0)
                before(6, &m_open_u, &m_open_v);   /* (the lid starts closing before the gaze is lost) */
            m_closed_frames++;
            if (m_closed_frames == 36 && !m_blink_armed) {
                m_blink_armed = true;
                app_vr_vibrate(14);
                m_buzz_frames = 5;
            }
            if (m_closed_frames > 150) {             /* 2.5 s: not a blink - cancel */
                m_blink_armed = false;
                m_have = false;
            }
            valid = m_have;
            u = m_open_u;
            v = m_open_v;
        } else if (eyes_tracked) {
            if (m_blink_armed) {
                eng_bt("vr: blink select (%d frames closed)", m_closed_frames);
                valid = true;
                pressed = released = true;
                press_u = u = m_open_u;
                press_v = v = m_open_v;
                m_flash = 8;
            }
            m_blink_armed = false;
            m_closed_frames = 0;
        }
        pinching = m_flash > 0 && m_flash--;
        drag = 0;
        return;
    }

    /* -- pinch to select (modes 1, 2) ---------------------------------------- */
    const bool on = eyes ? (edge_on[0] || edge_on[1]) : (h >= 0 && edge_on[h]);
    const bool off = eyes ? ((edge_off[0] || edge_off[1]) && !any_pinch)
                          : (m_active >= 0 && edge_off[m_active]);
    if (on && pointing)
        before(5, &press_u, &press_v);       /* where it pointed before the pinch moved it */
    pressed = on && pointing;
    released = off;
    pinching = any_pinch;
    if (any_pinch && m_active >= 0 && have[m_active])
        drag = eyes ? (m_pinch_y0 - pinch_y[m_active]) / 0.25f : v - press_v;
    else
        drag = 0;
    /* a short pinch with no panel under it (the player shows its controls),
     * and a long one anywhere (its menu) */
    if (any_pinch) {
        if (!m_any_pinch)
            m_tap_on_panel = pointing;
        if (++m_pinch_frames >= 48 && !m_long_done) {
            held_long = true;
            m_long_done = true;
        }
    } else {
        if (m_any_pinch && m_pinch_frames < 24 && !m_tap_on_panel)
            tapped = true;
        m_pinch_frames = 0;
        m_long_done = false;
    }
    m_any_pinch = any_pinch;
}

/* -- settings ---------------------------------------------------------------- */
static VrSettings s_settings;
static int s_det_projection = 360, s_det_stereo = 0;

VrSettings &vr_settings()
{
    return s_settings;
}

void vr_ui_set_detected(int projection, int stereo)
{
    s_det_projection = projection;
    s_det_stereo = stereo;
}

void vr_ui_apply_format()
{
    app_vr_format f;
    f.projection = s_settings.projection >= 0 ? s_settings.projection : s_det_projection;
    f.stereo = s_settings.stereo >= 0 ? s_settings.stereo : s_det_stereo;
    app_vr_set_format(&f);
}

/* -- library ----------------------------------------------------------------- */
struct VrLibrary::Job {
    pthread_t t;
    VrSrc where;
    int mode;                         /* 0 list, 1 resolve */
    volatile bool done = false;
    bool ok = false;
    std::vector<VrSrc> items;
    VrSrc play;
    std::string err;
};

static void *library_worker(void *arg)
{
    VrLibrary::Job *j = (VrLibrary::Job *)arg;
    if (j->mode == 1)
        j->ok = vr_src_resolve(j->where, j->play, j->err);
    else
        j->ok = vr_src_list(j->where, j->items, j->err);
    j->done = true;
    return nullptr;
}

bool VrLibrary::busy() const
{
    return m_job != nullptr;
}

void VrLibrary::start(const VrSrc &where, int mode)
{
    if (m_job)
        return;
    m_job = new Job;
    m_job->where = where;
    m_job->mode = mode;
    if (pthread_create(&m_job->t, nullptr, library_worker, m_job) != 0) {
        delete m_job;
        m_job = nullptr;
        m_note = "Could not start loading";
    }
    m_dirty = true;
}

void VrLibrary::poll()
{
    if (!m_job || !m_job->done)
        return;
    pthread_join(m_job->t, nullptr);
    Job *j = m_job;
    m_job = nullptr;
    m_dirty = true;
    eng_bt("library: %s %s -> %s, %d items%s%s", j->mode ? "resolve" : "list", j->where.name.c_str(),
           j->ok ? "ok" : "failed", (int)j->items.size(), j->err.empty() ? "" : ": ", j->err.c_str());
    if (j->mode == 1) {
        if (j->ok) {
            m_chosen = j->play;
            m_have_chosen = true;
        } else {
            m_note = j->err;
        }
    } else {
        if (j->ok || !j->items.empty()) {
            m_items = j->items;
            m_focus = m_scroll = 0;
        }
        if (!j->err.empty())
            m_note = j->err;
        if (!j->ok && !m_stack.empty() && m_stack.back().kind != SRC_ROOT)
            m_stack.pop_back();           /* could not open it: stay where we were */
    }
    delete j;
}

void VrLibrary::open()
{
    if (m_job)
        return;
    m_stack.clear();
    VrSrc root;
    root.kind = SRC_ROOT;
    root.name = "Library";
    m_stack.push_back(root);
    m_items.clear();
    start(root, 0);
}

int VrLibrary::input(const app_input_state &in, std::string &url, std::string &title,
                     std::string &vr)
{
    poll();
    if (m_have_chosen) {
        m_have_chosen = false;
        eng_bt("library: play %s (vr \"%s\")", m_chosen.url.c_str(), m_chosen.vr.c_str());
        url = m_chosen.url;
        title = m_chosen.name;
        vr = m_chosen.vr;
        return 1;
    }
    const uint32_t p = in.pressed;
    if (!p || m_job)
        return 0;
    m_note.clear();
    m_dirty = true;
    const int n = (int)m_items.size();
    if (p & APP_BTN_DOWN && n) m_focus = std::min(n - 1, m_focus + 1);
    if (p & APP_BTN_UP && n) m_focus = std::max(0, m_focus - 1);
    if (p & APP_BTN_R1 && n) m_focus = std::min(n - 1, m_focus + 8);
    if (p & APP_BTN_L1 && n) m_focus = std::max(0, m_focus - 8);
    if (p & APP_BTN_CIRCLE && m_stack.size() > 1) {
        m_stack.pop_back();
        m_items.clear();
        start(m_stack.back(), 0);
        return 0;
    }
    if (p & APP_BTN_TRIANGLE) {             /* refresh */
        m_items.clear();
        start(m_stack.back(), 0);
        return 0;
    }
    if (p & APP_BTN_CROSS && m_focus < n) {
        const VrSrc e = m_items[m_focus];
        if (e.kind == SRC_PLAY) {
            eng_bt("library: play %s (vr \"%s\")", e.url.c_str(), e.vr.c_str());
            url = e.url;
            title = e.name;
            vr = e.vr;
            return 1;
        }
        if (e.kind == SRC_MIRROR) {
            vr_settings().tv_mirror = !vr_settings().tv_mirror;
            vr_prefs_set_mirror(vr_settings().tv_mirror);
            app_vr_set_mirror(vr_settings().tv_mirror);
            m_items[m_focus].detail = vr_settings().tv_mirror ? "On" : "Off";
        } else if (e.kind == SRC_HANDS) {
            VrSettings &st = vr_settings();
            st.control = (st.control + 1) % 4;
            vr_prefs_set_control(st.control);
            vr_hands_request(st.hands_needed());
            m_items[m_focus].detail = vr_control_name(st.control);
            eng_bt("ps5vr: hands & eyes: %s", vr_control_name(st.control));
        } else if (e.kind == SRC_SITE_VIDEO) {
            start(e, 1);
        } else if (e.kind != SRC_INFO) {
            m_stack.push_back(e);
            m_items.clear();
            start(e, 0);
        }
    }
    return 0;
}

namespace {
/* The library's list: rows big enough to pick with the eyes. */
const int kLibRows = 9, kLibRowH = 80, kLibTop = 186;
/* The header's button (Back in the library, Done in the menu). */
const float kHdrBtnX = 1340, kHdrBtnY = 56, kHdrBtnW = 212, kHdrBtnH = 68;
bool in_header_button(float x, float y)
{
    return x >= kHdrBtnX - 16 && x <= kHdrBtnX + kHdrBtnW + 16 && y >= kHdrBtnY - 16 &&
           y <= kHdrBtnY + kHdrBtnH + 16;
}
void draw_header_button(ui_canvas &c, bool hover, bool chevron, const char *label)
{
    ui_fill_rrect(&c, kHdrBtnX, kHdrBtnY, kHdrBtnW, kHdrBtnH, kHdrBtnH / 2,
                  hover ? kBlue : UI_RGBA(255, 255, 255, 24));
    const float tx = chevron ? kHdrBtnX + 78 : kHdrBtnX + kHdrBtnW / 2;
    if (chevron)
        ui_draw_icon(&c, UI_ICON_CHEVRON_LEFT, kHdrBtnX + 48, kHdrBtnY + kHdrBtnH / 2, 34, UI_WHITE);
    ui_text_draw_aligned(&c, UI_SEMIBOLD, 32, tx, kHdrBtnY + 45, chevron ? 0 : 1, UI_WHITE, label, 0);
}
} // namespace

void VrLibrary::point(const VrHandPointer &p, app_input_state &in)
{
    const int rows = kLibRows, row_h = kLibRowH, top = kLibTop;
    const int n = (int)m_items.size();
    const bool back_shown = m_stack.size() > 1;
    const float x = p.u * VR_UI_W, y = p.v * VR_UI_H;
    const bool hover_back = p.valid && !m_job && back_shown && in_header_button(x, y);
    if (hover_back != m_hover_back) {
        m_hover_back = hover_back;
        m_dirty = true;
    }
    if (!p.valid || m_job) {
        if (p.released)
            m_press = PRESS_NONE;
        return;
    }
    if (p.pressed) {
        const float px = p.press_u * VR_UI_W, py = p.press_v * VR_UI_H;
        m_drag = false;
        if (back_shown && in_header_button(px, py)) {
            m_press = PRESS_BACK;
        } else if (py >= top && py < top + rows * row_h) {
            m_press = PRESS_LIST;
            m_press_y = py;
            m_press_scroll = m_scroll;
            m_press_item = m_scroll + (int)((py - top) / row_h);
        } else {
            m_press = PRESS_NONE;
        }
    }
    if (p.pinching && m_press == PRESS_LIST) {          /* pinch and drag: scroll */
        const float dy = p.drag * VR_UI_H;
        if (std::fabs(dy) > 28.0f)
            m_drag = true;
        if (m_drag && n > rows) {
            const int sc = std::max(0, std::min(n - rows, m_press_scroll - (int)lroundf(dy / row_h)));
            if (sc != m_scroll) {
                m_scroll = sc;
                m_focus = std::max(sc, std::min(sc + rows - 1, m_focus));
                m_dirty = true;
            }
        }
    } else if (!p.pinching && y >= top && y < top + rows * row_h) {   /* focus follows */
        const int i = m_scroll + (int)((y - top) / row_h);
        if (i < n && i != m_focus) {
            m_focus = i;
            m_dirty = true;
        }
    }
    if (p.released) {
        if (m_press == PRESS_BACK && hover_back) {
            in.pressed |= APP_BTN_CIRCLE;
        } else if (m_press == PRESS_LIST && !m_drag && m_press_item < n) {
            m_focus = m_press_item;
            in.pressed |= APP_BTN_CROSS;
        }
        m_press = PRESS_NONE;
    }
}

bool VrLibrary::render(ui_canvas &c)
{
    poll();
    static int s_spin;
    if (m_job) {                              /* a moving dot while loading */
        s_spin++;
        m_dirty = m_dirty || (s_spin % 6) == 0;
    }
    if (!m_dirty)
        return false;
    m_dirty = false;
    const VrSrc &here = m_stack.empty() ? VrSrc() : m_stack.back();
    const bool root = m_stack.size() <= 1;
    draw_frame(c, root ? "Library" : here.name.c_str(),
               root ? "\xC3\x97 Open    \xE2\x96\xB3 Refresh    R3 Recentre"
                    : "\xC3\x97 Open    \xE2\x97\x8B Back    L1 / R1 Page    \xE2\x96\xB3 Refresh");
    const int rows = kLibRows, row_h = kLibRowH, top = kLibTop;
    if (m_focus < m_scroll) m_scroll = m_focus;
    if (m_focus >= m_scroll + rows) m_scroll = m_focus - rows + 1;
    if (m_job) {
        std::string dots = std::string(1 + (s_spin / 6) % 3, '.');
        ui_text_draw_aligned(&c, UI_MEDIUM, 36, VR_UI_W / 2.0f, 480, 1, kDim,
                             (m_job->mode ? "Opening" + dots : "Loading" + dots).c_str(), 0);
    } else if (m_items.empty()) {
        ui_text_draw_aligned(&c, UI_MEDIUM, 34, VR_UI_W / 2.0f, 480, 1, kDim,
                             m_note.empty() ? "Nothing here" : m_note.c_str(), 0);
    }
    for (int r = 0; !m_job && r < rows && m_scroll + r < (int)m_items.size(); r++) {
        const int i = m_scroll + r;
        const VrSrc &e = m_items[i];
        const float y = (float)(top + r * row_h);
        const bool f = i == m_focus;
        ui_fill_rrect(&c, 48, y, VR_UI_W - 96, row_h - 10, 16,
                      f ? kBlue : (e.kind == SRC_INFO ? UI_RGBA(255, 255, 255, 6) : kRow));
        ui_text_draw(&c, f ? UI_SEMIBOLD : UI_MEDIUM, 34, 84, y + 47, UI_WHITE, e.name.c_str(),
                     VR_UI_W - 640);
        if (!e.detail.empty())
            ui_text_draw_aligned(&c, UI_REGULAR, 28, VR_UI_W - 84.0f, y + 47, 2,
                                 f ? UI_WHITE : kDim, e.detail.c_str(), 600);
    }
    if (!root)
        draw_header_button(c, m_hover_back, true, "Back");
    if (!m_job && (int)m_items.size() > rows) {        /* where in a long list */
        const float track = (float)(rows * row_h - 8);
        const float h = std::max(40.0f, track * rows / (float)m_items.size());
        const float y0 = top + (track - h) * m_scroll / (float)(m_items.size() - rows);
        ui_fill_rrect(&c, VR_UI_W - 40, top, 6, track, 3, UI_RGBA(255, 255, 255, 20));
        ui_fill_rrect(&c, VR_UI_W - 40, y0, 6, h, 3, UI_RGBA(255, 255, 255, 110));
    }
    if (!m_note.empty() && !m_items.empty())
        ui_text_draw_aligned(&c, UI_MEDIUM, 28, VR_UI_W / 2.0f, VR_UI_H - 84, 1,
                             UI_RGBA(255, 190, 90, 255), m_note.c_str(), 0);
    return true;
}

/* -- VR menu --------------------------------------------------------------- */
namespace {
enum MenuRow { ROW_TYPE, ROW_LAYOUT, ROW_SWAP, ROW_SCREEN, ROW_SHARPEN, ROW_SMOOTH, ROW_HDR,
               ROW_AUDIO, ROW_MIRROR, ROW_HANDS, ROW_RECENTRE, ROW_COUNT };
const char *kTypes[] = {"Auto", "360\xC2\xB0", "180\xC2\xB0", "Flat screen"};
const int kTypeVal[] = {-1, 360, 180, 0};
const char *kLayouts[] = {"Auto", "2D (mono)", "3D side by side", "3D top-bottom"};
const int kLayoutVal[] = {-1, 0, 1, 2};
const char *kScreens[] = {"Small", "Medium", "Large", "Huge"};
const float kScreenDeg[] = {50, 70, 90, 115};
const char *kSharpen[] = {"Auto", "Off", "Sharpen", "Upscale + sharpen"};
const int kSharpenVal[] = {-1, 0, 1, 2};

template <size_t N> int index_of(const int (&vals)[N], int v)
{
    for (size_t i = 0; i < N; i++)
        if (vals[i] == v)
            return (int)i;
    return 0;
}
} // namespace

void VrMenu::open()
{
    m_open = true;
    m_dirty = true;
    m_row = 0;
}

void VrMenu::input(const app_input_state &in)
{
    const uint32_t p = in.pressed;
    if (!p)
        return;
    m_dirty = true;
    if (p & (APP_BTN_CIRCLE | APP_BTN_OPTIONS)) {
        m_open = false;
        return;
    }
    if (p & APP_BTN_DOWN) m_row = (m_row + 1) % ROW_COUNT;
    if (p & APP_BTN_UP) m_row = (m_row + ROW_COUNT - 1) % ROW_COUNT;
    const int d = (p & APP_BTN_RIGHT) ? 1 : (p & APP_BTN_LEFT) ? -1 : (p & APP_BTN_CROSS) ? 1 : 0;
    if (!d)
        return;
    VrSettings &s = vr_settings();
    app_vr_options o;
    app_vr_get_options(&o);
    switch (m_row) {
    case ROW_TYPE:
        s.projection = kTypeVal[(index_of(kTypeVal, s.projection) + 4 + d) % 4];
        vr_ui_apply_format();
        break;
    case ROW_LAYOUT:
        s.stereo = kLayoutVal[(index_of(kLayoutVal, s.stereo) + 4 + d) % 4];
        vr_ui_apply_format();
        break;
    case ROW_SWAP:
        o.swap = !o.swap;
        break;
    case ROW_SCREEN: {
        int i = 1;
        for (int k = 0; k < 4; k++)
            if (std::fabs(kScreenDeg[k] - o.screen_deg) < 1) i = k;
        o.screen_deg = kScreenDeg[(i + 4 + d) % 4];
        break;
    }
    case ROW_SHARPEN:
        o.sharpen = kSharpenVal[(index_of(kSharpenVal, o.sharpen) + 4 + d) % 4];
        break;
    case ROW_SMOOTH:
        s.smooth_motion = !s.smooth_motion;
        eng_agc_vr_set_framegen(s.smooth_motion);
        break;
    case ROW_HDR: {
        const bool on = !vr_prefs_hdr();
        vr_prefs_set_hdr(on);
        o.bits = on ? 16 : 10;
        app_vr_set_options(&o);
        app_vr_restart();                     /* new eye buffers */
        app_vr_get_options(&o);
        break;
    }
    case ROW_AUDIO:
        s.spatial_audio = !s.spatial_audio;
        eng_vr_audio_set_enabled(s.spatial_audio);
        break;
    case ROW_MIRROR:
        s.tv_mirror = !s.tv_mirror;
        vr_prefs_set_mirror(s.tv_mirror);
        app_vr_set_mirror(s.tv_mirror);
        break;
    case ROW_HANDS:
        s.control = (s.control + 4 + d) % 4;
        vr_prefs_set_control(s.control);
        vr_hands_request(s.hands_needed());
        break;
    case ROW_RECENTRE:
        if (p & APP_BTN_CROSS)
            app_vr_recenter();
        break;
    }
    app_vr_set_options(&o);
}

void VrMenu::point(const VrHandPointer &p, app_input_state &in)
{
    const int row_h = 66, top = 186;
    const float x = p.u * VR_UI_W, y = p.v * VR_UI_H;
    const bool hover_done = p.valid && in_header_button(x, y);
    if (hover_done != m_hover_done) {
        m_hover_done = hover_done;
        m_dirty = true;
    }
    if (!p.valid) {
        if (p.released)
            m_press = -1;
        return;
    }
    auto row_at = [&](float yy) { return yy >= top ? (int)((yy - top) / row_h) : -1; };
    if (p.pressed) {
        const float px = p.press_u * VR_UI_W, py = p.press_v * VR_UI_H;
        const int r = row_at(py);
        m_press = in_header_button(px, py) ? -2 : (r >= 0 && r < ROW_COUNT ? r : -1);
    }
    const int r = row_at(y);
    if (!p.pinching && r >= 0 && r < ROW_COUNT && r != m_row) {
        m_row = r;
        m_dirty = true;
    }
    if (p.released) {
        if (m_press == -2 && hover_done) {
            in.pressed |= APP_BTN_CIRCLE;
        } else if (m_press >= 0) {
            m_row = m_press;
            in.pressed |= APP_BTN_CROSS;
        }
        m_press = -1;
    }
}

bool VrMenu::render(ui_canvas &c)
{
    if (!m_dirty)
        return false;
    m_dirty = false;
    draw_frame(c, "View",
               "\xE2\x86\x91\xE2\x86\x93 Choose    \xE2\x86\x90\xE2\x86\x92 Change    \xE2\x97\x8B Close");
    draw_header_button(c, m_hover_done, false, "Done");
    const VrSettings &s = vr_settings();
    app_vr_options o;
    app_vr_get_options(&o);
    int scr = 1;
    for (int k = 0; k < 4; k++)
        if (std::fabs(kScreenDeg[k] - o.screen_deg) < 1) scr = k;
    const char *labels[ROW_COUNT] = {"Video type", "3D layout", "Swap eyes", "Screen size",
                                     "Sharpening", "Smooth motion", "HDR (experimental)", "Spatial audio", "Mirror to TV", "Hands & eyes (experimental)", "Recentre view"};
    std::string values[ROW_COUNT] = {
        kTypes[index_of(kTypeVal, s.projection)], kLayouts[index_of(kLayoutVal, s.stereo)],
        o.swap ? "On" : "Off", kScreens[scr], kSharpen[index_of(kSharpenVal, o.sharpen)],
        s.smooth_motion ? "On" : "Off", vr_prefs_hdr() ? "On" : "Off", s.spatial_audio ? "On" : "Off", s.tv_mirror ? "On" : "Off", vr_control_name(s.control),
        "Press \xC3\x97"};
    const int row_h = 66, top = 186;
    for (int r = 0; r < ROW_COUNT; r++) {
        const float y = (float)(top + r * row_h);
        const bool f = r == m_row;
        ui_fill_rrect(&c, 48, y, VR_UI_W - 96, row_h - 8, 14, f ? kBlue : kRow);
        ui_text_draw(&c, f ? UI_SEMIBOLD : UI_MEDIUM, 30, 84, y + 40, UI_WHITE, labels[r], 900);
        std::string v = r == ROW_RECENTRE ? values[r] : "\xE2\x80\xB9  " + values[r] + "  \xE2\x80\xBA";
        ui_text_draw_aligned(&c, UI_MEDIUM, 28, VR_UI_W - 84.0f, y + 40, 2,
                             f ? UI_WHITE : kDim, v.c_str(), 700);
    }
    return true;
}

/* -- playback controls ----------------------------------------------------------- */
namespace {
const float kBarX = 60, kBarW = 1480, kBarY = 118;
const float kBtnX = 70, kBtnY = 180, kBtnW = 260, kBtnH = 200, kBtnGap = 40;

int control_at(float x, float y)
{
    if (y >= kBarY - 30 && y <= kBarY + 34 && x >= kBarX - 20 && x <= kBarX + kBarW + 20)
        return 5;
    if (y >= kBtnY - 10 && y <= kBtnY + kBtnH + 10) {
        const int i = (int)((x - kBtnX) / (kBtnW + kBtnGap));
        if (i >= 0 && i < 5 && x - kBtnX - i * (kBtnW + kBtnGap) <= kBtnW)
            return i;
    }
    return -1;
}

std::string clock_text(double s)
{
    const int t = s > 0 ? (int)s : 0;
    char b[32];
    if (t >= 3600)
        std::snprintf(b, sizeof b, "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60);
    else
        std::snprintf(b, sizeof b, "%d:%02d", t / 60, t % 60);
    return b;
}
} // namespace

void VrControls::open()
{
    m_open = true;
    m_dirty = true;
    m_idle = 0;
    m_hover = m_press = -1;
}

VrControls::Action VrControls::point(const VrHandPointer &p, double duration, double *seek_to)
{
    if (!m_open)
        return NONE;
    const float x = p.u * VR_CTRL_W, y = p.v * VR_CTRL_H;
    const int hov = p.valid ? control_at(x, y) : -1;
    if (p.valid)
        m_idle = 0;
    if (hov != m_hover || (hov == 5 && std::fabs(p.u - m_hover_u) > 0.002f)) {
        m_hover = hov;
        m_hover_u = p.u;
        m_dirty = true;
    }
    if (p.pressed)
        m_press = control_at(p.press_u * VR_CTRL_W, p.press_v * VR_CTRL_H);
    if (!p.released)
        return NONE;
    const int t = m_press;
    m_press = -1;
    if (t < 0 || t != hov)
        return NONE;
    switch (t) {
    case 0: return BACK10;
    case 1: return PAUSE;
    case 2: return FWD10;
    case 3: return VIEW;
    case 4: return LIBRARY;
    default:
        *seek_to = std::max(0.0f, std::min(1.0f, (x - kBarX) / kBarW)) * duration;
        return duration > 0 ? SEEK : NONE;
    }
}

void VrControls::tick()
{
    if (m_open && ++m_idle > 360)
        close();
}

bool VrControls::render(ui_canvas &c, const std::string &title, double position, double duration,
                        bool paused)
{
    const int sec = (int)position;
    if (!m_dirty && sec == m_last_sec && paused == m_last_paused)
        return false;
    m_dirty = false;
    m_last_sec = sec;
    m_last_paused = paused;
    ui_canvas_clear(&c);
    ui_fill_rrect(&c, 0, 0, VR_CTRL_W, VR_CTRL_H, 36, kPanel);
    ui_text_draw(&c, UI_BOLD, 36, kBarX, 74, kText, title.c_str(), 1100);
    const std::string t = clock_text(position) + "  /  " + clock_text(duration);
    ui_text_draw_aligned(&c, UI_MEDIUM, 32, kBarX + kBarW, 74, 2, kDim, t.c_str(), 0);
    /* progress */
    const float f = duration > 0 ? (float)std::max(0.0, std::min(1.0, position / duration)) : 0.0f;
    ui_fill_rrect(&c, kBarX, kBarY, kBarW, 12, 6, UI_RGBA(255, 255, 255, 40));
    ui_fill_rrect(&c, kBarX, kBarY, std::max(12.0f, kBarW * f), 12, 6, kBlue);
    ui_fill_rrect(&c, kBarX + kBarW * f - 12, kBarY - 6, 24, 24, 12, UI_WHITE);
    if (m_hover == 5) {
        const float hx = kBarX + kBarW * std::max(0.0f, std::min(1.0f, m_hover_u));
        ui_fill_rrect(&c, hx - 3, kBarY - 14, 6, 40, 3, UI_RGBA(255, 255, 255, 200));
        const std::string at = clock_text(duration * std::max(0.0f, std::min(1.0f, m_hover_u)));
        ui_text_draw_aligned(&c, UI_SEMIBOLD, 28, std::max(kBarX + 40, std::min(kBarX + kBarW - 40, hx)),
                             kBarY - 22, 1, UI_WHITE, at.c_str(), 0);
    }
    /* buttons */
    struct Btn { ui_icon icon; const char *label; } btn[5] = {
        {UI_ICON_FAST_REWIND, "\xE2\x88\x92" "10 s"},
        {paused ? UI_ICON_PLAY : UI_ICON_PAUSE, paused ? "Play" : "Pause"},
        {UI_ICON_FAST_FORWARD, "+10 s"},
        {UI_ICON_ASPECT, "View"},
        {UI_ICON_CHEVRON_LEFT, "Library"},
    };
    for (int i = 0; i < 5; i++) {
        const float x = kBtnX + i * (kBtnW + kBtnGap);
        ui_fill_rrect(&c, x, kBtnY, kBtnW, kBtnH, 28, m_hover == i ? kBlue : kRow);
        ui_draw_icon(&c, btn[i].icon, x + kBtnW / 2, kBtnY + 78, 72, UI_WHITE);
        ui_text_draw_aligned(&c, UI_MEDIUM, 30, x + kBtnW / 2, kBtnY + 162, 1, UI_WHITE, btn[i].label, 0);
    }
    return true;
}
