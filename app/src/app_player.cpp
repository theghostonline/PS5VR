/*
 * The PS5VR Player's frame loop.
 *
 * One playback: parse the page's request, show PS5VR's loading screen while
 * the stream opens on a worker thread, then every frame
 *
 *   input -> interface commands -> status -> timers
 *   GPU clear, video quad, subtitles layer, interface layer, present
 *
 * presenting only when something on screen changed. The page gets progress
 * every five seconds and the final result (position, what to do next, the
 * tracks chosen) before it reopens.
 */
#include "app_player.h"

#include "app_bridge.h"
#include "app_control.h"
#include "app_input.h"
#include "app_osd.h"
#include "app_session.h"
#include "app_subs.h"
#include "app_vr.h"
#include "vr_ui.h"
#include "ui_canvas.h"
#include "ui_image.h"
#include "ui_text.h"

#include "eng/Application.hpp"
#include "eng/services/PlaybackController.hpp"

#include "eng_agc_runtime.h"
#include "eng_boot_log.h"
#include "eng_boot_trace.h"
#include "eng_playback.h"
#include "eng_thread.h"
#include "eng_vdec.h"
#include "pp_playback.h"

#include "cJSON.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <new>
#include <pthread.h>
#include <string>
#include <strings.h>
#include <unistd.h>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/dovi_meta.h>

extern pp_playback g_pp_pb;
extern char app_stream_headers[4096];
extern char app_stream_user_agent[512];
/* Which the engine screen is up. The engine's decode, audio and subtitle threads
 * park unless it is the player (2): the engine's screen manager set it on entering
 * its player screen, and nothing else does. */
extern int screen;
extern AVFormatContext *play_fmt;
extern int video_stream_index;
extern int audio_stream_index;
extern volatile int pb_prebuffer_hold;
extern int pb_prebuffer_packets;
int eng_demux_seek_busy(void);
void eng_demux_rebuffer(int64_t target_us, int max_ms);
float eng_demux_prebuffer_progress(void);
double eng_demux_buffered_s(void);
extern volatile int eng_demux_state;
extern volatile unsigned long packet_queue_ring_fallbacks;
extern int dbg_video_packets;
extern int g_vdec_force_ffmpeg;
extern char app_vdec_conf[512];
extern volatile unsigned app_control_beats;
extern volatile int app_control_stage;
extern int dbg_video_frames;
void eng_log_alloc_state(const char *when);
}

namespace engine {
extern int DisplayWidth;
extern int DisplayHeight;
}

namespace {

constexpr int kScreenNone = 0;
constexpr int kScreenPlayer = 2;
constexpr int CW = 1920, CH = 1080;     /* overlay canvases */

engine::PlaybackController *s_pb = nullptr;
int s_user_id = -1;
ui_canvas s_osd_canvas, s_sub_canvas;
AppOsd s_osd;

/* The viewer's language preferences for the audio pick (set per playback). */
std::vector<std::string> s_audio_langs;

double now_s()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ---- names for the quality line and track lists ----------------------------------- */

std::string video_codec_name(enum AVCodecID id)
{
    switch (id) {
    case AV_CODEC_ID_HEVC: return "HEVC";
    case AV_CODEC_ID_H264: return "H.264";
    case AV_CODEC_ID_AV1: return "AV1";
    case AV_CODEC_ID_VP9: return "VP9";
    case AV_CODEC_ID_VP8: return "VP8";
    case AV_CODEC_ID_MPEG2VIDEO: return "MPEG-2";
    case AV_CODEC_ID_MPEG4: return "MPEG-4";
    case AV_CODEC_ID_VC1: return "VC-1";
    default: return avcodec_get_name(id);
    }
}

std::string audio_codec_name(const AVCodecParameters *par)
{
    switch (par->codec_id) {
    case AV_CODEC_ID_TRUEHD: return "TrueHD";
    case AV_CODEC_ID_MLP: return "MLP";
    case AV_CODEC_ID_DTS:
        if (par->profile == AV_PROFILE_DTS_HD_MA) return "DTS-HD MA";
        if (par->profile == AV_PROFILE_DTS_HD_HRA) return "DTS-HD HRA";
        if (par->profile == AV_PROFILE_DTS_EXPRESS) return "DTS Express";
        return "DTS";
    case AV_CODEC_ID_EAC3:
        return par->profile == AV_PROFILE_EAC3_DDP_ATMOS ? "Dolby Digital+ Atmos" : "Dolby Digital+";
    case AV_CODEC_ID_AC3: return "Dolby Digital";
    case AV_CODEC_ID_AAC: return "AAC";
    case AV_CODEC_ID_FLAC: return "FLAC";
    case AV_CODEC_ID_OPUS: return "Opus";
    case AV_CODEC_ID_VORBIS: return "Vorbis";
    case AV_CODEC_ID_MP3: return "MP3";
    case AV_CODEC_ID_ALAC: return "ALAC";
    default:
        if (par->codec_id >= AV_CODEC_ID_PCM_S16LE && par->codec_id < AV_CODEC_ID_ADPCM_IMA_QT)
            return "PCM";
        return avcodec_get_name(par->codec_id);
    }
}

std::string channel_name(int ch)
{
    switch (ch) {
    case 1: return "Mono";
    case 2: return "Stereo";
    case 3: return "2.1";
    case 6: return "5.1";
    case 7: return "6.1";
    case 8: return "7.1";
    default: return ch > 0 ? std::to_string(ch) + " ch" : std::string();
    }
}

/* How good an audio stream is, for the default pick. */
int audio_rank(const AVStream *st)
{
    const AVCodecParameters *p = st->codecpar;
    int r = 0;
    switch (p->codec_id) {
    case AV_CODEC_ID_TRUEHD: case AV_CODEC_ID_MLP: r = 90; break;
    case AV_CODEC_ID_FLAC: case AV_CODEC_ID_ALAC: r = 85; break;
    case AV_CODEC_ID_DTS: r = p->profile == AV_PROFILE_DTS_HD_MA ? 88 : 60; break;
    case AV_CODEC_ID_EAC3: r = 55; break;
    case AV_CODEC_ID_AC3: r = 45; break;
    case AV_CODEC_ID_OPUS: r = 40; break;
    case AV_CODEC_ID_AAC: r = 35; break;
    default:
        r = (p->codec_id >= AV_CODEC_ID_PCM_S16LE && p->codec_id < AV_CODEC_ID_ADPCM_IMA_QT) ? 86 : 20;
    }
    return r * 10 + std::min(8, p->ch_layout.nb_channels);
}

bool is_commentary(const AVStream *st)
{
    if (st->disposition & (AV_DISPOSITION_COMMENT | AV_DISPOSITION_VISUAL_IMPAIRED))
        return true;
    const AVDictionaryEntry *t = av_dict_get(st->metadata, "title", nullptr, 0);
    return t && t->value && (strcasestr(t->value, "commentary") || strcasestr(t->value, "description"));
}

std::string stream_lang(const AVStream *st)
{
    const AVDictionaryEntry *l = av_dict_get(st->metadata, "language", nullptr, 0);
    return l && l->value ? l->value : "";
}

} // namespace

/*
 * PlaybackController asks this when a file opens (APP_APP hook): the first
 * stream in the viewer's preferred languages, and within the language the
 * best one (lossless before lossy, more channels first), never commentary.
 */
extern "C" int app_pick_audio_stream(AVFormatContext *fmt, int current)
{
    if (!fmt)
        return -1;
    auto decodable = [&](unsigned i) {
        const AVStream *st = fmt->streams[i];
        return st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && avcodec_find_decoder(st->codecpar->codec_id);
    };
    auto best_in = [&](const std::string &key) {
        int best = -1, best_rank = -1;
        for (unsigned i = 0; i < fmt->nb_streams; i++) {
            if (!decodable(i) || is_commentary(fmt->streams[i]))
                continue;
            if (!key.empty() && app_language_key(stream_lang(fmt->streams[i])) != key)
                continue;
            int rank = audio_rank(fmt->streams[i]);
            if (fmt->streams[i]->disposition & AV_DISPOSITION_DEFAULT)
                rank += 1;
            if (rank > best_rank) {
                best_rank = rank;
                best = (int)i;
            }
        }
        return best;
    };
    for (const std::string &lang : s_audio_langs) {
        const int pick = best_in(app_language_key(lang));
        if (pick >= 0)
            return pick;
    }
    /* No preference matched: the best track in the file's own default language. */
    if (current >= 0 && current < (int)fmt->nb_streams) {
        const int pick = best_in(app_language_key(stream_lang(fmt->streams[current])));
        if (pick >= 0)
            return pick;
    }
    return current;
}

namespace {

/* ---- background work -------------------------------------------------------------- */

struct OpenJob {
    pthread_t thread{};
    std::atomic<bool> running{false};
    std::atomic<bool> done{false};
    bool ok = false;
    int kind = 0;                 /* 0 open, 1 audio switch */
    engine::PlaybackSource src;
    double at = 0;
    int audio_stream = -1;
};

void *open_thread(void *arg)
{
    OpenJob *j = static_cast<OpenJob *>(arg);
    if (j->kind == 1)
        j->ok = s_pb->switchAudioTrack(j->audio_stream);
    else
        j->ok = s_pb->startPlaybackSource(j->src, j->at);
    j->done = true;
    return nullptr;
}

bool start_job(OpenJob &j)
{
    j.done = false;
    j.ok = false;
    j.running = true;
    if (eng_thread_create(&j.thread, open_thread, &j) != 0) {
        j.running = false;
        return false;
    }
    return true;
}

void finish_job(OpenJob &j)
{
    if (j.running) {
        pthread_join(j.thread, nullptr);
        j.running = false;
    }
}

/* Addon subtitles: asks each Stremio subtitle URL the page computed and adds
 * what comes back (preferred languages first). Runs beside the playback. */
struct SubFetch {
    std::vector<AppRequest::SubtitleRequest> requests;
    std::vector<std::string> langs;
    int session = 0;
};
std::atomic<int> s_session{0};

std::string fetch_text(const std::string &url, int timeout_s)
{
    AVIOContext *io = nullptr;
    AVDictionary *o = nullptr;
    std::string out;
    av_dict_set(&o, "rw_timeout", std::to_string(timeout_s * 1000000).c_str(), 0);
    av_dict_set(&o, "user_agent", "Mozilla/5.0 (PlayStation 5) PS5VR", 0);
    if (avio_open2(&io, url.c_str(), AVIO_FLAG_READ, nullptr, &o) >= 0) {
        unsigned char buf[16384];
        int n;
        while ((n = avio_read(io, buf, sizeof buf)) > 0 && out.size() < 4 * 1024 * 1024)
            out.append((const char *)buf, (size_t)n);
        avio_closep(&io);
    }
    av_dict_free(&o);
    return out;
}

void *subtitle_fetch_thread(void *arg)
{
    SubFetch *f = static_cast<SubFetch *>(arg);
    struct Found { std::string url, lang, label, headers; int pref; };
    std::vector<Found> found;
    for (const auto &q : f->requests) {
        if (s_session != f->session)
            break;
        const std::string body = fetch_text(q.url, 12);
        cJSON *root = cJSON_Parse(body.c_str());
        const cJSON *arr = cJSON_GetObjectItem(root, "subtitles");
        const cJSON *it;
        int n = 0;
        cJSON_ArrayForEach(it, arr) {
            const char *url = cJSON_GetStringValue(cJSON_GetObjectItem(it, "url"));
            const char *lang = cJSON_GetStringValue(cJSON_GetObjectItem(it, "lang"));
            if (!url || !*url)
                continue;
            Found x;
            x.url = url;
            x.lang = lang ? lang : "";
            x.label = q.addon;
            const cJSON *h = cJSON_GetObjectItem(it, "headers");
            if (!h) {
                const cJSON *bh = cJSON_GetObjectItem(it, "behaviorHints");
                h = cJSON_GetObjectItem(cJSON_GetObjectItem(bh, "proxyHeaders"), "request");
            }
            if (h)
                x.headers = app_headers_from_json(h, nullptr);
            x.pref = 1000;
            for (size_t k = 0; k < f->langs.size(); k++)
                if (app_language_key(x.lang) == app_language_key(f->langs[k]))
                    x.pref = (int)k;
            found.push_back(x);
            n++;
        }
        cJSON_Delete(root);
        eng_bt("subs: addon '%s' gave %d", q.addon.c_str(), n);
    }
    std::stable_sort(found.begin(), found.end(), [](const Found &a, const Found &b) { return a.pref < b.pref; });
    int added = 0;
    for (const Found &x : found) {
        if (s_session != f->session || added >= 40)
            break;
        app_subs_add_external(x.url.c_str(), x.lang.c_str(), x.label.c_str(), x.headers.c_str());
        added++;
    }
    delete f;
    return nullptr;
}

/* ---- per-playback state ------------------------------------------------------------ */

struct Session {
    AppRequest req;
    AppResult res;
    OpenJob job;
    bool opened = false;          /* the stream is open (job finished ok) */
    bool started = false;         /* a frame is on screen */
    bool failed = false;
    std::string error;
    bool cancel = false;
    bool done = false;
    bool user_picked_subs = false;
    double open_started = 0;
    double last_frame_at = 0;
    double last_pos = 0, last_pos_change = 0;
    double wd_since = 0;              /* hardware decoder watchdog */
    int wd_frames = -1;
    int wd_consumed = 0;
        double vq_empty_since = 0;        /* underrun watch */
    int rebuffers = 0;
    double last_rebuffer_at = -1e9;
    int64_t last_pts = INT64_MIN;
    app_vr_format vr_fmt = {0, 0};
    bool vr_wanted = false;       /* PS5VR: this video goes to the headset */
    bool vr_on = false;
    bool vr_failed = false;
    AppStatus st;
    std::vector<int> audio_streams;   /* status.audio index -> stream */
};

void update_hdr(bool playing)
{
    static int s_want = 0;
    static bool s_refused = false;
    const int trc = eng_agc_runtime_last_video_trc();
    const int want = (playing && (trc == 16 || trc == 18)) ? 1 : 0;
    if (!playing)
        s_refused = false;
    if (want == s_want || (want && s_refused))
        return;
    eng_bt("ps5vr: hdr10 %s (trc=%d)", want ? "on" : "off", trc);
    if (eng_agc_runtime_set_hdr_output(want) == 0)
        s_want = want;
    else if (want)
        s_refused = true;   /* the display said no: stay tone-mapped */
}

/* Dolby Vision profile of a stream, 0 when it carries none. */
int dolby_vision_profile(const AVCodecParameters *p)
{
    for (int i = 0; i < p->nb_coded_side_data; i++)
        if (p->coded_side_data[i].type == AV_PKT_DATA_DOVI_CONF &&
            p->coded_side_data[i].size >= sizeof(AVDOVIDecoderConfigurationRecord))
            return ((const AVDOVIDecoderConfigurationRecord *)p->coded_side_data[i].data)->dv_profile;
    return 0;
}

std::string quality_line()
{
    if (!play_fmt)
        return "";
    const char *dot = "  \xC2\xB7  ";
    std::string q;
    if (video_stream_index >= 0 && video_stream_index < (int)play_fmt->nb_streams) {
        const AVCodecParameters *p = play_fmt->streams[video_stream_index]->codecpar;
        const int h = p->height;
        if (h >= 2000) q = "4K";
        else if (h >= 1400) q = "1440p";
        else if (h >= 1000) q = "1080p";
        else if (h >= 700) q = "720p";
        else if (h > 0) q = std::to_string(h) + "p";
        /* The PS5 cannot output Dolby Vision: profile 8 shows its HDR10/HLG
         * base layer, and profile 5 is rebuilt into HDR10 (src/dv_rpu.c) -
         * which is what the label says. */
        const int dv = dolby_vision_profile(p);
        const char *hdr = p->color_trc == AVCOL_TRC_SMPTE2084 ? (dv ? "HDR10 (Dolby Vision)" : "HDR10")
                          : p->color_trc == AVCOL_TRC_ARIB_STD_B67 ? (dv ? "HLG (Dolby Vision)" : "HLG")
                          : dv == 5 ? "HDR10 (Dolby Vision)" : nullptr;
        if (hdr)
            q += std::string(q.empty() ? "" : dot) + hdr;
        q += std::string(q.empty() ? "" : dot) + video_codec_name(p->codec_id);
    }
    if (audio_stream_index >= 0 && audio_stream_index < (int)play_fmt->nb_streams) {
        const AVCodecParameters *p = play_fmt->streams[audio_stream_index]->codecpar;
        q += std::string(q.empty() ? "" : dot) + audio_codec_name(p);
        const std::string ch = channel_name(p->ch_layout.nb_channels);
        if (!ch.empty())
            q += " " + ch;
    }
    return q;
}

void refresh_audio(Session &s)
{
    s.st.audio.clear();
    s.audio_streams.clear();
    s.st.audio_active = -1;
    for (const auto &t : s_pb->getAudioTracks()) {
        AppAudioTrack a;
        a.stream = t.streamIndex;
        a.lang = (t.language == "UND" || t.language == "und") ? "" : t.language;
        a.title = t.title;
        if (play_fmt && t.streamIndex >= 0 && t.streamIndex < (int)play_fmt->nb_streams) {
            const AVStream *st = play_fmt->streams[t.streamIndex];
            a.codec = audio_codec_name(st->codecpar);
            a.is_default = (st->disposition & AV_DISPOSITION_DEFAULT) != 0;
        } else {
            a.codec = t.codecName;
        }
        a.channels = channel_name(t.channels);
        if (t.streamIndex == s_pb->getActiveAudioStream())
            s.st.audio_active = (int)s.st.audio.size();
        s.st.audio.push_back(a);
        s.audio_streams.push_back(t.streamIndex);
    }
}

/* The viewer's subtitle preference: their language, else a forced track in
 * the audio's language when subtitles are off. Re-run as addon tracks arrive
 * until the viewer picks one themselves. */
void auto_select_subtitles(Session &s)
{
    if (s.user_picked_subs || app_subs_selected() >= 0)
        return;
    const AppPrefs &p = s.req.prefs;
    const int n = app_subs_count();
    auto find = [&](const std::string &lang, bool forced_only) {
        const std::string key = app_language_key(lang);
        int best = -1, best_score = -1;
        for (int i = 0; i < n; i++) {
            app_sub_track t;
            if (app_subs_track(i, &t) != 0 || t.state != 1)
                continue;
            if (app_language_key(t.lang) != key || forced_only != (t.forced != 0))
                continue;
            const int score = (t.external ? 0 : 20) + (t.hearing_impaired ? 0 : 5) +
                              (t.is_default ? 2 : 0) + (t.bitmap ? 0 : 1);
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        return best;
    };
    int pick = -1;
    if (p.subtitles_enabled) {
        for (const std::string &l : p.subtitle_langs)
            if ((pick = find(l, false)) >= 0)
                break;
    } else if (p.forced_only_when_off && s.st.audio_active >= 0) {
        pick = find(s.st.audio[s.st.audio_active].lang, true);
    }
    if (pick >= 0) {
        app_subs_select(pick);
        s_osd.note_subtitle_choice();
    }
}

app_rect video_rect(const pp_video_frame &f)
{
    app_rect r = {0, 0, (float)CW, (float)CH};
    float dw = (float)(f.disp_w ? f.disp_w : 16), dh = (float)(f.disp_h ? f.disp_h : 9);
    if (play_fmt && video_stream_index >= 0 && video_stream_index < (int)play_fmt->nb_streams) {
        const AVRational sar = play_fmt->streams[video_stream_index]->codecpar->sample_aspect_ratio;
        if (sar.num > 0 && sar.den > 0)
            dw *= (float)sar.num / (float)sar.den;
    }
    const int mode = (int)s_pb->getViewMode();
    if (mode == 2)
        return r;
    const float sx = CW / dw, sy = CH / dh;
    const float k = mode == 1 ? std::max(sx, sy) : std::min(sx, sy);
    r.w = dw * k;
    r.h = dh * k;
    r.x = (CW - r.w) * 0.5f;
    r.y = (CH - r.h) * 0.5f;
    return r;
}

/* Open source `index` of the request at `at` seconds, on the worker thread. */
void open_source(Session &s, int index, double at)
{
    const AppSource src = s.req.sources[index];
    app_subs_close();
    for (const AppSubtitleRef &r : s.req.subtitles)
        app_subs_add_external(r.url.c_str(), r.lang.c_str(), r.label.c_str(), r.headers.c_str());
    s.req.url = src.url;
    s.req.headers = src.headers;
    s.req.user_agent = src.user_agent;
    s.req.source_index = index;
    s.req.stream_title = src.title;
    s.req.stream_description = src.description;
    s.req.stream_addon = src.addon;
    s.req.start_position = at;
    std::snprintf(app_stream_headers, sizeof app_stream_headers, "%s", src.headers.c_str());
    std::snprintf(app_stream_user_agent, sizeof app_stream_user_agent, "%s", src.user_agent.c_str());
    s.opened = s.started = s.failed = false;
    s.error.clear();
    s.last_pts = INT64_MIN;
    s.user_picked_subs = false;
    s.job.kind = 0;
    s.job.src = engine::PlaybackSource();
    s.job.src.url = src.url;
    s.job.src.title = s.req.header_title();
    s.job.src.provider = "ps5vr";
    s.job.at = at;
    s.open_started = now_s();
    s_osd.begin(&s.req, s.open_started);
    start_job(s.job);
    eng_bt("ps5vr: opening source %d at %.1f s", index, at);
}

void apply(Session &s, const OsdCommand &c)
{
    switch (c.cmd) {
    case OsdCmd::TogglePause:
        s_pb->togglePause();
        break;
    case OsdCmd::SeekTo:
        s_pb->seekTo(std::max(0.0, c.value));
        break;
    case OsdCmd::Stop:
        s.done = true;
        if (s.res.state.empty())
            s.res.state = s.failed ? "error" : "stopped";
        break;
    case OsdCmd::PlayNext:
        s.done = true;
        s.res.state = "ended";
        s.res.action = "next";
        s.res.season = s.req.next.season;
        s.res.episode = s.req.next.episode;
        break;
    case OsdCmd::PlayEpisode:
        s.done = true;
        s.res.state = "stopped";
        s.res.action = "episode";
        s.res.season = c.season;
        s.res.episode = c.episode;
        break;
    case OsdCmd::SelectAudio:
        if (c.index >= 0 && c.index < (int)s.audio_streams.size() && !s.job.running) {
            s.job.kind = 1;
            s.job.audio_stream = s.audio_streams[c.index];
            s.st.switching = true;
            start_job(s.job);
        }
        break;
    case OsdCmd::SelectSubtitle:
        s.user_picked_subs = true;
        app_subs_select(c.index);
        break;
    case OsdCmd::SubtitleDelay:
        app_subs_set_delay_ms((int)c.value);
        break;
    case OsdCmd::SubtitleStyle:
        break;
    case OsdCmd::SetViewMode:
        s_pb->setViewMode((engine::ViewMode)c.index);
        break;
    case OsdCmd::SwitchSource:
        if (c.index >= 0 && c.index < (int)s.req.sources.size() && !s.job.running) {
            const double at = s.started ? s_pb->getPositionSeconds() : s.req.start_position;
            s_pb->stopPlayback();
            open_source(s, c.index, at);
        }
        break;
    }
}

} // namespace

/* ---- public -------------------------------------------------------------------------- */

extern "C" void app_player_init(int user_id)
{
    s_user_id = user_id;
    s_pb = new engine::PlaybackController();
    engine::Application::getInstance().setPlaybackController(s_pb);
    avformat_network_init();
    pp_playback_init(&g_pp_pb);
    pp_playback_set_output(&g_pp_pb, engine::DisplayWidth, engine::DisplayHeight, PP_ASPECT_FIT);
    eng_vdec_prefer_nv12(1);
    if (ui_canvas_init(&s_osd_canvas, CW, CH) != 0 || ui_canvas_init(&s_sub_canvas, CW, CH) != 0)
        eng_bt("ps5vr: overlay canvases could not be allocated");
    if (ui_text_init() != 0)
        eng_bt("ps5vr: fonts failed");
    if (app_subs_init() != 0)
        eng_bt("ps5vr: libass failed");
    eng_bt("ps5vr: player ready (%dx%d)", engine::DisplayWidth, engine::DisplayHeight);
}

/*
 * The hardware decoder is taking the stream but returning no pictures (or
 * gave up): reopen the same source at the same point on the software decoder.
 * One-way for this playback - g_vdec_force_ffmpeg stays set until the next
 * request - so a seek cannot land back on the decoder that failed.
 */
static void reopen_software(Session &s, const char *why)
{
    const double at = s.started ? s_pb->getPositionSeconds() : s.req.start_position;
    eng_bt("ps5vr: hardware decoder %s - reopening at %.1f s on the software decoder", why, at);
    g_vdec_force_ffmpeg = 1;
    s_pb->stopPlayback();
    app_subs_close();
    for (const AppSubtitleRef &r : s.req.subtitles)
        app_subs_add_external(r.url.c_str(), r.lang.c_str(), r.label.c_str(), r.headers.c_str());
    s.req.start_position = at;
    s.opened = s.started = s.failed = false;
    s.error.clear();
    s.last_pts = INT64_MIN;
    s.job.kind = 0;
    s.job.src = engine::PlaybackSource();
    s.job.src.url = s.req.url;
    s.job.src.title = s.req.header_title();
    s.job.src.provider = "ps5vr";
    s.job.at = at;
    s.open_started = now_s();
    s.wd_frames = -1;
    s_osd.begin(&s.req, s.open_started);
    start_job(s.job);
}

#ifdef APP_VR
/* Mean |chroma - neutral| over a grid of one half of the picture (8-bit
 * scale): ~0 for a greyscale half. half: 0 left/top, 1 right/bottom. */
static double chroma_dev(const pp_video_frame &f, bool sbs, int half)
{
    const int cw = (int)f.disp_w / 2, ch = (int)f.disp_h / 2;
    if (cw < 16 || ch < 16)
        return 0;
    double sum = 0;
    int n = 0;
    for (int gy = 0; gy < 24; gy++) {
        for (int gx = 0; gx < 24; gx++) {
            int x = (gx * 2 + 1) * cw / 48, y = (gy * 2 + 1) * ch / 48;
            if (sbs) x = x / 2 + half * cw / 2;   /* grid inside the chosen half */
            else     y = y / 2 + half * ch / 2;
            int cb, cr;
            if (f.uv) {
                if (f.ten_bit) {
                    const uint16_t *p = (const uint16_t *)(f.uv + (size_t)y * f.uv_pitch) + 2 * x;
                    cb = p[0] >> 8; cr = p[1] >> 8;
                } else {
                    const uint8_t *p = f.uv + (size_t)y * f.uv_pitch + 2 * x;
                    cb = p[0]; cr = p[1];
                }
            } else if (f.u && f.v) {
                if (f.ten_bit) {
                    cb = ((const uint16_t *)(f.u + (size_t)y * f.u_pitch))[x] >> 2;
                    cr = ((const uint16_t *)(f.v + (size_t)y * f.v_pitch))[x] >> 2;
                } else {
                    cb = f.u[(size_t)y * f.u_pitch + x];
                    cr = f.v[(size_t)y * f.v_pitch + x];
                }
            } else {
                return 0;
            }
            sum += std::abs(cb - 128) + std::abs(cr - 128);
            n++;
        }
    }
    return n ? sum / n : 0;
}

/* A "stereo" video whose second half has no colour is colour + depth map
 * (Kandao, RGBD captures): show the colour half to both eyes. */
static void vr_check_depth_half(app_vr_format &fmt, const pp_video_frame &f)
{
    if (fmt.stereo != 1 && fmt.stereo != 2)
        return;
    const bool sbs = fmt.stereo == 1;
    const double a = chroma_dev(f, sbs, 0), b = chroma_dev(f, sbs, 1);
    eng_bt("ps5vr: vr stereo colour check %s: first half %.2f, second half %.2f",
           sbs ? "sbs" : "tb", a, b);
    if (a > 2.0 && b < 0.3)
        fmt.stereo = sbs ? 4 : 3;      /* ENG_AGC_VR_SBS_LEFT / _TB_TOP */
}
#endif

/* PS5VR: once the first frame is up, the picture is projected into the headset
 * on a steady 60 Hz cadence (app_vr_begin_frame), so every video frame is held
 * for the same number of headset refreshes; Hmd2 turns each projection with the
 * head on the refresh in between - exact for video at infinity. new_frame only
 * tells whether the source picture changed. Returns false when the TV draws. */
#ifdef APP_VR
/* PS5VR's panel while a video plays: the VR menu when open, else the player
 * controls and subtitles (PS5VR's OSD layers, composited into one canvas). */
static VrMenu s_vrmenu;
static VrPanelPlace s_vrplace;
static ui_canvas s_vrmenu_canvas, s_vrmix_canvas, s_vrctrl_canvas;
/* the hand control bar, its own place (lower than the other panels), and
 * what it shows - copied from the session each frame */
static VrControls s_vrctrl;
static VrPanelPlace s_ctrlplace;
static struct { std::string title; double position = 0, duration = 0; bool paused = false; } s_vrctrl_st;
constexpr float CTRL_DIST = 1.7f, CTRL_WIDTH = 1.5f;

/* dst = osd over sub (premultiplied) */
static void vr_mix(const ui_canvas &sub, const ui_canvas &osd, ui_canvas &dst)
{
    const size_t n = (size_t)dst.w * dst.h;
    for (size_t i = 0; i < n; i++) {
        const uint32_t o = osd.px[i], b = sub.px[i];
        const uint32_t a = o >> 24;
        if (a == 255 || !b) { dst.px[i] = o; continue; }
        if (!a) { dst.px[i] = b; continue; }
        const uint32_t k = 255 - a;
        uint32_t r = 0;
        for (int sh = 0; sh < 32; sh += 8) {
            const uint32_t c = ((o >> sh) & 255) + (((b >> sh) & 255) * k + 127) / 255;
            r |= (c > 255 ? 255 : c) << sh;
        }
        dst.px[i] = r;
    }
}

/* The panel for this frame; true when it changed (the headset must redraw). */
static bool vr_panel_update(bool subs_on, bool sub_changed, bool osd_changed)
{
    if (!s_vrmenu_canvas.px) {
        ui_canvas_init(&s_vrmenu_canvas, VR_UI_W, VR_UI_H);
        ui_canvas_init(&s_vrmix_canvas, CW, CH);
        ui_canvas_init(&s_vrctrl_canvas, VR_CTRL_W, VR_CTRL_H);
        s_ctrlplace.below = 22.0f;
    }
    const bool ctrl_was = s_ctrlplace.shown;
    s_ctrlplace.update(s_vrctrl.is_open() && !s_vrmenu.is_open());
    if (s_ctrlplace.shown) {
        const bool ch = s_vrctrl.render(s_vrctrl_canvas, s_vrctrl_st.title, s_vrctrl_st.position,
                                        s_vrctrl_st.duration, s_vrctrl_st.paused) || !ctrl_was;
        eng_agc_vr_panel(s_vrctrl_canvas.px, VR_CTRL_W, VR_CTRL_H, ch, 1, s_ctrlplace.yaw,
                         s_ctrlplace.pitch, CTRL_DIST, CTRL_WIDTH, 1.0f);
        s_vrplace.shown = false;                /* the next panel opens fresh (and uploads) */
        return ch;
    }
    if (s_vrmenu.is_open()) {
        const bool ch = s_vrmenu.render(s_vrmenu_canvas);
        const bool was = s_vrplace.shown;
        s_vrplace.update(true);
        eng_agc_vr_panel(s_vrmenu_canvas.px, VR_UI_W, VR_UI_H, ch || !was, 1, s_vrplace.yaw,
                         s_vrplace.pitch, 2.0f, 2.0f, 1.0f);
        return ch || !was;
    }
    const bool osd = s_osd.visible();
    const bool show = osd || subs_on;
    const bool was = s_vrplace.shown;
    s_vrplace.update(show);
    if (!show) {
        eng_agc_vr_panel(nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        return was;
    }
    const ui_canvas *src = &s_osd_canvas;
    const bool changed = sub_changed || osd_changed || !was;
    if (osd && subs_on) {
        if (changed)
            vr_mix(s_sub_canvas, s_osd_canvas, s_vrmix_canvas);
        src = &s_vrmix_canvas;
    } else if (subs_on) {
        src = &s_sub_canvas;
    }
    eng_agc_vr_panel(src->px, CW, CH, changed, 1, s_vrplace.yaw, s_vrplace.pitch, 2.3f, 2.6f, 1.0f);
    return changed;
}

/* Controller presses PS5VR handles itself while a video plays in the headset:
 * Options opens the VR menu (it then has the controller), R3 recentres. */
static VrHandPointer s_vrhand;

static void vr_input(Session &s, app_input_state &in, std::vector<OsdCommand> &cmds)
{
    if (!s.vr_on)
        return;
    /* hands: a short pinch shows the control bar, a long one the View menu;
     * both are pointed at and pinched */
    s_vrctrl_st.title = s.req.title.empty() ? s.req.str("title", "") : s.req.title;
    s_vrctrl_st.position = s.st.position;
    s_vrctrl_st.duration = s.st.duration;
    s_vrctrl_st.paused = s.st.paused;
    const bool menu = s_vrmenu.is_open(), ctrl = s_vrctrl.is_open() && !menu;
    if (menu)
        s_vrhand.update(s_vrplace.shown ? &s_vrplace : nullptr, 2.0f, 2.0f, (float)VR_UI_H / VR_UI_W);
    else if (ctrl)
        s_vrhand.update(s_ctrlplace.shown ? &s_ctrlplace : nullptr, CTRL_DIST, CTRL_WIDTH,
                        (float)VR_CTRL_H / VR_CTRL_W);
    else
        s_vrhand.update(nullptr, 0, 0, 0);
    eng_agc_vr_panel_cursor((menu || ctrl) && s_vrhand.valid && !s_vrhand.gaze, s_vrhand.u, s_vrhand.v,
                            s_vrhand.pinching);
    in.pressed |= s_vrhand.buttons_pressed &
                  (APP_BTN_CIRCLE | APP_BTN_TRIANGLE | APP_BTN_OPTIONS | APP_BTN_R3 | APP_BTN_L3);
    if (in.pressed & APP_BTN_L3)
        in.pressed |= APP_BTN_R3;
    if (menu) {
        s_vrmenu.point(s_vrhand, in);
    } else if (ctrl) {
        double to = 0;
        switch (s_vrctrl.point(s_vrhand, s.st.duration, &to)) {
        case VrControls::PAUSE: cmds.push_back({OsdCmd::TogglePause}); break;
        case VrControls::BACK10: cmds.push_back({OsdCmd::SeekTo, std::max(0.0, s.st.position - 10.0)}); break;
        case VrControls::FWD10: cmds.push_back({OsdCmd::SeekTo, s.st.position + 10.0}); break;
        case VrControls::SEEK: cmds.push_back({OsdCmd::SeekTo, to}); break;
        case VrControls::VIEW: s_vrctrl.close(); s_vrmenu.open(); break;
        case VrControls::LIBRARY: cmds.push_back({OsdCmd::Stop}); break;
        default: break;
        }
        s_vrctrl.tick();
        if (s_vrhand.tapped)                       /* a pinch away from the bar hides it */
            s_vrctrl.close();
    } else if (s_vrhand.held_long) {
        in.pressed |= APP_BTN_OPTIONS;
    } else if (s_vrhand.tapped) {
        s_vrctrl.open();
    }
    if (in.pressed & APP_BTN_R3) {
        app_vr_recenter();
        s_vrplace.shown = false;
        in.pressed &= ~APP_BTN_R3;
    }
    if (s_vrmenu.is_open()) {
        s_vrmenu.input(in);
        std::memset(&in, 0, sizeof in);
    } else if (in.pressed & APP_BTN_OPTIONS) {
        s_vrmenu.open();
        std::memset(&in, 0, sizeof in);
    }
}

/* Taking the headset off pauses; putting it back on resumes what that paused. */
static void vr_autopause(Session &s, std::vector<OsdCommand> &cmds)
{
    static bool s_auto_paused;
    if (!s.vr_on || !s.started) {
        s_auto_paused = false;
        return;
    }
    const bool worn = app_vr_worn() != 0 || !s.req.vr_autopause;
    if (!worn && !s.st.paused && !s_auto_paused) {
        eng_bt("vr: headset not worn - paused");
        cmds.push_back({OsdCmd::TogglePause});
        s_auto_paused = true;
    } else if (worn && s_auto_paused) {
        eng_bt("vr: headset worn again - playing");
        if (s.st.paused)
            cmds.push_back({OsdCmd::TogglePause});
        s_auto_paused = false;
    }
}
#endif

static bool vr_frame(Session &s, bool have, const pp_video_frame &f, int64_t pts, double now,
                     bool new_frame, bool panel_changed)
{
#ifdef APP_VR
    if (s.vr_wanted && !s.vr_on && !s.vr_failed && s.started && have) {
        app_vr_options o;
        app_vr_get_options(&o);
        if (s.req.vr_sharpen != -1)
            o.sharpen = s.req.vr_sharpen;
        if (s.req.vr_eye > 0)
            o.eye_px = s.req.vr_eye;
        if (s.req.vr_bits != 10)                /* (the session's own unless asked) */
            o.bits = s.req.vr_bits;
        app_vr_set_options(&o);
        if (s.req.vr.empty())                 /* a guess: check it against the picture */
            app_vr_fit_frame(&s.vr_fmt, (int)f.disp_w, (int)f.disp_h);
        if (s.req.vr.empty() && eng_vdec_multiview_active) {
            s.vr_fmt.projection = 0;          /* spatial video: a 3D screen */
            s.vr_fmt.stereo = 1;
        }
        vr_check_depth_half(s.vr_fmt, f);
        vr_ui_set_detected(s.vr_fmt.projection, s.vr_fmt.stereo);
        s_vrplace.shown = false;
        if (app_vr_active() || app_vr_start(&s.vr_fmt, s_user_id) == 0) {
            vr_ui_apply_format();             /* the detected format, the viewer's overrides */
            eng_agc_vr_reset_picture();
            s.vr_on = true;
        } else {
            s.vr_failed = true;
            s_osd.toast("The VR headset could not be started - playing on the TV", now);
        }
    }
    if (!s.vr_on)
        return false;
    /* Sample the video at the slot, not before the wait: the picture shown then
     * follows the slot cadence exactly. */
    app_vr_wait_slot();
    pp_video_frame cur;
    std::memset(&cur, 0, sizeof cur);
    if (pp_playback_get_video_frame(&g_pp_pb, &cur) && cur.ready) {
        have = true;
        pts = g_pp_pb.display_pts_us;
    } else {
        cur = f;
    }
    /* Draw when the picture changed, and at least every other slot (Hmd2 turns
     * the last frame with the head in between). A 30 fps video is then drawn 30
     * times a second, each frame held for exactly four refreshes, and the GPU
     * time saved goes to the decoder, which shares the GPU. */
    static int s_idle_slots;
    const bool changed = (have && pts != s.last_pts) || panel_changed;
    /* (frame generation: the slot after a generated picture shows the real one) */
    if (!changed && s_idle_slots < 1 && eng_agc_vr_has_picture() && !eng_agc_vr_fg_pending()) {
        s_idle_slots++;
        return true;
    }
    s_idle_slots = 0;
    if (app_vr_begin_frame() != 0) {
        usleep(2000);
        return true;
    }
    if (have) {
        const pp_video_frame &f = cur;
        const int is_direct =
            (eng_pb_active_backend() == ENG_VDEC_BACKEND_NATIVE && !f.held && f.uv != nullptr) ? 1 : 0;
        eng_agc_blit_yuv(f.y, f.y_pitch, f.uv, f.uv_pitch, f.u, f.u_pitch, f.v, f.v_pitch,
                         (int)f.coded_w, (int)f.coded_h, (int)f.disp_w, (int)f.disp_h,
                         0, f.ten_bit, f.color_trc, is_direct, pts);
        s.last_pts = pts;
    } else {
        eng_agc_vr_draw();
    }
    eng_agc_runtime_present();
    return true;
#else
    (void)s; (void)have; (void)f; (void)pts; (void)now; (void)new_frame; (void)panel_changed;
    return false;
#endif
}

extern "C" void app_player_run(const char *json)
{
    static Session s_storage;
    Session &s = s_storage;
    s.~Session();
    new (&s) Session();
    if (!app_request_parse(json, s.req)) {
        eng_bt("ps5vr: unusable play request");
        return;
    }
    g_vdec_force_ffmpeg = 0;      /* each playback starts on the hardware decoder */
    {
        /* Decoder test overrides (LAN debug builds of the payload serve them). */
        char *conf = nullptr;
        app_vdec_conf[0] = 0;
        if (app_bridge_get("/api/debug/vdec-conf", &conf) == 200 && conf)
            std::snprintf(app_vdec_conf, sizeof app_vdec_conf, "%s", conf);
        free(conf);
    }
        const int session = ++s_session;
    s_audio_langs = s.req.prefs.audio_langs;
    std::snprintf(app_stream_headers, sizeof app_stream_headers, "%s", s.req.headers.c_str());
    std::snprintf(app_stream_user_agent, sizeof app_stream_user_agent, "%s", s.req.user_agent.c_str());

    app_input_open(s_user_id);
    app_control_set_playing(1);
    app_subs_close();
    app_subs_set_style(&s.req.prefs.style);
    app_subs_set_delay_ms(0);
    for (const AppSubtitleRef &r : s.req.subtitles)
        app_subs_add_external(r.url.c_str(), r.lang.c_str(), r.label.c_str(), r.headers.c_str());
    if (!s.req.subtitle_requests.empty()) {
        SubFetch *f = new SubFetch{s.req.subtitle_requests, s.req.prefs.subtitle_langs, session};
        pthread_t t;
        if (pthread_create(&t, nullptr, subtitle_fetch_thread, f) == 0)
            pthread_detach(t);
        else
            delete f;
    }

    screen = kScreenPlayer;
    eng_agc_runtime_set_player_mode(1);
    s.open_started = now_s();
    s_osd.begin(&s.req, s.open_started);

    s.job.kind = 0;
    s.job.src.url = s.req.url;
    s.job.src.title = s.req.header_title();
    s.job.src.provider = "ps5vr";   /* a network stream: no local resume file */
    s.job.at = s.req.start_position;
    eng_bt("ps5vr: open '%s' at %.1f s", s.req.header_title().c_str(), s.req.start_position);
#ifdef APP_VR
    /* An explicit "vr" field wins ("none"/"off"/"2d" = the TV); otherwise the
     * file name or title says ("..._360_TB.mp4", "VR180 SBS"). */
    {
        const char *vr = s.req.vr.c_str();
        /* "tv" (or "none"/"off") keeps the TV; anything not recognised as VR or
         * 3D plays on a flat screen in the headset - PS5VR is worn while it plays. */
        const bool off = !strcasecmp(vr, "tv") || !strcasecmp(vr, "none") || !strcasecmp(vr, "off");
        s.vr_wanted = !off;
        if (!off && !(app_vr_parse(vr, &s.vr_fmt) ||
                      app_vr_guess(s.req.url.c_str(), &s.vr_fmt) ||
                      app_vr_guess(s.req.title.c_str(), &s.vr_fmt) ||
                      app_vr_guess(s.req.stream_title.c_str(), &s.vr_fmt))) {
            s.vr_fmt.projection = 0;
            s.vr_fmt.stereo = 0;
        }
        eng_bt("ps5vr: vr %s (projection %d stereo %d, field '%s')", s.vr_wanted ? "on" : "off",
               s.vr_fmt.projection, s.vr_fmt.stereo, vr);
    }
#endif
    start_job(s.job);

    double last_report = 0, last_diag = 0, last_auto = 0;
    bool seen_active = false;
    std::vector<OsdCommand> cmds;

    while (!s.done) {
        const double now = now_s();
        s.st.now = now;
        cmds.clear();

        /* Background open / audio switch finished? */
        if (s.job.running && s.job.done) {
            finish_job(s.job);
            s.st.switching = false;
            if (s.job.kind == 0) {
                if (s.cancel) {
                    s.done = true;
                    s.res.state = "stopped";
                } else if (!s.job.ok) {
                    s.failed = true;
                    s.error = s.req.str("open_failed", "This stream could not be opened. The link may have "
                                                       "expired, or the source is unavailable.");
                    eng_bt("ps5vr: open failed");
                } else {
                    s.opened = true;
                    app_subs_open(play_fmt, video_stream_index);
                    refresh_audio(s);
                    auto_select_subtitles(s);
                    s.st.quality_line = quality_line();
                    seen_active = false;
                    eng_bt("ps5vr: open ok - %s", s.st.quality_line.c_str());
                    /* Profile 5 is rebuilt on the GPU; only a build without that
                     * pipeline would still show it in the wrong colours. */
                    if (play_fmt && video_stream_index >= 0 &&
                        dolby_vision_profile(play_fmt->streams[video_stream_index]->codecpar) == 5 &&
                        !eng_agc_runtime_pipeline_valid(ENG_AGC_PIPE_VIDEO_DV5))
                        s_osd.toast(s.req.str("dv5_unsupported",
                                              "Dolby Vision profile 5 can't be shown on PS5 - colours may be off. Try another source."),
                                    now);
                }
            } else {
                refresh_audio(s);
                s.st.quality_line = quality_line();
                if (!s.job.ok)
                    s_osd.toast(s.req.str("audio_switch_failed", "That audio track could not be played"), now);
            }
            if (s.done)
                break;
        }
        const bool engine_ready = s.opened && !s.job.running;

        /* Input. */
        app_input_state in;
        app_input_poll(&in);
        if (app_control_take_stop() || app_control_quit_requested()) {
            if (s.job.running && !s.opened) {
                s.cancel = true;
            } else {
                s.done = true;
                s.res.state = "stopped";
                break;
            }
        }

        /* Status. */
        s.st.started = s.started;
        s.st.error = s.error;
        if (engine_ready && !s.failed) {
            const bool active = s_pb->isActive();
            if (active)
                seen_active = true;
            s.st.paused = s_pb->isPaused();
            s.st.position = s_pb->getPositionSeconds();
            s.st.duration = s_pb->getDurationSeconds();
            int vq = 0, aq = 0, ab = 0;
            eng_pb_queue_depth(&vq, &aq, &ab);
            s.st.buffered = s.st.position + eng_demux_buffered_s();
            if (!s.started) {
                s.st.open_progress = eng_demux_prebuffer_progress();
                s.st.open_stage = s.req.str("buffering", "Buffering\xE2\x80\xA6");
            }
            if (std::fabs(s.st.position - s.last_pos) > 0.05) {
                s.last_pos = s.st.position;
                s.last_pos_change = now;
            }
            s.st.buffering = s.started && !s.st.paused && !eng_pb_is_eof() &&
                             (pb_prebuffer_hold || (vq == 0 && now - s.last_pos_change > 0.7));
            /*
             * Underrun: the network fell behind and the video queue ran dry.
             * Left alone the decoders limp on frame by frame while audio
             * drifts ahead - a stutter for as long as the dip lasts. Hold
             * instead and refill a few seconds, longer each time it recurs
             * (a link that cannot sustain the bitrate wants more cushion per
             * stop, not more stops).
             */
            const bool starving = s.started && !s.st.paused && !pb_prebuffer_hold &&
                                  !eng_pb_is_eof() && video_stream_index >= 0 && vq == 0 &&
                                  !eng_demux_seek_busy() && !s_osd.post_play_active();
            if (!starving) {
                s.vq_empty_since = 0;
            } else if (s.vq_empty_since == 0) {
                s.vq_empty_since = now;
            } else if (now - s.vq_empty_since >= 0.15) {
                static const int64_t kRefillUs[] = {3000000, 6000000, 10000000};
                if (now - s.last_rebuffer_at > 120)
                    s.rebuffers = 0;
                eng_demux_rebuffer(kRefillUs[std::min(s.rebuffers, 2)], 20000);
                s.rebuffers++;
                s.last_rebuffer_at = now;
                s.vq_empty_since = 0;
            }
            /* Hardware decoder fed but silent - packets go in, no picture comes
             * out (6 s and three dozen packets, outside seeks and pauses) - or
             * given up outright: software decoder from the same point. */
            {
                const int consumed = dbg_video_packets - vq;
                const bool watch = !g_vdec_force_ffmpeg && video_stream_index >= 0 &&
                                   eng_pb_active_backend() == ENG_VDEC_BACKEND_NATIVE &&
                                   !s.st.paused && !eng_demux_seek_busy();
                if (!watch || dbg_video_frames != s.wd_frames) {
                    s.wd_frames = dbg_video_frames;
                    s.wd_consumed = consumed;
                    s.wd_since = now;
                }
                if (!g_vdec_force_ffmpeg && eng_pb_active_backend() == ENG_VDEC_BACKEND_NATIVE &&
                    (eng_pb_decode_fatal() ||
                     (watch && now - s.wd_since > 6.0 && consumed - s.wd_consumed > 36))) {
                    reopen_software(s, eng_pb_decode_fatal() ? "gave up" : "returns no pictures");
                    continue;
                }
            }
            if (eng_pb_decode_fatal()) {
                s.failed = true;
                s.error = s.req.str("decode_failed", "This video could not be decoded.");
            }
            /* Ended: the engine stopped by itself, or the last frame has shown. */
            if (!s.failed && s.started && !s_osd.post_play_active() &&
                ((seen_active && !active) ||
                 (eng_pb_is_eof() && vq == 0 &&
                  (now - s.last_frame_at > 0.8 || (s.st.duration > 0 && s.st.position >= s.st.duration - 0.3))))) {
                s.res.state = "ended";
                s_osd.playback_ended(s.st, cmds);
            }
        } else if (!s.started) {
            s.st.open_stage = s.req.str("opening", "Opening stream\xE2\x80\xA6");
            s.st.open_progress = 0;
        }
        if (!s.started && !s.failed && s.opened && now - s.open_started > 30.0) {
            s.failed = true;
            s.error = s.req.str("did_not_start", "Playback did not start. The source may be too slow or unavailable.");
        }
        s.st.error = s.error;

#ifdef APP_VR
        vr_input(s, in, cmds);
        vr_autopause(s, cmds);
#endif
        s_osd.input(in, s.st, cmds);
        s_osd.tick(s.st, cmds);
        for (const OsdCommand &c : cmds) {
            if (c.cmd == OsdCmd::Stop && s.job.running && !s.opened) {
                s.cancel = true;     /* the open finishes, then we leave */
                continue;
            }
            apply(s, c);
            if (s.done)
                break;
        }
        if (s.done)
            break;

        /* Subtitles appear as addon tracks arrive. */
        if (s.opened && now - last_auto > 1.0) {
            last_auto = now;
            auto_select_subtitles(s);
        }

        /* ---- draw ---- */
        /* The headset takes SDR: a VR video never switches the TV to HDR10
         * (that would remap every pipeline to PQ output). */
        update_hdr(s.started && engine_ready && !s.failed && !(s.vr_wanted && !s.vr_failed));
        pp_video_frame f;
        std::memset(&f, 0, sizeof f);
        bool have = false, new_frame = false;
        int64_t pts = s.last_pts;
        if (engine_ready && !s.failed) {
            have = pp_playback_get_video_frame(&g_pp_pb, &f) && f.ready;
            pts = g_pp_pb.display_pts_us;
            new_frame = have && (pts != s.last_pts || g_pp_pb.seek_discarding);
            if (have && !s.started) {
                s.started = true;
                s.st.started = true;
                eng_bt("ps5vr: first frame %ux%u ten_bit=%d trc=%d after %.2f s", f.disp_w, f.disp_h,
                       f.ten_bit, f.color_trc, now - s.open_started);
            }
            if (new_frame)
                s.last_frame_at = now;
        }

        bool sub_changed = false;
        if (s.started && have)
            sub_changed = app_subs_render(&s_sub_canvas, pts, video_rect(f), s_osd.subtitle_lift()) != 0;
        const bool osd_changed = s_osd.render(s_osd_canvas, s.st);
        const bool stale = have && eng_agc_runtime_video_slot_stale(pts);

#ifdef APP_VR
        const bool panel_changed = s.vr_on &&
            vr_panel_update(have && app_subs_visible(), sub_changed, osd_changed);
#else
        const bool panel_changed = false;
#endif
        if (vr_frame(s, have, f, pts, now, new_frame, panel_changed)) {
            /* drawn to the headset (or waiting for it) */
        } else if (new_frame || stale || sub_changed || osd_changed) {
            eng_agc_runtime_frame_begin();
            eng_agc_runtime_clear_black();
            if (have) {
                const int is_direct =
                    (eng_pb_active_backend() == ENG_VDEC_BACKEND_NATIVE && !f.held && f.uv != nullptr) ? 1 : 0;
                eng_agc_blit_yuv(f.y, f.y_pitch, f.uv, f.uv_pitch, f.u, f.u_pitch, f.v, f.v_pitch,
                                 (int)f.coded_w, (int)f.coded_h, (int)f.disp_w, (int)f.disp_h,
                                 (int)s_pb->getViewMode(), f.ten_bit, f.color_trc, is_direct, pts);
                s.last_pts = pts;
            }
            if (have && (app_subs_visible() || sub_changed))
                eng_agc_composite_overlay(0, s_sub_canvas.px, CW, CH, sub_changed ? 1 : 0, 1.0f);
            if (s_osd.visible())
                eng_agc_composite_overlay(1, s_osd_canvas.px, CW, CH, osd_changed ? 1 : 0, 1.0f);
            eng_agc_runtime_present();
        } else {
            usleep(2000);
        }

        if (now - last_report >= 1.0 && s.started) {
            last_report = now;
            app_control_report(s.req.id.c_str(), s.st.position, s.st.duration);
        }
        if (now - last_diag >= 5.0 && engine_ready) {
            last_diag = now;
            int vq = 0, aq = 0, ab = 0;
            eng_pb_queue_depth(&vq, &aq, &ab);
            eng_bt("ps5vr: pos=%.1f/%.1f paused=%d buffering=%d vq=%d aq=%d ab=%d buf=%.1fs demux=%d eof=%d ringfb=%lu vpk=%d vfr=%d ctl=%u/%d vclk=%.2f aclk=%.2f",
                   s.st.position, s.st.duration, (int)s.st.paused, (int)s.st.buffering, vq, aq, ab,
                   eng_demux_buffered_s(), eng_demux_state, eng_pb_is_eof(),
                   (unsigned long)packet_queue_ring_fallbacks,
                   dbg_video_packets, dbg_video_frames, app_control_beats, app_control_stage,
                   eng_pb_video_clock_s(), eng_pb_audio_clock_s());
            eng_log_alloc_state("play");
        }
    }

    /* ---- leave ---- */
    if (s.vr_on) {                    /* the headset goes back to the library */
        eng_agc_vr_reset_picture();
        eng_agc_vr_panel(nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        s_vrmenu.close();
        s_vrctrl.close();
        eng_agc_vr_panel_cursor(0, 0, 0, 0);
        s.vr_on = false;
    }
    s_session++;
    finish_job(s.job);
    s.res.position = s.started ? s_pb->getPositionSeconds() : s.req.start_position;
    s.res.duration = s_pb->getDurationSeconds();
    if (s.failed && s.res.error.empty())
        s.res.error = s.error;
    if (s.res.state.empty())
        s.res.state = s.failed ? "error" : "stopped";
    if (!s.st.audio.empty() && s.st.audio_active >= 0)
        s.res.audio_lang = s.st.audio[s.st.audio_active].lang;
    {
        app_sub_track t;
        const int sel = app_subs_selected();
        s.res.subtitles_on = sel >= 0;
        if (sel >= 0 && app_subs_track(sel, &t) == 0)
            s.res.subtitle_lang = t.lang;
        s.res.subtitle_delay_ms = app_subs_delay_ms();
    }
    eng_bt("ps5vr: %s at %.1f / %.1f s%s%s%s%s", s.res.state.c_str(), s.res.position, s.res.duration,
           s.res.action.empty() ? "" : " -> ", s.res.action.c_str(), s.res.error.empty() ? "" : " - ",
           s.res.error.c_str());
    s_pb->stopPlayback();
    screen = kScreenNone;
    app_subs_close();
    ui_image_clear();
    s_osd.end();
    update_hdr(false);
    eng_agc_runtime_set_player_mode(0);
    eng_agc_runtime_frame_begin();
    eng_agc_runtime_clear_black();
    eng_agc_runtime_present();
    app_control_set_playing(0);
    app_input_close();
    eng_boot_log_flush();
    app_bridge_state_json(app_result_json(s.req, s.res).c_str());
}
