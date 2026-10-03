/*
 * vr_sources - where PS5VR finds videos: folders (this PS5, USB, extended
 * storage), DLNA/UPnP media servers, RSS feeds, Real-Debrid, and DeoVR /
 * HereSphere library sites. Every call blocks (network, disk); the library
 * runs them on a worker thread.
 */
#pragma once

#include <string>
#include <vector>

enum VrSrcKind {
    SRC_ROOT,        /* the list of sources */
    SRC_DIR,         /* key = folder path */
    SRC_DLNA_LIST,   /* the servers on the network */
    SRC_DLNA,        /* key = container id, base = ContentDirectory control URL */
    SRC_RSS_LIST,    /* the configured feeds */
    SRC_RSS,         /* key = feed URL */
    SRC_RD,          /* Real-Debrid downloads */
    SRC_SITE,        /* key = DeoVR / HereSphere library URL */
    SRC_SITE_GROUP,  /* key = site URL, base = group name */
    SRC_SITE_VIDEO,  /* key = a scene URL: resolved to its file when chosen */
    SRC_PLAY,        /* a playable item: url (+ vr format) */
    SRC_INFO,        /* a line of information, not selectable */
    SRC_HANDS,       /* the hands & eyes switch */
    SRC_MIRROR,      /* the mirror-to-TV switch */
};

struct VrSrc {
    VrSrcKind kind = SRC_ROOT;
    std::string name, detail;
    std::string key, base;   /* meaning per kind */
    std::string url, vr;     /* SRC_PLAY: what to open, and its format if known */
};

/* The entries of `where` (a SRC_* container). false + err when it failed. */
bool vr_src_list(const VrSrc &where, std::vector<VrSrc> &out, std::string &err);
/* A SRC_SITE_VIDEO chosen: its playable file (url, vr, name). */
bool vr_src_resolve(const VrSrc &video, VrSrc &play, std::string &err);

/* Hands & eyes control (experimental), kept in /data/ps5vr/prefs.json. */
/* Hands & eyes control (experimental): 0 off, 1 hands (ray + pinch),
 * 2 eyes + pinch, 3 eyes + blink (no hands needed). Controllers always work. */
int  vr_prefs_control(void);
void vr_prefs_set_control(int mode);
const char *vr_control_name(int mode);
/* HDR in the headset (experimental): RGBA16F eye buffers. */
bool vr_prefs_hdr(void);
void vr_prefs_set_hdr(bool on);
bool vr_prefs_mirror(void);
void vr_prefs_set_mirror(bool on);

/* The settings (sites, feeds, Real-Debrid token) in /data/ps5vr/config.json,
 * edited from a phone or computer at http://<console>:<port>/. */
void vr_settings_server_start(void);
int  vr_settings_server_port(void);
/* A play request posted to http://<console>:<port>/play ({"url","title","vr"}). */
bool vr_settings_take_play(std::string &json);
/* A play request is waiting (the player then gives way to it). */
bool vr_settings_play_pending(void);
std::string vr_console_ip(void);
