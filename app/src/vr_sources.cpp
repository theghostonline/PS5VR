/*
 * vr_sources - see vr_sources.h.
 */
#include "vr_sources.h"

#include "cJSON.h"
#include "eng_boot_trace.h"
#include "eng_readdir.h"

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
}
#include <libxml/parser.h>
#include <libxml/tree.h>

#include <algorithm>
#include <atomic>
#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#define PS5VR_DIR    "/data/ps5vr"
#define PS5VR_VIDEOS PS5VR_DIR "/videos"
#define PS5VR_CONFIG PS5VR_DIR "/config.json"

namespace {

/* -- small helpers ------------------------------------------------------------- */
const char *kVideoExt[] = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".ts", ".m2ts", ".mts",
                           ".avi", ".264", ".h264", ".265", ".hevc", ".wmv", ".flv", ".mpg",
                           ".mpeg", ".mk3d"};

bool is_video_name(const std::string &n)
{
    const size_t q = n.find_first_of("?#");
    const std::string s = n.substr(0, q);
    const size_t dot = s.rfind('.');
    if (dot == std::string::npos)
        return false;
    for (const char *e : kVideoExt)
        if (!strcasecmp(s.c_str() + dot, e))
            return true;
    return false;
}

std::string human_size(double b)
{
    char s[32];
    if (b >= 1073741824.0)
        std::snprintf(s, sizeof s, "%.1f GB", b / 1073741824.0);
    else if (b >= 1048576.0)
        std::snprintf(s, sizeof s, "%.0f MB", b / 1048576.0);
    else
        std::snprintf(s, sizeof s, "%.0f KB", b / 1024.0);
    return s;
}

/* HTTP(S) through FFmpeg's protocols: GET, or POST when body is given. */
bool http_fetch(const std::string &url, const std::string &headers, const std::string &body,
                std::string &out, std::string &err)
{
    AVDictionary *o = nullptr;
    av_dict_set(&o, "rw_timeout", "10000000", 0);      /* 10 s */
    av_dict_set(&o, "user_agent", "PS5VR/1.0", 0);
    if (!headers.empty())
        av_dict_set(&o, "headers", headers.c_str(), 0);
    if (!body.empty()) {
        static const char hx[] = "0123456789abcdef";
        std::string h;
        for (unsigned char c : body) { h += hx[c >> 4]; h += hx[c & 15]; }
        av_dict_set(&o, "method", "POST", 0);
        av_dict_set(&o, "post_data", h.c_str(), 0);
    }
    AVIOContext *pb = nullptr;
    const int r = avio_open2(&pb, url.c_str(), AVIO_FLAG_READ, nullptr, &o);
    av_dict_free(&o);
    if (r < 0) {
        char e[128];
        av_strerror(r, e, sizeof e);
        err = std::string("Could not reach ") + url + " (" + e + ")";
        return false;
    }
    out.clear();
    char buf[16384];
    for (;;) {
        const int n = avio_read(pb, (unsigned char *)buf, sizeof buf);
        if (n <= 0)
            break;
        out.append(buf, (size_t)n);
        if (out.size() > (16u << 20))
            break;
    }
    avio_closep(&pb);
    return true;
}

std::string resolve_url(const std::string &base, const std::string &rel)
{
    if (rel.find("://") != std::string::npos)
        return rel;
    const size_t scheme = base.find("://");
    if (scheme == std::string::npos)
        return rel;
    const size_t host_end = base.find('/', scheme + 3);
    const std::string origin = base.substr(0, host_end);
    if (!rel.empty() && rel[0] == '/')
        return origin + rel;
    const size_t last = base.rfind('/');
    return (last != std::string::npos && last > scheme + 2 ? base.substr(0, last + 1) : origin + "/") + rel;
}

/* libxml2: first child element (any depth) with this local name. */
xmlNode *find(xmlNode *n, const char *name)
{
    for (; n; n = n->next) {
        if (n->type == XML_ELEMENT_NODE && !strcmp((const char *)n->name, name))
            return n;
        if (xmlNode *c = find(n->children, name))
            return c;
    }
    return nullptr;
}

std::string text(xmlNode *n)
{
    if (!n)
        return "";
    xmlChar *t = xmlNodeGetContent(n);
    std::string s = t ? (const char *)t : "";
    xmlFree(t);
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    size_t i = 0;
    while (i < s.size() && isspace((unsigned char)s[i])) i++;
    return s.substr(i);
}

std::string attr(xmlNode *n, const char *name)
{
    if (!n)
        return "";
    xmlChar *v = xmlGetProp(n, (const xmlChar *)name);
    std::string s = v ? (const char *)v : "";
    xmlFree(v);
    return s;
}

/* -- config ------------------------------------------------------------------------ */
struct Config {
    std::vector<std::pair<std::string, std::string>> feeds, sites;   /* (name, url) */
    std::string rd_token;
};

Config load_config()
{
    Config c;
    FILE *f = std::fopen(PS5VR_CONFIG, "rb");
    if (!f)
        return c;
    std::string s;
    char b[4096];
    size_t n;
    while ((n = std::fread(b, 1, sizeof b, f)) > 0)
        s.append(b, n);
    std::fclose(f);
    cJSON *j = cJSON_Parse(s.c_str());
    if (!j)
        return c;
    auto list = [&](const char *key, std::vector<std::pair<std::string, std::string>> &out) {
        cJSON *a = cJSON_GetObjectItem(j, key);
        cJSON *it;
        cJSON_ArrayForEach(it, a) {
            const char *u = cJSON_GetStringValue(cJSON_GetObjectItem(it, "url"));
            const char *nm = cJSON_GetStringValue(cJSON_GetObjectItem(it, "name"));
            if (u && *u)
                out.push_back({nm && *nm ? nm : u, u});
        }
    };
    list("feeds", c.feeds);
    list("sites", c.sites);
    if (const char *t = cJSON_GetStringValue(cJSON_GetObjectItem(j, "realdebrid")))
        c.rd_token = t;
    cJSON_Delete(j);
    return c;
}

/* -- folders --------------------------------------------------------------------- */
bool list_dir(const std::string &dir, std::vector<VrSrc> &out, std::string &err)
{
    /* opendir() is EPERM in an app; eng_opendir reads with getdents */
    eng_dir_t *d = eng_opendir(dir.c_str());
    if (!d) {
        eng_bt("library: open %s errno %d", dir.c_str(), errno);
        err = "This folder could not be opened";
        return false;
    }
    std::vector<VrSrc> dirs, files;
    while (dirent *e = eng_readdir(d)) {
        if (e->d_name[0] == '.')
            continue;
        VrSrc it;
        it.name = e->d_name;
        const std::string path = dir + "/" + e->d_name;
        struct stat st;
        if (stat(path.c_str(), &st) != 0)
            continue;
        if (S_ISDIR(st.st_mode)) {
            it.kind = SRC_DIR;
            it.key = path;
            dirs.push_back(it);
        } else if (is_video_name(e->d_name)) {
            it.kind = SRC_PLAY;
            it.url = path;
            it.detail = human_size((double)st.st_size);
            files.push_back(it);
        }
    }
    eng_closedir(d);
    auto by_name = [](const VrSrc &a, const VrSrc &b) { return strcasecmp(a.name.c_str(), b.name.c_str()) < 0; };
    std::sort(dirs.begin(), dirs.end(), by_name);
    std::sort(files.begin(), files.end(), by_name);
    out = dirs;
    out.insert(out.end(), files.begin(), files.end());
    return true;
}

/* -- DLNA / UPnP ------------------------------------------------------------------- */
struct DlnaServer { std::string name, control; };

std::vector<DlnaServer> dlna_discover()
{
    std::vector<std::string> locations;
    const int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s >= 0) {
        timeval tv = {0, 300000};
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        unsigned char ttl = 4;
        setsockopt(s, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof ttl);
        sockaddr_in to;
        std::memset(&to, 0, sizeof to);
        to.sin_family = AF_INET;
        to.sin_port = htons(1900);
        to.sin_addr.s_addr = inet_addr("239.255.255.250");
        const char *msg = "M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\n"
                          "MAN: \"ssdp:discover\"\r\nMX: 2\r\n"
                          "ST: urn:schemas-upnp-org:device:MediaServer:1\r\n\r\n";
        for (int round = 0; round < 8; round++) {          /* ~2.4 s, re-asking twice */
            if (round % 3 == 0)
                sendto(s, msg, std::strlen(msg), 0, (sockaddr *)&to, sizeof to);
            char buf[2048];
            const ssize_t n = recv(s, buf, sizeof buf - 1, 0);
            if (n <= 0)
                continue;
            buf[n] = 0;
            for (char *line = std::strtok(buf, "\r\n"); line; line = std::strtok(nullptr, "\r\n"))
                if (!strncasecmp(line, "LOCATION:", 9)) {
                    std::string loc = line + 9;
                    loc.erase(0, loc.find_first_not_of(' '));
                    if (std::find(locations.begin(), locations.end(), loc) == locations.end())
                        locations.push_back(loc);
                }
        }
        close(s);
    }
    std::vector<DlnaServer> out;
    for (const std::string &loc : locations) {
        std::string xml, err;
        if (!http_fetch(loc, "", "", xml, err))
            continue;
        xmlDoc *doc = xmlReadMemory(xml.data(), (int)xml.size(), nullptr, nullptr, XML_PARSE_NOERROR | XML_PARSE_NOWARNING);
        if (!doc)
            continue;
        xmlNode *root = xmlDocGetRootElement(doc);
        DlnaServer sv;
        sv.name = text(find(root, "friendlyName"));
        const std::string base = text(find(root, "URLBase"));
        for (xmlNode *svc = find(root, "service"); svc; svc = svc->next) {
            if (svc->type != XML_ELEMENT_NODE)
                continue;
            if (text(find(svc->children, "serviceType")).find("ContentDirectory") != std::string::npos) {
                sv.control = resolve_url(base.empty() ? loc : base, text(find(svc->children, "controlURL")));
                break;
            }
        }
        xmlFreeDoc(doc);
        if (!sv.control.empty())
            out.push_back(sv);
    }
    return out;
}

bool dlna_browse(const std::string &control, const std::string &id, std::vector<VrSrc> &out,
                 std::string &err)
{
    std::string esc;
    for (char c : id) {
        if (c == '&') esc += "&amp;"; else if (c == '<') esc += "&lt;"; else if (c == '>') esc += "&gt;"; else esc += c;
    }
    const std::string body =
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
        "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" "
        "s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\"><s:Body>"
        "<u:Browse xmlns:u=\"urn:schemas-upnp-org:service:ContentDirectory:1\">"
        "<ObjectID>" + esc + "</ObjectID><BrowseFlag>BrowseDirectChildren</BrowseFlag>"
        "<Filter>*</Filter><StartingIndex>0</StartingIndex><RequestedCount>1000</RequestedCount>"
        "<SortCriteria></SortCriteria></u:Browse></s:Body></s:Envelope>";
    const std::string headers =
        "Content-Type: text/xml; charset=\"utf-8\"\r\n"
        "SOAPACTION: \"urn:schemas-upnp-org:service:ContentDirectory:1#Browse\"\r\n";
    std::string resp;
    if (!http_fetch(control, headers, body, resp, err))
        return false;
    xmlDoc *doc = xmlReadMemory(resp.data(), (int)resp.size(), nullptr, nullptr, XML_PARSE_NOERROR | XML_PARSE_NOWARNING);
    const std::string didl = doc ? text(find(xmlDocGetRootElement(doc), "Result")) : "";
    if (doc)
        xmlFreeDoc(doc);
    xmlDoc *d = xmlReadMemory(didl.data(), (int)didl.size(), nullptr, nullptr, XML_PARSE_NOERROR | XML_PARSE_NOWARNING);
    if (!d) {
        err = "The server's answer could not be read";
        return false;
    }
    for (xmlNode *n = xmlDocGetRootElement(d)->children; n; n = n->next) {
        if (n->type != XML_ELEMENT_NODE)
            continue;
        VrSrc it;
        it.name = text(find(n->children, "title"));
        if (!strcmp((const char *)n->name, "container")) {
            it.kind = SRC_DLNA;
            it.key = attr(n, "id");
            it.base = control;
            out.push_back(it);
        } else if (!strcmp((const char *)n->name, "item")) {
            /* the first video resource */
            for (xmlNode *r = n->children; r; r = r->next) {
                if (r->type != XML_ELEMENT_NODE || strcmp((const char *)r->name, "res"))
                    continue;
                const std::string proto = attr(r, "protocolInfo");
                if (proto.find("video") == std::string::npos && !is_video_name(text(r)))
                    continue;
                it.kind = SRC_PLAY;
                it.url = text(r);
                const std::string size = attr(r, "size"), res = attr(r, "resolution");
                it.detail = res.empty() ? "" : res;
                if (!size.empty())
                    it.detail += (it.detail.empty() ? "" : "  ") + human_size(atof(size.c_str()));
                out.push_back(it);
                break;
            }
        }
    }
    xmlFreeDoc(d);
    return true;
}

/* -- RSS / Atom ---------------------------------------------------------------------- */
bool rss_list(const std::string &url, std::vector<VrSrc> &out, std::string &err)
{
    std::string xml;
    if (!http_fetch(url, "", "", xml, err))
        return false;
    xmlDoc *doc = xmlReadMemory(xml.data(), (int)xml.size(), nullptr, nullptr, XML_PARSE_NOERROR | XML_PARSE_NOWARNING);
    if (!doc) {
        err = "This feed could not be read";
        return false;
    }
    xmlNode *root = xmlDocGetRootElement(doc);
    const bool atom = !strcmp((const char *)root->name, "feed");
    for (xmlNode *ch = atom ? root->children : (find(root, "channel") ? find(root, "channel")->children : nullptr);
         ch; ch = ch->next) {
        if (ch->type != XML_ELEMENT_NODE || strcmp((const char *)ch->name, atom ? "entry" : "item"))
            continue;
        VrSrc it;
        it.kind = SRC_PLAY;
        it.name = text(find(ch->children, "title"));
        for (xmlNode *c = ch->children; c && it.url.empty(); c = c->next) {
            if (c->type != XML_ELEMENT_NODE)
                continue;
            const char *nm = (const char *)c->name;
            if (!strcmp(nm, "enclosure") || !strcmp(nm, "content")) {
                const std::string type = attr(c, "type"), u = attr(c, "url");
                if (!u.empty() && (type.find("video") != std::string::npos || is_video_name(u))) {
                    it.url = u;
                    const std::string len = attr(c, "length").empty() ? attr(c, "fileSize") : attr(c, "length");
                    if (!len.empty() && atof(len.c_str()) > 0)
                        it.detail = human_size(atof(len.c_str()));
                }
            } else if (!strcmp(nm, "link")) {
                const std::string href = atom ? attr(c, "href") : text(c);
                if (is_video_name(href) && (attr(c, "rel").empty() || attr(c, "rel") == "enclosure"))
                    it.url = href;
            }
        }
        if (!it.url.empty()) {
            it.url = resolve_url(url, it.url);
            if (it.name.empty())
                it.name = it.url.substr(it.url.rfind('/') + 1);
            out.push_back(it);
        }
    }
    xmlFreeDoc(doc);
    if (out.empty())
        err = "No videos in this feed";
    return true;
}

/* -- Real-Debrid ---------------------------------------------------------------------- */
bool rd_list(const std::string &token, std::vector<VrSrc> &out, std::string &err)
{
    std::string body;
    if (!http_fetch("https://api.real-debrid.com/rest/1.0/downloads?limit=100",
                    "Authorization: Bearer " + token + "\r\n", "", body, err))
        return false;
    cJSON *a = cJSON_Parse(body.c_str());
    if (!cJSON_IsArray(a)) {
        err = "Real-Debrid did not accept the token";
        cJSON_Delete(a);
        return false;
    }
    cJSON *it;
    cJSON_ArrayForEach(it, a) {
        const char *fn = cJSON_GetStringValue(cJSON_GetObjectItem(it, "filename"));
        const char *dl = cJSON_GetStringValue(cJSON_GetObjectItem(it, "download"));
        if (!fn || !dl || !is_video_name(fn))
            continue;
        VrSrc v;
        v.kind = SRC_PLAY;
        v.name = fn;
        v.url = dl;
        if (cJSON *sz = cJSON_GetObjectItem(it, "filesize"))
            v.detail = human_size(sz->valuedouble);
        out.push_back(v);
    }
    cJSON_Delete(a);
    if (out.empty())
        err = "No videos in your Real-Debrid downloads";
    return true;
}

/* -- DeoVR / HereSphere ------------------------------------------------------------------ */
bool heresphere_url(const std::string &u) { return u.find("heresphere") != std::string::npos; }

/* (projection, stereo) hints -> the player's "vr" field */
std::string vr_of(const std::string &proj, const std::string &stereo)
{
    std::string p = "180";
    if (proj == "sphere" || proj == "equirectangular360" || proj == "360") p = "360";
    else if (proj == "flat" || proj == "perspective" || proj == "rectilinear") p = "flat";
    std::string s;
    if (stereo == "sbs" || stereo == "LR") s = "_sbs";
    else if (stereo == "tb" || stereo == "TB") s = "_tb";
    return p + s;
}

bool site_fetch(const std::string &url, std::string &body, std::string &err)
{
    return http_fetch(url, heresphere_url(url) ? "Content-Type: application/json\r\n" : "",
                      heresphere_url(url) ? "{}" : "", body, err);
}

bool site_groups(const std::string &url, std::vector<VrSrc> &out, std::string &err)
{
    std::string body;
    if (!site_fetch(url, body, err))
        return false;
    cJSON *j = cJSON_Parse(body.c_str());
    cJSON *groups = j ? cJSON_GetObjectItem(j, heresphere_url(url) ? "library" : "scenes") : nullptr;
    if (!cJSON_IsArray(groups)) {
        err = "This is not a DeoVR or HereSphere library";
        cJSON_Delete(j);
        return false;
    }
    cJSON *g;
    cJSON_ArrayForEach(g, groups) {
        VrSrc it;
        it.kind = SRC_SITE_GROUP;
        const char *n = cJSON_GetStringValue(cJSON_GetObjectItem(g, "name"));
        it.name = n ? n : "Videos";
        it.key = url;
        it.base = it.name;
        cJSON *l = cJSON_GetObjectItem(g, "list");
        it.detail = std::to_string(cJSON_GetArraySize(l)) + " videos";
        out.push_back(it);
    }
    cJSON_Delete(j);
    return true;
}

/* HereSphere lists only scene URLs: ask the first ones for their titles,
 * several at a time ({"needsMediaSource": false} keeps the answers small). */
struct TitleJob {
    std::vector<VrSrc> *items;
    std::atomic<int> next{0};
    int count = 0;
};

void *title_worker(void *arg)
{
    TitleJob *t = (TitleJob *)arg;
    for (int i; (i = t->next++) < t->count;) {
        VrSrc &it = (*t->items)[i];
        std::string body, err;
        if (!http_fetch(it.key, "Content-Type: application/json\r\n", "{\"needsMediaSource\":false}", body, err))
            continue;
        cJSON *j = cJSON_Parse(body.c_str());
        if (const char *title = cJSON_GetStringValue(cJSON_GetObjectItem(j, "title")))
            if (*title)
                it.name = title;
        if (cJSON *d = cJSON_GetObjectItem(j, "duration")) {        /* milliseconds */
            const int sec = (int)(cJSON_GetNumberValue(d) / 1000.0);
            if (sec > 0) {
                char b[32];
                std::snprintf(b, sizeof b, "%d:%02d", sec / 60, sec % 60);
                it.detail = b;
            }
        }
        cJSON_Delete(j);
    }
    return nullptr;
}

void heresphere_titles(std::vector<VrSrc> &items)
{
    TitleJob t;
    t.items = &items;
    t.count = (int)std::min<size_t>(items.size(), 48);
    pthread_t th[8];
    int n = 0;
    for (; n < 8 && n < t.count; n++)
        if (pthread_create(&th[n], nullptr, title_worker, &t) != 0)
            break;
    if (n == 0)
        title_worker(&t);
    for (int i = 0; i < n; i++)
        pthread_join(th[i], nullptr);
}

bool site_group(const std::string &url, const std::string &group, std::vector<VrSrc> &out,
                std::string &err)
{
    std::string body;
    if (!site_fetch(url, body, err))
        return false;
    cJSON *j = cJSON_Parse(body.c_str());
    const bool hs = heresphere_url(url);
    cJSON *groups = j ? cJSON_GetObjectItem(j, hs ? "library" : "scenes") : nullptr;
    cJSON *g;
    cJSON_ArrayForEach(g, groups) {
        const char *n = cJSON_GetStringValue(cJSON_GetObjectItem(g, "name"));
        if ((n ? n : "Videos") != group)
            continue;
        cJSON *v;
        cJSON_ArrayForEach(v, cJSON_GetObjectItem(g, "list")) {
            VrSrc it;
            it.kind = SRC_SITE_VIDEO;
            if (cJSON_IsString(v)) {                   /* HereSphere: scene URLs */
                it.key = v->valuestring;
                it.name = "Video " + std::to_string(out.size() + 1);
            } else {                                   /* DeoVR: scene objects */
                const char *t = cJSON_GetStringValue(cJSON_GetObjectItem(v, "title"));
                const char *u = cJSON_GetStringValue(cJSON_GetObjectItem(v, "video_url"));
                if (!u)
                    continue;
                it.key = u;
                it.name = t ? t : u;
                if (cJSON *len = cJSON_GetObjectItem(v, "videoLength")) {
                    char d[32];
                    std::snprintf(d, sizeof d, "%d:%02d", (int)len->valuedouble / 60, (int)len->valuedouble % 60);
                    it.detail = d;
                }
            }
            it.base = hs ? "heresphere" : "deovr";
            out.push_back(it);
        }
    }
    cJSON_Delete(j);
    if (hs)
        heresphere_titles(out);
    return true;
}

} // namespace

bool vr_src_resolve(const VrSrc &video, VrSrc &play, std::string &err)
{
    std::string body;
    const bool hs = video.base == "heresphere";
    if (!http_fetch(video.key, hs ? "Content-Type: application/json\r\n" : "", hs ? "{}" : "", body, err))
        return false;
    cJSON *j = cJSON_Parse(body.c_str());
    if (!j) {
        err = "This video's details could not be read";
        return false;
    }
    play.kind = SRC_PLAY;
    const char *title = cJSON_GetStringValue(cJSON_GetObjectItem(j, "title"));
    play.name = title ? title : video.name;
    /* the highest resolution up to 8K: the PS5's decoder limit */
    int best = -1;
    auto consider = [&](cJSON *src, int res) {
        const char *u = cJSON_GetStringValue(cJSON_GetObjectItem(src, "url"));
        if (u && res <= 4320 && res > best) {
            best = res;
            play.url = u;
        }
    };
    cJSON *enc, *src;
    if (hs) {
        cJSON_ArrayForEach(enc, cJSON_GetObjectItem(j, "media"))
            cJSON_ArrayForEach(src, cJSON_GetObjectItem(enc, "sources"))
                consider(src, (int)cJSON_GetNumberValue(cJSON_GetObjectItem(src, "height")));
        const char *p = cJSON_GetStringValue(cJSON_GetObjectItem(j, "projection"));
        const char *s = cJSON_GetStringValue(cJSON_GetObjectItem(j, "stereo"));
        play.vr = vr_of(p ? p : "", s ? s : "");
    } else {
        cJSON_ArrayForEach(enc, cJSON_GetObjectItem(j, "encodings"))
            cJSON_ArrayForEach(src, cJSON_GetObjectItem(enc, "videoSources"))
                consider(src, (int)cJSON_GetNumberValue(cJSON_GetObjectItem(src, "resolution")));
        const char *p = cJSON_GetStringValue(cJSON_GetObjectItem(j, "screenType"));
        const char *s = cJSON_GetStringValue(cJSON_GetObjectItem(j, "stereoMode"));
        play.vr = vr_of(p ? p : "", s ? s : "");
    }
    cJSON_Delete(j);
    if (play.url.empty()) {
        err = "No playable file for this video";
        return false;
    }
    return true;
}

/* /mnt/usbN and /mnt/extN always exist; a drive is there when something is
 * mounted on them (another device than /mnt itself). */
static bool mounted(const char *path)
{
    struct stat mnt, st;
    return stat("/mnt", &mnt) == 0 && stat(path, &st) == 0 && S_ISDIR(st.st_mode) &&
           st.st_dev != mnt.st_dev;
}

/* Choices made in the headset, kept in /data/ps5vr/prefs.json. */
#define PS5VR_PREFS PS5VR_DIR "/prefs.json"
static int s_prefs_loaded, s_prefs_control, s_prefs_hdr, s_prefs_mirror;

static void prefs_load(void)
{
    if (s_prefs_loaded)
        return;
    s_prefs_loaded = 1;
    if (FILE *f = std::fopen(PS5VR_PREFS, "rb")) {
        char b[256] = {0};
        std::fread(b, 1, sizeof b - 1, f);
        std::fclose(f);
        if (cJSON *j = cJSON_Parse(b)) {
            /* control: 0 off, 1 hands, 2 eyes + pinch, 3 eyes + blink (1.1.0 had handsEyes) */
            if (cJSON *c = cJSON_GetObjectItem(j, "control"))
                s_prefs_control = (int)cJSON_GetNumberValue(c);
            else
                s_prefs_control = cJSON_IsTrue(cJSON_GetObjectItem(j, "handsEyes")) ? 2 : 0;
            if (s_prefs_control < 0 || s_prefs_control > 3)
                s_prefs_control = 0;
            s_prefs_hdr = cJSON_IsTrue(cJSON_GetObjectItem(j, "hdr"));
            s_prefs_mirror = cJSON_IsTrue(cJSON_GetObjectItem(j, "mirror"));
            cJSON_Delete(j);
        }
    }
}

static void prefs_save(void)
{
    mkdir(PS5VR_DIR, 0777);
    if (FILE *f = std::fopen(PS5VR_PREFS, "wb")) {
        std::fprintf(f, "{\"control\":%d,\"hdr\":%s,\"mirror\":%s}\n", s_prefs_control,
                     s_prefs_hdr ? "true" : "false", s_prefs_mirror ? "true" : "false");
        std::fclose(f);
    }
}

int vr_prefs_control(void) { prefs_load(); return s_prefs_control; }
void vr_prefs_set_control(int mode) { prefs_load(); s_prefs_control = mode; prefs_save(); }
const char *vr_control_name(int mode)
{
    static const char *const k[] = {"Off", "Hands", "Eyes + pinch", "Eyes + blink"};
    return k[mode >= 0 && mode < 4 ? mode : 0];
}
bool vr_prefs_hdr(void) { prefs_load(); return s_prefs_hdr; }
void vr_prefs_set_hdr(bool on) { prefs_load(); s_prefs_hdr = on; prefs_save(); }
bool vr_prefs_mirror(void) { prefs_load(); return s_prefs_mirror; }
void vr_prefs_set_mirror(bool on) { prefs_load(); s_prefs_mirror = on; prefs_save(); }

bool vr_src_list(const VrSrc &where, std::vector<VrSrc> &out, std::string &err)
{
    out.clear();
    switch (where.kind) {
    case SRC_ROOT: {
        mkdir(PS5VR_DIR, 0777);
        mkdir(PS5VR_VIDEOS, 0777);
        auto add = [&](VrSrcKind k, const std::string &n, const std::string &key, const std::string &d) {
            VrSrc s;
            s.kind = k;
            s.name = n;
            s.key = key;
            s.detail = d;
            out.push_back(s);
        };
        add(SRC_DIR, "This PS5", PS5VR_VIDEOS, PS5VR_VIDEOS);
        for (int i = 0; i < 8; i++) {
            char p[32];
            std::snprintf(p, sizeof p, "/mnt/usb%d", i);
            if (mounted(p))
                add(SRC_DIR, "USB drive " + std::to_string(i + 1), p, p);
        }
        for (int i = 0; i < 2; i++) {
            char p[32];
            std::snprintf(p, sizeof p, "/mnt/ext%d", i);
            if (mounted(p))
                add(SRC_DIR, i ? "Extended storage 2" : "Extended storage", p, p);
        }
        add(SRC_DLNA_LIST, "Media servers (DLNA)", "", "");
        const Config c = load_config();
        add(SRC_RSS_LIST, "RSS feeds", "", c.feeds.empty() ? "" : std::to_string(c.feeds.size()));
        if (!c.rd_token.empty())
            add(SRC_RD, "Real-Debrid", "", "");
        for (const auto &s : c.sites)
            add(SRC_SITE, s.first, s.second, heresphere_url(s.second) ? "HereSphere" : "DeoVR");
        const std::string ip = vr_console_ip();
        add(SRC_MIRROR, "Mirror to TV", "", vr_prefs_mirror() ? "On" : "Off");
        add(SRC_HANDS, "Hands & eyes (experimental)", "", vr_control_name(vr_prefs_control()));
        add(SRC_INFO, "Settings", "", "http://" + ip + ":" +
                                       std::to_string(vr_settings_server_port()));
        add(SRC_INFO, "Support PS5VR", "", "buymeacoffee.com/theghostonline");
        return true;
    }
    case SRC_DIR:
        return list_dir(where.key, out, err);
    case SRC_DLNA_LIST: {
        for (const DlnaServer &s : dlna_discover()) {
            VrSrc it;
            it.kind = SRC_DLNA;
            it.name = s.name.empty() ? "Media server" : s.name;
            it.key = "0";
            it.base = s.control;
            out.push_back(it);
        }
        if (out.empty())
            err = "No media servers found on your network";
        return true;
    }
    case SRC_DLNA:
        return dlna_browse(where.base, where.key, out, err);
    case SRC_RSS_LIST: {
        for (const auto &f : load_config().feeds) {
            VrSrc it;
            it.kind = SRC_RSS;
            it.name = f.first;
            it.key = f.second;
            out.push_back(it);
        }
        if (out.empty())
            err = "No feeds yet - add them in Settings";
        return true;
    }
    case SRC_RSS:
        return rss_list(where.key, out, err);
    case SRC_RD:
        return rd_list(load_config().rd_token, out, err);
    case SRC_SITE:
        return site_groups(where.key, out, err);
    case SRC_SITE_GROUP:
        return site_group(where.key, where.base, out, err);
    default:
        err = "Nothing to open here";
        return false;
    }
}

/* -- the settings page ---------------------------------------------------------------- */
extern "C" {
typedef union { char ip[16]; uint8_t pad[256]; } netctl_info;
int sceNetCtlInit(void);
int sceNetCtlGetInfo(int code, netctl_info *info);
}

std::string vr_console_ip(void)
{
    static std::string s_ip;
    if (s_ip.empty()) {
        sceNetCtlInit();
        netctl_info info;
        std::memset(&info, 0, sizeof info);
        if (sceNetCtlGetInfo(14 /* IP address */, &info) == 0 && info.ip[0])
            s_ip = info.ip;
    }
    return s_ip.empty() ? "<your PS5's IP>" : s_ip;
}

namespace {
const int kSettingsPort = 8090;

std::string html_escape(const std::string &s)
{
    std::string o;
    for (char c : s) {
        if (c == '&') o += "&amp;"; else if (c == '<') o += "&lt;"; else if (c == '>') o += "&gt;";
        else if (c == '"') o += "&quot;"; else o += c;
    }
    return o;
}

std::string url_decode(const std::string &s)
{
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '+') o += ' ';
        else if (s[i] == '%' && i + 2 < s.size()) { o += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16); i += 2; }
        else o += s[i];
    }
    return o;
}

std::string form_field(const std::string &body, const std::string &key)
{
    size_t p = 0;
    while (p < body.size()) {
        size_t amp = body.find('&', p);
        if (amp == std::string::npos) amp = body.size();
        const std::string kv = body.substr(p, amp - p);
        const size_t eq = kv.find('=');
        if (eq != std::string::npos && kv.substr(0, eq) == key)
            return url_decode(kv.substr(eq + 1));
        p = amp + 1;
    }
    return "";
}

/* "Name | URL" lines <-> (name, url) */
std::string lines_of(const std::vector<std::pair<std::string, std::string>> &v)
{
    std::string o;
    for (const auto &e : v)
        o += (e.first == e.second ? "" : e.first + " | ") + e.second + "\n";
    return o;
}

cJSON *json_list(const std::string &text)
{
    cJSON *a = cJSON_CreateArray();
    size_t p = 0;
    while (p < text.size()) {
        size_t nl = text.find('\n', p);
        if (nl == std::string::npos) nl = text.size();
        std::string line = text.substr(p, nl - p);
        p = nl + 1;
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        std::string name, url = line;
        const size_t bar = line.find('|');
        if (bar != std::string::npos) {
            name = line.substr(0, bar);
            url = line.substr(bar + 1);
        }
        auto trim = [](std::string &s) {
            while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
            s.erase(0, s.find_first_not_of(" \t"));
        };
        trim(name);
        trim(url);
        if (url.empty())
            continue;
        cJSON *o = cJSON_CreateObject();
        cJSON_AddStringToObject(o, "name", name.empty() ? url.c_str() : name.c_str());
        cJSON_AddStringToObject(o, "url", url.c_str());
        cJSON_AddItemToArray(a, o);
    }
    return a;
}

/* strcasestr: the console's libc does not export it */
const char *strcasestr_compat(const char *h, const char *n)
{
    const size_t k = std::strlen(n);
    for (; *h; h++)
        if (!strncasecmp(h, n, k))
            return h;
    return nullptr;
}

/* A play request from the network (POST /play), waiting for the home screen. */
pthread_mutex_t s_play_lock = PTHREAD_MUTEX_INITIALIZER;
std::string s_play_pending;

void serve(int fd)
{
    std::string req;
    char buf[8192];
    timeval tv = {5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    size_t head_end = std::string::npos, want = 0;
    for (;;) {
        const ssize_t n = recv(fd, buf, sizeof buf, 0);
        if (n <= 0)
            break;
        req.append(buf, (size_t)n);
        if (head_end == std::string::npos && (head_end = req.find("\r\n\r\n")) != std::string::npos) {
            const char *cl = strcasestr_compat(req.c_str(), "Content-Length:");
            want = head_end + 4 + (cl ? (size_t)atoi(cl + 15) : 0);
        }
        if (head_end != std::string::npos && req.size() >= want)
            break;
        if (req.size() > (1u << 20))
            break;
    }
    std::string reply;
    if (!req.compare(0, 10, "POST /play")) {
        /* {"url": a path or URL, "title", "vr"} - same shape as the library's */
        const std::string body = head_end != std::string::npos ? req.substr(head_end + 4) : "";
        cJSON *j = cJSON_Parse(body.c_str());
        const bool ok = j && cJSON_GetStringValue(cJSON_GetObjectItem(j, "url"));
        cJSON_Delete(j);
        if (ok) {
            pthread_mutex_lock(&s_play_lock);
            s_play_pending = body;
            pthread_mutex_unlock(&s_play_lock);
        }
        reply = ok ? "HTTP/1.0 202 Accepted\r\nContent-Length: 0\r\n\r\n"
                   : "HTTP/1.0 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
    } else if (!req.compare(0, 10, "POST /link")) {
        /* the settings page's "Play a link": form field url (and an optional vr) */
        const std::string body = head_end != std::string::npos ? req.substr(head_end + 4) : "";
        std::string url = form_field(body, "url");
        url.erase(0, url.find_first_not_of(" \t\r\n"));
        while (!url.empty() && isspace((unsigned char)url.back()))
            url.pop_back();
        const bool ok = !url.compare(0, 7, "http://") || !url.compare(0, 8, "https://");
        if (ok) {
            std::string name = url.substr(0, url.find('?'));
            name = name.substr(name.rfind('/') + 1);
            cJSON *j = cJSON_CreateObject();
            cJSON_AddStringToObject(j, "url", url.c_str());
            cJSON_AddStringToObject(j, "title", name.empty() ? "Video" : name.c_str());
            const std::string vr = form_field(body, "vr");
            if (!vr.empty())
                cJSON_AddStringToObject(j, "vr", vr.c_str());
            char *txt = cJSON_PrintUnformatted(j);
            pthread_mutex_lock(&s_play_lock);
            s_play_pending = txt ? txt : "";
            pthread_mutex_unlock(&s_play_lock);
            free(txt);
            cJSON_Delete(j);
            eng_bt("ps5vr: link sent from the settings page");
        }
        reply = std::string("HTTP/1.0 303 See Other\r\nLocation: /?") + (ok ? "sent=1" : "badlink=1") +
                "\r\nContent-Length: 0\r\n\r\n";
    } else if (!req.compare(0, 10, "POST /save")) {
        const std::string body = head_end != std::string::npos ? req.substr(head_end + 4) : "";
        cJSON *j = cJSON_CreateObject();
        cJSON_AddItemToObject(j, "sites", json_list(form_field(body, "sites")));
        cJSON_AddItemToObject(j, "feeds", json_list(form_field(body, "feeds")));
        cJSON_AddStringToObject(j, "realdebrid", form_field(body, "rd").c_str());
        char *txt = cJSON_Print(j);
        mkdir(PS5VR_DIR, 0777);
        if (FILE *f = std::fopen(PS5VR_CONFIG, "wb")) {
            std::fputs(txt, f);
            std::fclose(f);
        }
        eng_bt("ps5vr: settings saved from the web page");
        free(txt);
        cJSON_Delete(j);
        reply = "HTTP/1.0 303 See Other\r\nLocation: /?saved=1\r\nContent-Length: 0\r\n\r\n";
    } else {
        const Config c = load_config();
        const bool saved = req.find("saved=1") != std::string::npos && req.find("saved=1") < req.find("\r\n");
        std::string page =
            "<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width'>"
            "<title>PS5VR settings</title><style>body{font-family:system-ui;background:#0a0e1a;color:#e8eef8;"
            "max-width:760px;margin:2em auto;padding:0 1em}h1{display:flex;gap:.5em;align-items:center}"
            ".vr{background:#0070d1;color:#fff;font-weight:800;padding:.2em .35em;border-radius:.2em}"
            "label{display:block;margin:1.4em 0 .4em;font-weight:600}textarea,input{width:100%;box-sizing:border-box;"
            "background:#141a2c;color:#fff;border:1px solid #2a3554;border-radius:.4em;padding:.6em;font:inherit}"
            "small{color:#93a3c0}button{margin-top:1.5em;background:#0070d1;color:#fff;border:0;border-radius:.4em;"
            "padding:.8em 1.6em;font:inherit;font-weight:700}.ok{background:#12391f;padding:.6em;border-radius:.4em}"
            "</style></head><body><h1><span class=vr>VR</span>PS5VR settings</h1>";
        auto flag = [&](const char *k) {
            const size_t a = req.find(k), e = req.find("\r\n");
            return a != std::string::npos && a < e;
        };
        if (saved)
            page += "<p class=ok>Saved. In PS5VR, press &#9651; in the library to refresh.</p>";
        if (flag("sent=1"))
            page += "<p class=ok>Sent. It starts in the headset in a moment (PS5VR has to be on its library).</p>";
        if (flag("badlink=1"))
            page += "<p class=ok style='background:#3a1d1d'>That doesn't look like a video link "
                    "(it has to start with http:// or https://).</p>";
        page += "<form method=post action=/link><label>Play a link</label>"
                "<input name=url type=url inputmode=url autocomplete=off placeholder='https://example.com/video.mp4'>"
                "<select name=vr style='margin-top:.6em;width:100%;background:#141a2c;color:#fff;border:1px solid #2a3554;"
                "border-radius:.4em;padding:.6em;font:inherit'>"
                "<option value=''>Video type: detect from the name</option>"
                "<option value='180_sbs'>180&deg; 3D side by side</option><option value='360_tb'>360&deg; 3D top-bottom</option>"
                "<option value='360'>360&deg; 2D</option><option value='180'>180&deg; 2D</option>"
                "<option value='flat_sbs'>Flat 3D side by side</option><option value='flat'>Flat 2D</option></select>"
                "<button>Play</button><small style='display:block;margin-top:.6em'>Direct links to a video file "
                "(.mp4, .mkv...). Plays in the headset straight away.</small></form>";
        page += "<form method=post action=/save>"
                "<label>DeoVR / HereSphere libraries</label><textarea name=sites rows=4 placeholder='My Stash | http://192.168.0.10:9999/deovr'>" +
                html_escape(lines_of(c.sites)) +
                "</textarea><small>One per line: <code>Name | URL</code>. Stash, XBVR and many VR sites offer a DeoVR "
                "(<code>/deovr</code>) or HereSphere (<code>/heresphere</code>) link.</small>"
                "<label>RSS / Atom feeds</label><textarea name=feeds rows=4 placeholder='My feed | https://example.com/videos.rss'>" +
                html_escape(lines_of(c.feeds)) +
                "</textarea><small>One per line: <code>Name | URL</code>. Items with a video enclosure are listed.</small>"
                "<label>Real-Debrid API token</label><input name=rd value='" + html_escape(c.rd_token) +
                "' placeholder='from real-debrid.com/apitoken'><small>Lists your Real-Debrid downloads.</small>"
                "<button>Save</button></form>"
                "<p style='margin-top:3em;color:#6b7a96;font-size:.9em'>PS5VR is free. If you enjoy it, you can "
                "<a style='color:#93a3c0' href='https://buymeacoffee.com/theghostonline'>support it here</a>.</p>"
                "</body></html>";
        reply = "HTTP/1.0 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: " +
                std::to_string(page.size()) + "\r\nConnection: close\r\n\r\n" + page;
    }
    send(fd, reply.data(), reply.size(), 0);
    close(fd);
}

void *server_thread(void *)
{
    const int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a;
    std::memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons(kSettingsPort);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (s < 0 || bind(s, (sockaddr *)&a, sizeof a) != 0 || listen(s, 4) != 0) {
        eng_bt("ps5vr: settings page could not listen on port %d", kSettingsPort);
        return nullptr;
    }
    eng_bt("ps5vr: settings page on port %d", kSettingsPort);
    for (;;) {
        const int fd = accept(s, nullptr, nullptr);
        if (fd >= 0)
            serve(fd);
    }
    return nullptr;
}
} // namespace

void vr_settings_server_start(void)
{
    static bool started;
    if (started)
        return;
    started = true;
    pthread_t t;
    if (pthread_create(&t, nullptr, server_thread, nullptr) == 0)
        pthread_detach(t);
}

int vr_settings_server_port(void)
{
    return kSettingsPort;
}

bool vr_settings_play_pending(void)
{
    pthread_mutex_lock(&s_play_lock);
    const bool p = !s_play_pending.empty();
    pthread_mutex_unlock(&s_play_lock);
    return p;
}

bool vr_settings_take_play(std::string &json)
{
    pthread_mutex_lock(&s_play_lock);
    json.swap(s_play_pending);
    s_play_pending.clear();
    pthread_mutex_unlock(&s_play_lock);
    return !json.empty();
}
