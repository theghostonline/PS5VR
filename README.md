# PS5VR

[![Buy Me a Coffee](https://img.shields.io/badge/Buy_Me_a_Coffee-Support_the_project-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/theghostonline)

A VR video player for PlayStation VR2, running natively on a jailbroken PS5.

180° and 360° video, 3D side-by-side and top-bottom, spatial video and 3D
Blu-ray, flat 2D and 3D films on a big virtual screen - decoded on the PS5's
own video hardware up to 8K and shown in the headset through the system's
PS VR2 compositor, at 120 Hz.

> Homebrew. Not affiliated with or endorsed by Sony. It plays your own files
> and the sources you point it at; it includes no content.

---

## Features

**Picture**
- Hardware HEVC (8 and 10-bit) and H.264 decoding, up to 8K (7680x3840)
- 120 Hz headset output with an even frame cadence (60 fps pictures show for
  two refreshes, 30 fps for four)
- Smooth motion: 8K60 decodes at 30 fps on the PS5, so frame generation
  (motion estimation on the GPU) makes the picture in between - the headset
  gets ~55-60 pictures a second instead of 30. On by default, in the View menu
- 10-bit eye buffers, 4096 px per eye, FSR 1 upscaling and sharpening for
  pictures thinner than the panel
- HDR (experimental, off by default): HDR10 and HLG keep their highlights
  above SDR white in float eye buffers instead of being tone-mapped to SDR

**Formats**
- 360° and 180° (equirectangular), mono, side-by-side or top-bottom
- Flat 2D and flat 3D (full and half SBS, OU) on a screen of your size
- Spatial video (MV-HEVC) and 3D Blu-ray (H.264 MVC), both eyes
- Format detection from the file name and the picture's shape, overridable
  in the View menu

**Sound**
- Head-tracked spatial audio: stereo and 5.1/7.1 placed around you as
  virtual speakers that stay put when you turn your head
- First-order ambisonics (AmbiX) rotated with your head
- Automatic audio delay correction for slow decodes

**Sources**
- Files on the PS5's internal storage, USB drives and extended storage
- DLNA / UPnP media servers on your network (Plex, Jellyfin, Emby, ...)
- RSS / Atom feeds with video enclosures
- DeoVR and HereSphere libraries (Stash, XBVR and many VR sites)
- Real-Debrid downloads, with parallel read-ahead (6 connections) for big
  files over the internet
- A settings page on your phone or computer for feeds, libraries and tokens

**In the headset**
- A library and a View menu drawn in VR, driven by the DualSense
- Recentre (R3), auto-pause when the headset comes off, TV mirroring
- PS VR2 Sense controllers: tracked controllers in the headset with a laser
  pointer for the panels (trigger to select, drag to scroll, circle for back)
- Play a link from your phone: scan the QR code on the TV home screen, paste
  any direct video link (.mp4, .mkv...) and press Play - it starts in the
  headset straight away, no typing on the console
- Hands & eyes (experimental), four modes:
  - Hands: point with your hand, pinch to select, with your hands drawn in VR
  - Eyes + pinch: look at an item and pinch with either hand
  - Eyes + blink: look at an item, close your eyes until the headset buzzes,
    then open them to select - no hands needed
  - A quick pinch shows the playback controls, a held pinch the View menu

---

## Install

You need a jailbroken PS5 with an ELF loader (tested on 13.60 with Relapse,
kstuff, ShadowMount+ and etaHEN) and a PS VR2.

1. Copy the `PPSA99177` folder from the release to `/data/homebrew/` on the
   console. ShadowMount+ installs it as a "PS5VR" tile.
2. Send `ps5vr-helper.elf` to your ELF loader (port 9021), or add it to your
   autoload list. It gives PS5VR access to `/data` and USB drives and, if you
   turn on Hands & eyes, to the headset cameras. It only acts when PS5VR asks.
3. Put videos in `/data/ps5vr/videos` (FTP), on a USB drive, or on a media
   server, and start PS5VR from the home screen with the headset on.

Settings for feeds, DeoVR/HereSphere libraries and Real-Debrid are at
`http://<your PS5's IP>:8090` (shown in the library).

---

## Controls

| | |
|---|---|
| Cross | open / select |
| Circle | back |
| Options | View menu (while playing) |
| R3 | recentre |
| L1 / R1 | page through long lists |
| Triangle | refresh the list |

The View menu: video type, 3D layout, swap eyes, screen size, sharpening,
smooth motion, HDR, spatial audio, mirror to TV, hands & eyes, recentre.

---

## Building

The app builds natively on a Mac (Apple silicon) against the
[PS5 payload SDK](https://github.com/ps5-payload-dev/sdk) sysroot:

    PS5_PAYLOAD_SDK=~/ps5sdk/opt/ps5-payload-sdk app/scripts/build-native-mac.sh

The output is `app/build-mac/app/PPSA99177/`. The helper is a plain payload:

    prospero-clang -O2 -o ps5vr-helper.elf helper/ps5vr-helper.c -lkernel_sys

The GPU shader pipelines (`app/engine/shaders/agc/`) are compiled with AMD's
LLPC for gfx1013; their generators are in `app/tools/shaders/`.

---

## How it works

- The PS VR2 system libraries (libSceHmd2, libSceVrTracker2, libSceVrHand)
  are loaded at runtime and called by their exports on firmware 13.60. The
  picture goes to the headset as a type 0x10 layer through Hmd2's
  reprojection, the way PS VR2 games submit it.
- Each decoded frame is converted once on the GPU and projected per eye
  from the tracked head pose; a floating panel draws the interface.
- `tools/ps5_dynlib.py` reads decrypted system libraries (exports, imports,
  NIDs) and `tools/gen_export_table.py` writes the export offsets the app uses.

Other firmware versions need their own export table, and it has to be
generated from that firmware's own libraries: a 13.60 console cannot produce a
12.40 table. If you are on another firmware and want it supported, see
[docs/firmware-support.md](docs/firmware-support.md) for the three libraries
needed and how to generate the table. Wrong-firmware tables fail safely: each
one carries an anchor string that is checked before any offset is used.

---

## Support

If you use PS5VR and want to help cover the time it took to build, there's a
[Buy Me a Coffee](https://buymeacoffee.com/theghostonline). Completely
optional.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
