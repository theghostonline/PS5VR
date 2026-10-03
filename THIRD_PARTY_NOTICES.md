# Third-party notices

PS5VR - Copyright (C) 2026 Husam Osman - is distributed under the GNU General
Public License v3.0 or later. It builds on, links or bundles the following
components, each under its own license.

## Source in this repository

| Component | Where | License |
| --- | --- | --- |
| PS5 media engine and native-app packaging toolkit, from an open-source GPL-3.0 PS5 media player (itself a fork of ProsperoPlayer); modified for this project in 2026 | `app/engine/`, `app/tools/native-app/`, `app/scripts/` | GPL-3.0-or-later (`app/engine/LICENSE.ENGINE`) |
| Player interface and session code from an open-source GPL-3.0 PS5 media app, modified | `app/src/` | GPL-3.0-or-later |
| edge264 (H.264 / MVC decoder) | `app/third_party/edge264/` | BSD-3-Clause |
| NanoSVG | `app/third_party/nanosvg/` | zlib |
| cJSON | `app/engine/addons/src/cJSON.c` | MIT |
| QR Code generator library, Copyright (c) Project Nayuki | `app/src/qrcodegen.c`, `qrcodegen.h` | MIT |
| FSR 1 (EASU, RCAS), GLSL port | `app/engine/shaders/agc/` | MIT (AMD) |
| Anime4K CNN weights | `app/engine/shaders/agc/` | MIT (bloc97) |
| Inter typeface | `app/assets/fonts/` | SIL Open Font License 1.1 |
| Roboto typeface | `app/assets/fonts/` | Apache-2.0 |
| Noto Naskh Arabic, Noto Emoji | `app/assets/fonts/` | SIL Open Font License 1.1 |
| Player icons; Material Design icons | `app/assets/icons/` | GPL-3.0; Apache-2.0 |

## Linked libraries (from the PS5 SDK sysroot)

| Component | License |
| --- | --- |
| FFmpeg 7.1 (libavformat, libavcodec, libavutil, libswresample, libswscale) | LGPL-2.1+ / GPL as configured |
| dav1d | BSD-2-Clause |
| libass | ISC |
| FreeType | FreeType License |
| HarfBuzz | MIT |
| FriBidi | LGPL-2.1+ |
| Fontconfig, Expat | MIT |
| libpng | libpng License |
| libjpeg-turbo | IJG / BSD-3-Clause |
| libwebp | BSD-3-Clause |
| libxml2 | MIT |
| OpenSSL | Apache-2.0 |
| zlib, bzip2, xz, zstd | zlib / bzip2 / 0BSD / BSD |

PlayStation, PS5 and PS VR2 are trademarks of Sony Interactive Entertainment.
This project is not affiliated with Sony.
