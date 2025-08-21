## Description:

This is an SPH fluid simulation for the vita.

## Building:

Building on Linux:

1. Install [VitaSDK](https://vitasdk.org/)
2. Build and install [VitaGL](https://github.com/Rinnegatamante/vitaGL) with the ``HAVE_GLSL_SUPPORT=1`` make flag
2. Run ``cmake . && make``

Building on Windows:

1. Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with ``wsl --install``
2. Follow the Linux directions above