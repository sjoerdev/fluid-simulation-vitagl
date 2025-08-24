## Description:

This is an SPH fluid simulation for the vita.

## Showcase:
https://github.com/user-attachments/assets/a05789b8-3cdc-4c62-8b07-8e2bfecd9748

## Building:

Building on Linux:

1. Install the [VitaSDK](https://vitasdk.org/)
2. Build and install [VitaGL](https://github.com/Rinnegatamante/vitaGL) with these make flags:
    - ``HAVE_GLSL_SUPPORT=1``
    - ``CIRCULAR_VERTEX_POOL=2``
    - ``USE_SCRATCH_MEMORY=1``
    - ``NO_DEBUG=1``
3. Build with ``cmake . && make``

Building on Windows:

1. Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with ``wsl --install``
2. Follow the Linux directions above

## Resources
- https://github.com/sjoerdev/fluid-simulation
- https://matthias-research.github.io/pages/publications/sca03.pdf
- https://www.cs.cornell.edu/~bindel/class/cs5220-f11/code/sph.pdf
- https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf
- https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf
- https://lucasschuermann.com/writing/particle-based-fluid-simulation
- https://lucasschuermann.com/writing/implementing-sph-in-2d
