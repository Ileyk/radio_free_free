# radio_free_free

Define the three-dimensional flow in Python and compute its thermal free-free
radio emission with a Kokkos C++ ray tracer. `rff-transfer` accepts analytical
Python models; the separate flow-preparation tools can still create `.rff`
grids for visualization and preprocessing, but the transfer executable does
not read them. Physical density, temperature, length scale, frequency, and
distance are supplied separately. Density is normalized with a baryon number
density `n0` in cm^-3, and the reference length `R0` is provided as a number
of solar radii. Changing source distance therefore only changes the reported
flux, not the image intensity or transfer solution.

## Flow representation

`FlowGrid` is a uniform, cell-centered Cartesian grid. Coordinates are in units
of a chosen reference length `R0`; number density and temperature fields are
ratios to `n0` and `T0`. Along with the occultation mask, it stores four
physical scalar fields:

| Field | Meaning |
| --- | --- |
| `density` | baryon number density `n_b / n0` |
| `temperature` | `T / T0` |
| `electron_per_baryon` | `n_e / n_b` |
| `ion_charge2_per_baryon` | `(sum_i Z_i^2 n_i) / n_b` |

The occultation mask is a fifth, dimensionless field: zero means transparent,
one means opaque. It defaults to zero.

For fully ionized pure hydrogen, both abundance fields are one. For a different
composition, supply the corresponding abundance ratios; these let the transfer
solver support spatially varying ionization and composition without embedding
chemistry assumptions in C++.

Create the isothermal `n_b proportional to r^-2` wind benchmark:

```sh
python -m pip install -e .
rff-flow wind wind.rff --resolution 128 128 128 \
  --inner-radius 0.5 --outer-radius 10
```

The uniform ion composition defaults to fully ionized hydrogen
(`ni_Zi2_per_baryon = 1`, `eff_Z_i = 1`). Set the two
values explicitly for another uniform composition; for example, a
fully-ionized helium wind uses:

```sh
rff-flow wind helium-wind.rff --ni_Zi2_per_baryon 1 --eff_Z_i 2
```

`ni_Zi2_per_baryon` is the uniform `(sum_i Z_i^2 n_i) / n_b` abundance. The
electron abundance is derived as `ni_Zi2_per_baryon / eff_Z_i`; both
abundances are zero outside the spherical wind cutoff. The stored flow-field
name remains `ion_charge2_per_baryon`.

In this file, `R0` is the radial unit and `n0` is the number density at
`r = R0`. `outer-radius` sets a spherical wind cutoff; the Cartesian grid
extends to the enclosing cube, but density, temperature, and abundances are
zero outside the sphere. The inner sphere is vacuum (stellar interior),
avoiding a singularity at the origin and keeping it resolved by the example
grid. By default, within the spherical domain the wind has `T/T0 = 1` and
pure-hydrogen abundances.

## Prescribed flow functions

Use the `functions` command to sample custom fields from a Python file on the
requested Cartesian bounds and resolution:

```sh
rff-flow functions disk_flow.py runs/disk/flow.rff \
  --coordinates cylindrical --bounds -10 10 -10 10 -5 5 \
  --resolution 128 128 64
```

The file must define callable `density`, `temperature`,
`electron_per_baryon`, and `ion_charge2_per_baryon` functions. Each receives
three NumPy arrays at Cartesian cell centers, transformed to either
cylindrical `(R, phi, z)` or spherical `(r, theta, phi)` coordinates. Lengths
are in `R0` units, and angles are in radians; spherical `theta` is measured
from +z and `phi` is the azimuth in `[0, 2*pi)`. Functions may return a scalar
or an array broadcastable to the grid shape. `density` and `temperature` are
dimensionless (`n_b/n0` and `T/T0`); the abundance fields are per baryon.

For example, `disk_flow.py` can contain:

```python
import numpy as np

def density(R, phi, z):
    return np.exp(-R) * np.exp(-np.abs(z))

def temperature(R, phi, z):
    return 1.0

def electron_per_baryon(R, phi, z):
    return 1.0

def ion_charge2_per_baryon(R, phi, z):
    return 1.0
```

The Python input file is executed locally, so only use files you trust. Like
the other `rff-flow` modes, this command writes both the `.rff` grid and its
same-stem `.vtk` visualization file.

## Direct analytical transfer

`rff-transfer` executes the analytical input file locally and evaluates its
profiles directly, without creating a Cartesian flow grid or an occultation
mask. Only use Python files you trust:

```sh
rff-transfer --input runs/1/model.py --output runs/1/flow-spectrum.csv \
  --n0 1e9 --temperature0 3e5 --length0 70 \
  --frequency-min-hz 1e9 --frequency-max-hz 1e12 --frequency-count 100 \
  --view-theta-deg 53 --view-phi-deg 0 \
  --image-bounds -10 10 -10 10 --ray-bounds -10 10 \
  --image-resolution 96 96 --ray-samples 128
```

The file defines a `coordinates` string (`"cartesian"`, `"cylindrical"`, or
`"spherical"`) and vectorized
`density(x1, x2, x3)`, `temperature(x1, x2, x3)`,
`electron_per_baryon(x1, x2, x3)`, and
`ion_charge2_per_baryon(x1, x2, x3)` functions. The argument coordinates are
`(x, y, z)`, `(R, phi, z)`, or `(r, theta, phi)` respectively; angles are in
radians, with spherical `theta` measured from +z. Functions receive NumPy
arrays and may return arrays of matching shape or broadcastable scalars.
Density and temperature are normalized by `n0` and `T0`; ionic abundances are
dimensionless per-baryon ratios. `opaque_objects` is optional and may contain
`{"shape": "sphere", "center": (x, y, z), "radius": r}` entries or
`{"shape": "cylinder", "center": (x, y, z), "radius": r, "height": h,
"axis": (ax, ay, az)}` entries. Dimensions are in `R0`; cylinder height is
the full height and its axis is normalized by the solver. Each opaque object
must lie entirely inside the observer domain defined by `--image-bounds` and
`--ray-bounds`. Multiple and overlapping objects are supported; they are
non-emitting and occult the flow analytically.

Both bounds options are required for analytical input.
`--image-bounds UMIN UMAX VMIN VMAX` sets the image-plane extent, while
`--ray-bounds SMIN SMAX` sets the longitudinal integration interval. These
observer-oriented coordinates are in `R0` units and are independent of the
physical extent of the flow; physical cutoffs such as `Rin` and `Rout` belong
in the analytical profiles. `--image-resolution NU NV` sets the regularly
spaced pixels (default `128 128`), and `--ray-samples NS` sets uniform
midpoint samples per visible ray (default `128`). The solver evaluates the
four flow functions in batches on the host, caches physical ray-segment
quantities, and performs the frequency-dependent transfer with Kokkos.
Segment lengths are dimensionless in `R0` and are multiplied by `--length0`
exactly once for optical depths; projected pixel areas receive the square of
that scale exactly once. Adaptive longitudinal integration is not provided.

Building `rff-transfer` requires a Python development package, NumPy available
to the Python interpreter selected by CMake, and Kokkos. The Python dependency
is already included in the package requirements; if needed, install NumPy for
the interpreter shown by CMake's `Found Python3` message. The example
[`runs/1/model.py`](./runs/1/model.py) reproduces the spherical `r^-2` wind
benchmark with a central sphere and finite cylinder. Its
[`run.sh`](./runs/1/run.sh) runs the direct analytical solver. The CSV keeps
the existing `frequency_hz,projected_intensity_area` spectrum format, with
additional input, sampling, observer-oriented bounds, and viewing-direction
metadata, so `rff-plot` continues to read it.

To load simulation data, install the optional readers:

```sh
python -m pip install -e '.[vtk]'
rff-flow vtk simulation.vtu resampled.rff \
  --bounds -10 10 -10 10 -10 10 --resolution 128 128 128 \
  --n0 5.98e5 --density-kind mass --temperature0 1e4 \
  --fields '{"density":"rho","temperature":"temp","electron_per_baryon":"ye","ion_charge2_per_baryon":"ion_q2"}'
```

The loader uses `meshio` for VTK/VTU input and `scipy.interpolate.griddata`
for linear resampling. `--bounds` are ordered as `xmin xmax ymin ymax zmin
zmax`; the four scalar fields must be present in point or cell data. Cell
values are interpolated from cell centers. By default, the VTK density field is interpreted
as mass density in g cm^-3 and converted to number density by dividing by the
proton mass and `--n0`. Use `--density-kind number` if the field is already a
number density in cm^-3. `--temperature0` divides the input temperature field;
the defaults of one are appropriate when those fields are already normalized.
Samples outside the source point cloud's convex hull use zero for density and
abundances, and the minimum source value for temperature. The transfer solver
computes the frequency-dependent Gaunt factor from this temperature field.
Grid bounds retain
the VTK coordinate values and are dimensionless in units of `R0`;
`rff-transfer --length0` specifies the physical size of one grid coordinate
unit in solar radii.

`FlowGrid.write()` writes a compact little-endian binary file with a versioned
header, dimensions, bounds, and the four physical fields plus the occultation
mask. The layout is x-major, then y, then z, with fields contiguous per cell.
`FlowGrid.read()` can be used to inspect a generated file. Existing files
created with the former mass-density normalization contain the same density
ratios when `n0 = rho0 / m_p`; the binary file itself does not store either
physical normalization. Version-1 files with a stored Gaunt factor and
version-2 files without a mask remain readable; their absent mask is treated
as fully transparent.

The native `.rff` file is not a VTK dataset, so VisIt and ParaView cannot open
it directly. Both `rff-flow` and `rff-occult` automatically write a same-stem
legacy ASCII `.vtk` file alongside their `.rff` output. For example,
`wind.rff` is accompanied by `wind.vtk`, and `occulted-wind.rff` by
`occulted-wind.vtk`. For files created another way, export manually:

```python
from radio_free_free import FlowGrid

flow = FlowGrid.read("wind.rff")
flow.write_vtk("wind.vtk")
```

`write_vtk()` writes a uniform structured-points dataset with cell-centered
scalar fields, including `occultation_mask`. Coordinates remain in the
dimensionless `R0` units used by the flow.

## Add an opaque volume

After creating a flow file, use `rff-occult` to add a solid sphere or finite
cylinder. The center and dimensions use the dimensionless flow coordinate
units (`R0`):

```sh
rff-occult wind.rff occulted-wind.rff --shape sphere \
  --center 0 0 0 --radius 1.5
rff-occult wind.rff occulted-wind.rff --shape cylinder \
  --center 1 0 0 --radius 0.5 --height 3 --axis z
```

The cylinder is centered on the given coordinates; `--height` is its full
length, and `--axis` selects x, y, or z (default z). The mask is sampled at
flow cell centers. Run `rff-transfer` with the resulting output file as its
input to activate occultation. Rays are traversed from the observer side, and
integration stops at the first opaque sample encountered. Emission accumulated
in front of that surface remains visible; emission at or behind it is excluded.

## Plot radio outputs

Use `rff-plot` to inspect spectrum CSV output from `rff-transfer`; the format
is detected from its header. The plotter also remains able to read legacy
image CSVs created by older transfer versions:

```sh
rff-plot wind-spectrum.csv --distance-pc 100
rff-plot legacy-wind-image.csv --distance-pc 100
```

For separate runs, keep a `run.sh` and outputs in a subfolder and pass their
paths explicitly. For example, `rff-transfer --output runs/wind-a/spectrum.csv`
creates missing output directories, and `rff-plot runs/wind-a/spectrum.csv`
opens that result interactively. `rff-flow` and `rff-occult` also create the
parent directories for their `.rff` and `.vtk` outputs.

Spectrum plots show integrated flux density in mJy versus frequency in GHz on
logarithmic axes (the flux axis is linear if the data contain zeros). They
include a lower-right scale legend with `n0`, `T0`, `R0`, grid resolution, and
box dimensions. An unchecked `Linford et al. (2019)` checkbox overlays the
comparison points from `linford_19.txt` when selected. The logarithmic
distance slider sets the inverse-square flux scaling; `--distance-pc` sets its
initial value. Legacy image plots show specific intensity over the projected
`u`/`v` plane, with the distance-scaled integrated flux annotated in mJy. Use
`--mode image` only for legacy image CSVs, or `--mode spectrum` to force
spectrum parsing.
Outputs from older `rff-transfer` versions lack metadata needed to rescale
with distance; regenerate them to use the slider.

## Transfer model

The C++ code integrates rays through the flow with zero incident background
intensity. The viewing direction is set by `--view-theta-deg` and
`--view-phi-deg`: theta is measured from +z and phi is the azimuth measured
from +x in the xy plane. The flow is interpolated onto the viewing rays. In cgs
units the transfer uses

```text
n_b       = n0 * density
n_e       = n_b * electron_per_baryon
n_i,Z2    = n_b * ion_charge2_per_baryon
Z_i,eff   = ion_charge2_per_baryon / electron_per_baryon
g_ff      = ln(exp(5.960 - sqrt(3)/pi ln(Z_i,eff nu_9 T_4^(-3/2))) + e)
alpha_nu  = 3.692e8 T^(-1/2) nu^(-3)
            n_e n_i,Z2 (1 - exp(-h nu / k T)) g_ff   [cm^-1]
j_nu      = alpha_nu B_nu(T)
dI_nu/ds  = -alpha_nu I_nu + j_nu
```

The transfer is integrated from the observer-facing boundary toward the far
side. For each segment, its contribution is added as
`exp(-tau_foreground) B_nu(T) (1 - exp(-delta_tau))`, then its optical depth
is added to `tau_foreground`. Both exponentials use stable forms near zero
optical depth. This includes stimulated emission and does not assume the
Rayleigh-Jeans limit. An opaque surface stops the traversal while preserving
emission already accumulated in the foreground.

For each ray, the solver caches the maximum temperature in the still-unvisited
samples. Since `B_nu(T)` increases with temperature, this gives an upper bound
on their source function. The solver stops when the foreground-transmitted
upper bound on omitted intensity is less than the configured fraction of the
intensity already accumulated in the foreground. The default
`--relative-intensity-threshold` is `1e-3`; the underscore form
`--relative_intensity_threshold` is also accepted. The value must be greater
than zero and less than one.

Analytical transfer currently uses uniform midpoint sampling controlled by
`--ray-samples`. Adaptive longitudinal refinement and compression are not
available. Hydrodynamical quantities are cached during ray construction, and
frequency-dependent transfer runs on the selected Kokkos backend.

Here `T` is the local physical temperature in K and `nu` is the observing
frequency in Hz; `T_4 = T / 10^4 K` and `nu_9 = nu / 10^9 Hz`. For a flow
with one ion species, `Z_i,eff` equals its ionic charge number. For mixtures,
the flow's aggregate composition fields define the effective value as
`(sum_i Z_i^2 n_i) / (sum_i Z_i n_i)`. `rff-transfer` computes `g_ff` from
the evaluated temperature and effective charge at every transfer sample
for every frequency in the spectrum.

The flow preparation tools operate on dimensionless coordinates in units of
`R0`. Set the physical size of one `R0` coordinate unit with
`rff-transfer --length0`, in solar radii. The executable converts this value
to centimeters internally. `rff-transfer` accepts analytical Python models and calculates the projected
intensity-area integral at logarithmically spaced frequencies, including both
endpoints:

```sh
./build/rff-transfer --mode spectrum --input runs/1/model.py \
  --output wind-spectrum.csv \
  --n0 5.98e5 --temperature0 1e4 --length0 215.61 \
  --frequency-min-hz 1e8 --frequency-max-hz 1e12 --frequency-count 100 \
  --view-theta-deg 30 --view-phi-deg 45 \
  --image-bounds -10 10 -10 10 --ray-bounds -10 10 \
  --image-resolution 128 128 --ray-samples 128 \
  --relative-intensity-threshold 1e-3
```

The spectrum CSV contains frequency in Hz and the projected intensity-area
integral in `erg s^-1 Hz^-1`. The viewing-angle options apply to the spectrum.
`rff-plot` converts this integral to observed flux density using
`F_nu = integral / distance_cm^2`; flux density is displayed in mJy. The `.rff`
file format remains available for flow preparation and visualization but is
not accepted by `rff-transfer`. Output CSV metadata stores `n0`, `T0`, and
`R0` for spectrum scale annotation, as well as observer-oriented image and
ray bounds and sampling resolution. Frequency values remain in Hz in the file
and are displayed in GHz. The CSV records `length0_cm` after converting the
input solar radii to centimeters.

Build against a Kokkos installation configured with CMake. `rff-transfer`
embeds the selected Python interpreter, so it must have development headers
and NumPy installed; set `Python3_EXECUTABLE` when CMake's default Python does
not meet those requirements:

```sh
cmake -S cpp -B build \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos \
  -DPython3_EXECUTABLE=/path/to/python3
cmake --build build --parallel
```

With Homebrew Kokkos/OpenMP and AppleClang on macOS, provide the `libomp`
settings explicitly if CMake cannot detect OpenMP:

```sh
cmake -S cpp -B build \
  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib" \
  -DOpenMP_CXX_INCLUDE_DIR="$(brew --prefix libomp)/include"
cmake --build build --parallel
```

Kokkos chooses the enabled execution backend at build time (for example
OpenMP/Serial or CUDA/HIP). Spectrum calculations run on the selected backend.

The required normalizations are `n0` in cm^-3, `temperature0` in K,
`length0` in solar radii, and spectrum frequency bounds in Hz. `n0`,
`temperature0`, and `length0` define the physical scaling of the dimensionless
flow. Source distance is not used by `rff-transfer`; set it in `rff-plot`.

## Tests

```sh
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

The tests cover cell-centered function sampling, binary round-tripping, wind
normalization, and input validation. When `build/rff-transfer` is available,
they also compare adaptive and fixed transfer and check opaque-surface handling.
