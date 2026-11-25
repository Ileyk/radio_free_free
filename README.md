# radio_free_free

Prepare a three-dimensional flow in Python, then postprocess its thermal
free-free radio emission with a Kokkos C++ ray tracer. The flow grid is
dimensionless: physical density, temperature, length scale, frequency, and
distance are supplied to the C++ program separately. Density is normalized
with a baryon number density `n0` in cm^-3, and the reference length `R0` is
provided as a number of solar radii. Changing source distance therefore only
changes the reported flux, not the image intensity or transfer solution.

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

In this file, `R0` is the radial unit and `n0` is the number density at
`r = R0`. `outer-radius` sets a spherical wind cutoff; the Cartesian grid
extends to the enclosing cube, but density, temperature, and abundances are
zero outside the sphere. The inner sphere is vacuum (stellar interior),
avoiding a singularity at the origin and keeping it resolved by the example
grid. Within the spherical domain the wind has `T/T0 = 1` and pure-hydrogen
abundances.

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
input to activate occultation. A ray crossing an opaque cell loses all
background emission behind it, while foreground flow remains visible.
Occulting volumes can also be inspected with `rff-view`: their slice boundary
is drawn in cyan over the selected flow field, or select `--field
occultation_mask` to view the mask itself.

## Interactive flow viewer

Install the optional Matplotlib dependency and open a resampled flow file:

```sh
python -m pip install -e '.[visualization]'
rff-view occulted-wind.rff --x 0 --y 0 --z 0 --field density
```

The viewer opens three separate windows for YZ, XZ, and XY slices. Each window
shows the slice nearest the requested `--x`, `--y`, or `--z` coordinate. Omit
any coordinate to use the middle grid cell on that axis; coordinates outside
the grid bounds are rejected. Use `--field` to choose `density`,
`temperature`, `electron_per_baryon`, `ion_charge2_per_baryon`, or
`occultation_mask`; all three windows show the chosen field with a consistent
color scale. Occulting-surface intersections are overlaid in cyan.
Coordinates are displayed in the flow grid's `R0` units. A graphical
Matplotlib backend is required.

## Plot radio outputs

Use `rff-plot` to inspect either CSV output from `rff-transfer`; the format is
detected from its header:

```sh
rff-plot wind-image.csv --distance-pc 100
rff-plot wind-spectrum.csv --distance-pc 100
```

Image plots show specific intensity over the projected `u`/`v` plane, with
frequency and viewing direction in the title when available; the distance-
scaled integrated flux is annotated in mJy. Spectrum plots show integrated
flux density in mJy versus frequency in GHz on logarithmic axes (the flux
axis is linear if the data contain zeros). Spectrum plots also include a
lower-right scale legend with `n0`, `T0`, and `R0` in solar radii. An
unchecked `Linford et al. (2019)` checkbox overlays the comparison points from
`linford_19.txt` on spectrum plots when selected. Both plots
include a logarithmic distance slider; `--distance-pc` sets its initial value.
Distance rescales flux as inverse-square while leaving image intensity
unchanged. Use `--mode image` or `--mode spectrum` to override automatic format
detection. Outputs from older `rff-transfer` versions lack the metadata needed
to rescale with distance; regenerate them to use the slider.

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
g_ff      = 9.77 (1 + 0.13 ln(T^(3/2) / (1.2 nu) + exp(-6.9)))
alpha_nu  = 3.692e8 T^(-1/2) nu^(-3)
            n_e n_i,Z2 (1 - exp(-h nu / k T)) g_ff   [cm^-1]
j_nu      = alpha_nu B_nu(T)
dI_nu/ds  = -alpha_nu I_nu + j_nu
```

The transfer is integrated along each ray using the exact constant-step update
`I_out = I_in exp(-tau) + B_nu(T) (1 - exp(-tau))`. Both exponentials use
stable forms near zero optical depth. This includes stimulated emission and
does not assume the Rayleigh-Jeans limit.
Here `T` is the local physical temperature in K and `nu` is the observing
frequency in Hz. `rff-transfer` computes `g_ff` from the interpolated
temperature at every transfer sample. In image mode it uses the requested
frequency; in spectrum mode it recomputes the factor for each frequency.

The flow preparation and viewing tools operate on dimensionless coordinates
in units of `R0`. Set the physical size of one `R0` coordinate unit with
`rff-transfer --length0`, in solar radii. The executable converts this value
to centimeters internally. Image mode calculates a specific intensity map
at one frequency:

```sh
./build/rff-transfer --mode image --input wind.rff --output wind-image.csv \
  --n0 5.98e5 --temperature0 1e4 --length0 215.61 \
  --frequency-hz 1e10 --view-theta-deg 30 --view-phi-deg 45
```

The image CSV contains intensities in `erg s^-1 cm^-2 Hz^-1 sr^-1`, with
projected coordinates in `R0`, plus projected pixel area in cm^2 for flux
postprocessing. Spectrum mode calculates the projected intensity-area
integral at logarithmically spaced frequencies, including both endpoints:

```sh
./build/rff-transfer --mode spectrum --input wind.rff --output wind-spectrum.csv \
  --n0 5.98e5 --temperature0 1e4 --length0 215.61 \
  --frequency-min-hz 1e8 --frequency-max-hz 1e12 --frequency-count 100 \
  --view-theta-deg 30 --view-phi-deg 45
```

The spectrum CSV contains frequency in Hz and the projected intensity-area
integral in `erg s^-1 Hz^-1`. The viewing-angle options apply to both modes.
`rff-plot` converts this integral to observed flux density using
`F_nu = integral / distance_cm^2`; flux density is displayed in mJy. The `.rff`
flow no longer contains a Gaunt-factor field. Output CSV metadata stores
`n0`, `T0`, and `R0` for spectrum scale
annotation; frequency values remain in Hz in the file and are displayed in
GHz. The CSV records `length0_cm` after converting the input solar radii to
centimeters.

Build against a Kokkos installation configured with CMake:

```sh
cmake -S cpp -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos
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
OpenMP/Serial or CUDA/HIP). Both output modes run on the selected backend.

The required normalizations are `n0` in cm^-3, `temperature0` in K,
`length0` in solar radii, and observing frequency or spectrum bounds in Hz. `n0`,
`temperature0`, and `length0` define the physical scaling of the dimensionless
flow. Source distance is not used by `rff-transfer`; set it in `rff-plot`.

## Tests

```sh
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

The tests cover cell-centered function sampling, binary round-tripping, wind
normalization, and input validation.
