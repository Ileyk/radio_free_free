"""Post-process and plot radio image or spectrum CSV output."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

_CM_PER_PC = 3.0856775814913673e18
_CGS_PER_JANSKY = 1.0e-23
_MILLIJANSKY_PER_JANSKY = 1.0e3
_CM_PER_SOLAR_RADIUS = 6.957e10
_LINFORD_DATA_PATH = Path(__file__).resolve().parents[2] / "linford_19.txt"


@dataclass(frozen=True)
class PhysicalScales:
    number_density_cm3: float
    temperature_k: float
    length_cm: float

    @property
    def length_solar_radii(self) -> float:
        return self.length_cm / _CM_PER_SOLAR_RADIUS


@dataclass(frozen=True)
class FlowGridParameters:
    resolution: Tuple[int, int, int]
    bounds: Tuple[float, float, float, float, float, float]

    @property
    def box_dimensions(self) -> Tuple[float, float, float]:
        return (
            self.bounds[1] - self.bounds[0],
            self.bounds[3] - self.bounds[2],
            self.bounds[5] - self.bounds[4],
        )


@dataclass(frozen=True)
class RadioImage:
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    intensity: NDArray[np.float64]
    frequency_hz: Optional[float]
    line_of_sight: Optional[Tuple[float, float, float]]
    units: str
    projected_pixel_area_cm2: Optional[float] = None
    axis_names: Tuple[str, str] = ("u", "v")
    scales: Optional[PhysicalScales] = None
    flow_grid: Optional[FlowGridParameters] = None


@dataclass(frozen=True)
class RadioSpectrum:
    frequency_hz: NDArray[np.float64]
    flux_density_jy: Optional[NDArray[np.float64]]
    units: str
    projected_intensity_area: Optional[NDArray[np.float64]] = None
    scales: Optional[PhysicalScales] = None
    flow_grid: Optional[FlowGridParameters] = None


RadioOutput = Union[RadioImage, RadioSpectrum]


def _read_csv(path: str | Path) -> Tuple[Dict[str, str], List[List[str]]]:
    metadata: Dict[str, str] = {}
    rows = []
    with Path(path).open(newline="") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                comment = stripped[1:].strip()
                if ":" in comment:
                    key, value = comment.split(":", 1)
                    metadata[key.strip().lower()] = value.strip()
                continue
            rows.extend(csv.reader([line]))
    if not rows:
        raise ValueError(f"{path} contains no CSV table")
    return metadata, rows


def _float_vector(values: List[str], description: str) -> NDArray[np.float64]:
    try:
        result = np.asarray([float(value) for value in values], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must contain numeric values") from error
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{description} must contain finite values")
    return result


def _validate_centers(values: NDArray[np.float64], description: str) -> None:
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(f"{description} coordinates must be strictly increasing")


def _read_physical_scales(metadata: Dict[str, str]) -> Optional[PhysicalScales]:
    keys = ("n0_cm^-3", "temperature0_k", "length0_cm")
    present = [key in metadata for key in keys]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("CSV must provide all three physical scale metadata values")
    try:
        values = [float(metadata[key]) for key in keys]
    except ValueError as error:
        raise ValueError("physical scale metadata must be numeric") from error
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("physical scale metadata must be finite and positive")
    return PhysicalScales(*values)


def _read_flow_grid_parameters(
    metadata: Dict[str, str],
) -> Optional[FlowGridParameters]:
    keys = ("flow_resolution", "flow_bounds_r0")
    present = [key in metadata for key in keys]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("CSV must provide both flow grid metadata values")
    resolution_values = _float_vector(
        metadata["flow_resolution"].split(","), "flow_resolution"
    )
    bounds_values = _float_vector(
        metadata["flow_bounds_r0"].split(","), "flow_bounds_r0"
    )
    if (
        resolution_values.size != 3
        or np.any(resolution_values < 1)
        or np.any(resolution_values != np.floor(resolution_values))
    ):
        raise ValueError("flow_resolution must contain three positive integers")
    if bounds_values.size != 6 or not all(
        bounds_values[2 * axis] < bounds_values[2 * axis + 1]
        for axis in range(3)
    ):
        raise ValueError("flow_bounds_r0 must contain three increasing bound pairs")
    return FlowGridParameters(
        tuple(int(value) for value in resolution_values),
        tuple(float(value) for value in bounds_values),
    )


def _read_linford_spectrum() -> Tuple[
    NDArray[np.float64], NDArray[np.float64]
]:
    with _LINFORD_DATA_PATH.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Frequency_GHz", "Flux_mJy"]:
            raise ValueError(
                f"{_LINFORD_DATA_PATH} must have Frequency_GHz,Flux_mJy columns"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{_LINFORD_DATA_PATH} contains no comparison points")
    frequencies = _float_vector(
        [row["Frequency_GHz"] for row in rows], "Linford frequencies"
    )
    flux = _float_vector([row["Flux_mJy"] for row in rows], "Linford fluxes")
    _validate_centers(frequencies, "Linford frequency")
    if np.any(frequencies <= 0) or np.any(flux < 0):
        raise ValueError("Linford frequencies must be positive and fluxes non-negative")
    return frequencies, flux


def read_radio_output(
    path: str | Path, mode: str = "auto"
) -> RadioOutput:
    """Read an image or spectrum CSV written by ``rff-transfer``."""
    if mode not in ("auto", "image", "spectrum"):
        raise ValueError("mode must be 'auto', 'image', or 'spectrum'")
    metadata, rows = _read_csv(path)
    header = rows[0]
    scales = _read_physical_scales(metadata)
    flow_grid = _read_flow_grid_parameters(metadata)

    if mode == "auto":
        if header and header[0] in ("u_over_R0", "x_over_R0"):
            mode = "image"
        elif (header and header[0] == "frequency_hz"
              and len(header) == 2
              and header[1] in ("flux_density_jy", "projected_intensity_area")):
            mode = "spectrum"
        else:
            raise ValueError("could not identify CSV as a radio image or spectrum")

    if mode == "image":
        if (not header or header[0] not in ("u_over_R0", "x_over_R0")
                or len(header) < 2):
            raise ValueError("image CSV must start with u_over_R0 or x_over_R0")
        axis_names = ("u", "v") if header[0] == "u_over_R0" else ("x", "y")
        v = _float_vector(header[1:], "image v")
        data_rows = rows[1:]
        if not data_rows:
            raise ValueError("image CSV contains no pixel rows")
        u_values = []
        image_rows = []
        for row in data_rows:
            if len(row) != len(header):
                raise ValueError("image CSV rows have inconsistent column counts")
            u_values.append(row[0])
            image_rows.append(row[1:])
        u = _float_vector(u_values, "image u")
        intensity = np.asarray(
            [_float_vector(row, "image intensity") for row in image_rows],
            dtype=np.float64,
        )
        _validate_centers(u, "image u")
        _validate_centers(v, "image v")
        if not np.all(np.isfinite(intensity)) or np.any(intensity < 0):
            raise ValueError("image intensities must be finite and non-negative")

        frequency = None
        if "frequency_hz" in metadata:
            frequency = float(metadata["frequency_hz"])
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError("image frequency_hz metadata must be positive")
        los = None
        if "line_of_sight" in metadata:
            components = _float_vector(
                metadata["line_of_sight"].split(","), "line_of_sight"
            )
            if components.size != 3:
                raise ValueError("line_of_sight metadata must contain three values")
            los = tuple(float(component) for component in components)
        pixel_area = None
        if "projected_pixel_area_cm2" in metadata:
            pixel_area = float(metadata["projected_pixel_area_cm2"])
            if not np.isfinite(pixel_area) or pixel_area <= 0:
                raise ValueError("projected_pixel_area_cm2 must be positive")
        return RadioImage(
            u,
            v,
            intensity,
            frequency,
            los,
            metadata.get(
                "intensity units", "erg s^-1 cm^-2 Hz^-1 sr^-1"
            ),
            pixel_area,
            axis_names,
            scales,
            flow_grid,
        )

    if header not in (
        ["frequency_hz", "flux_density_jy"],
        ["frequency_hz", "projected_intensity_area"],
    ):
        raise ValueError(
            "spectrum CSV header must contain frequency_hz and "
            "flux_density_jy or projected_intensity_area"
        )
    if len(rows) < 2:
        raise ValueError("spectrum CSV contains no samples")
    if any(len(row) != 2 for row in rows[1:]):
        raise ValueError("spectrum CSV rows must contain two values")
    frequencies = _float_vector([row[0] for row in rows[1:]], "frequency")
    _validate_centers(frequencies, "frequency")
    values = _float_vector([row[1] for row in rows[1:]], "spectrum values")
    if np.any(frequencies <= 0) or np.any(values < 0):
        raise ValueError("frequencies must be positive and spectrum values non-negative")
    if header[1] == "projected_intensity_area":
        return RadioSpectrum(
            frequencies,
            None,
            metadata.get("projected intensity-area units", "erg s^-1 Hz^-1"),
            values,
            scales,
            flow_grid,
        )
    return RadioSpectrum(
        frequencies,
        values,
        metadata.get("integrated flux density units", "Jy"),
        scales=scales,
        flow_grid=flow_grid,
    )


def _cell_edges(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    if centers.size == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5])
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([first], midpoints, [last]))


def projected_area_to_flux_jy(
    projected_intensity_area: NDArray[np.float64] | float, distance_pc: float
):
    """Convert the distance-independent image integral to observed flux."""
    if not np.isfinite(distance_pc) or distance_pc <= 0:
        raise ValueError("distance_pc must be finite and positive")
    return (
        np.asarray(projected_intensity_area)
        / (distance_pc * _CM_PER_PC) ** 2
        / _CGS_PER_JANSKY
    )


def _add_distance_slider(figure: Any, distance_pc: float, callback: Any):
    from matplotlib.widgets import Slider

    log_distance = np.log10(distance_pc)
    slider_axis = figure.add_axes((0.18, 0.06, 0.64, 0.04))
    slider = Slider(
        slider_axis,
        "log10(distance / pc)",
        log_distance - 0.5,
        log_distance + 0.5,
        valinit=log_distance,
        valstep=0.01,
    )
    slider.on_changed(lambda value: callback(10.0**value))
    return slider


def _add_linford_checkbutton(figure: Any, axis: Any) -> None:
    from matplotlib.widgets import CheckButtons

    frequencies_ghz, flux_mjy = _read_linford_spectrum()
    points, = axis.plot(
        frequencies_ghz,
        flux_mjy,
        linestyle="none",
        marker="o",
        color="tab:orange",
        label="Linford et al. (2019)",
        visible=False,
        zorder=5,
    )
    axis.relim(visible_only=True)
    axis.autoscale_view()
    check_axis = figure.add_axes((0.75, 0.11, 0.23, 0.1))
    check = CheckButtons(check_axis, ["Linford et al. (2019)"], [False])

    def toggle_comparison(_label: str) -> None:
        points.set_visible(not points.get_visible())
        legend = axis.get_legend()
        if points.get_visible():
            axis.legend(handles=[points], loc="best")
        elif legend is not None:
            legend.remove()
        axis.relim(visible_only=True)
        axis.autoscale_view(scalex=True, scaley=True)
        figure.canvas.draw_idle()

    check.on_clicked(toggle_comparison)
    figure._rff_linford_checkbutton = check


def plot_radio_output(
    output: RadioOutput, distance_pc: float = 100.0
) -> Tuple[Any, Optional[Any]]:
    """Create a Matplotlib figure and optional distance slider."""
    try:
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Radio-output plotting requires Matplotlib; "
            "install with `python -m pip install -e '.[visualization]'`"
        ) from error

    if not np.isfinite(distance_pc) or distance_pc <= 0:
        raise ValueError("distance_pc must be finite and positive")
    figure, axis = plt.subplots(figsize=(8, 6))
    slider = None
    has_comparison_toggle = False
    if isinstance(output, RadioImage):
        u_edges = _cell_edges(output.u)
        v_edges = _cell_edges(output.v)
        image = axis.imshow(
            output.intensity.T,
            origin="lower",
            extent=(u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]),
            aspect="equal",
            interpolation="nearest",
        )
        axis.set_xlabel(f"{output.axis_names[0]} / R0")
        axis.set_ylabel(f"{output.axis_names[1]} / R0")
        title = "Free-free radio image"
        if output.frequency_hz is not None:
            title += f" at {output.frequency_hz:.6g} Hz"
        if output.line_of_sight is not None:
            title += "\nLOS = (" + ", ".join(
                f"{value:.4g}" for value in output.line_of_sight
            ) + ")"
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label=f"Specific intensity [{output.units}]")
        if output.projected_pixel_area_cm2 is not None:
            projected_area = (
                float(np.sum(output.intensity)) * output.projected_pixel_area_cm2
            )
            figure.subplots_adjust(bottom=0.22)
            annotation = figure.text(0.5, 0.14, "", ha="center")

            def update_image_distance(distance: float) -> None:
                flux_mjy = (
                    projected_area_to_flux_jy(projected_area, distance)
                    * _MILLIJANSKY_PER_JANSKY
                )
                annotation.set_text(
                    f"Flux density at {distance:.5g} pc: "
                    f"{float(flux_mjy):.6g} mJy"
                )
                figure.canvas.draw_idle()

            update_image_distance(distance_pc)
            slider = _add_distance_slider(
                figure, distance_pc, update_image_distance
            )
        else:
            figure.text(
                0.5,
                0.08,
                "Distance scaling unavailable in this legacy image file; "
                "rerun rff-transfer.",
                ha="center",
            )
    else:
        if output.projected_intensity_area is not None:
            flux_mjy = (
                projected_area_to_flux_jy(
                    output.projected_intensity_area, distance_pc
                )
                * _MILLIJANSKY_PER_JANSKY
            )
            line, = axis.plot(output.frequency_hz / 1e9, flux_mjy, marker="o")
            figure.subplots_adjust(bottom=0.22)

            def update_spectrum_distance(distance: float) -> None:
                line.set_ydata(
                    projected_area_to_flux_jy(
                        output.projected_intensity_area, distance
                    ) * _MILLIJANSKY_PER_JANSKY
                )
                axis.relim(visible_only=True)
                axis.autoscale_view(scalex=False, scaley=True)
                axis.set_title(
                    f"Free-free radio spectrum at {distance:.5g} pc"
                )
                figure.canvas.draw_idle()

            update_spectrum_distance(distance_pc)
            slider = _add_distance_slider(
                figure, distance_pc, update_spectrum_distance
            )
        else:
            line, = axis.plot(
                output.frequency_hz / 1e9,
                output.flux_density_jy * _MILLIJANSKY_PER_JANSKY,
                marker="o",
            )
            axis.set_title("Legacy radio spectrum (fixed-distance flux)")
        axis.set_xscale("log")
        plotted_flux = line.get_ydata()
        if np.all(plotted_flux > 0):
            axis.set_yscale("log")
        axis.set_xlabel("Frequency [GHz]")
        axis.set_ylabel("Flux density [mJy]")
        if output.scales is not None:
            scales = output.scales
            legend = (
                rf"$n_0 = {scales.number_density_cm3:.3g}\,"
                rf"\mathrm{{cm^{{-3}}}}$" + "\n"
                + rf"$T_0 = {scales.temperature_k:.3g}\,\mathrm{{K}}$" + "\n"
                + rf"$R_0 = {scales.length_solar_radii:.3g}\,R_\odot$"
            )
            if output.flow_grid is not None:
                nx, ny, nz = output.flow_grid.resolution
                lx, ly, lz = output.flow_grid.box_dimensions
                legend += (
                    "\n"
                    + rf"$N = {nx} \times {ny} \times {nz}$"
                    + "\n"
                    + rf"$\mathrm{{Box}} = {lx:.3g} \times {ly:.3g} "
                    + rf"\times {lz:.3g}\,R_0$"
                )
            axis.text(
                0.98,
                0.02,
                legend,
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
                zorder=10,
            )
        if output.projected_intensity_area is None:
            figure.text(
                0.5,
                0.08,
                "Distance scaling unavailable in this legacy spectrum; "
                "rerun rff-transfer.",
                ha="center",
            )
        _add_linford_checkbutton(figure, axis)
        has_comparison_toggle = True
        axis.grid(True, which="both", alpha=0.3)
    if has_comparison_toggle:
        figure.subplots_adjust(bottom=0.24, top=0.9)
    elif slider is None:
        figure.tight_layout()
    else:
        figure.subplots_adjust(bottom=0.2, top=0.9)
    return figure, slider


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rff-plot",
        description="Plot an image or spectrum CSV produced by rff-transfer",
    )
    parser.add_argument("input", help="radio image or spectrum CSV")
    parser.add_argument(
        "--mode",
        choices=("auto", "image", "spectrum"),
        default="auto",
        help="output type (detected from CSV header by default)",
    )
    parser.add_argument(
        "--distance-pc",
        type=float,
        default=895.0,
        help="initial slider distance in parsecs (default: 895)",
    )
    args = parser.parse_args()

    figure, slider = plot_radio_output(
        read_radio_output(args.input, args.mode), args.distance_pc
    )
    from matplotlib import pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
