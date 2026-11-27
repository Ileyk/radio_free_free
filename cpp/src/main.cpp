#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::array<char, 8> kMagic{{'R', 'F', 'F', 'L', 'O', 'W', '3', '\0'}};
constexpr std::array<char, 8> kPreviousMagic{{'R', 'F', 'F', 'L', 'O', 'W', '2', '\0'}};
constexpr std::array<char, 8> kLegacyMagic{{'R', 'F', 'F', 'L', 'O', 'W', '1', '\0'}};
constexpr std::size_t kFieldCount = 5;
constexpr std::size_t kPreviousFieldCount = 4;
constexpr std::size_t kLegacyFieldCount = 5;
constexpr double kPi = 3.14159265358979323846;
constexpr double kPlanck = 6.62607015e-27;
constexpr double kBoltzmann = 1.380649e-16;
constexpr double kLightSpeed = 2.99792458e10;
constexpr double kFreeFreeCoefficient = 3.692e8;
constexpr double kSolarRadiusCm = 6.957e10;
constexpr double kSqrtThreeOverPi = 1.7320508075688772935 / kPi;
constexpr std::size_t kAdaptiveSegmentFieldCount = 6;
constexpr std::size_t kMaximumAdaptiveSegmentsPerRay = 262144;
constexpr int kMaximumAdaptiveDepth = 20;
using FieldView = Kokkos::View<double****, Kokkos::LayoutRight>;
using HostFieldView =
    Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::HostSpace>;
using RaySampleView = Kokkos::View<double****, Kokkos::LayoutRight>;
using AdaptiveSegmentView = Kokkos::View<double**, Kokkos::LayoutRight>;
using RayOffsetView = Kokkos::View<std::size_t*>;

KOKKOS_INLINE_FUNCTION
double free_free_gaunt_factor(const double temperature,
                              const double frequency,
                              const double ionic_charge) {
  const double log_argument =
      Kokkos::log(ionic_charge) + Kokkos::log(frequency / 1.0e9) -
      1.5 * Kokkos::log(temperature / 1.0e4);
  const double exponent = 5.960 - kSqrtThreeOverPi * log_argument;
  if (exponent > 1.0) {
    return exponent + Kokkos::log(1.0 + Kokkos::exp(1.0 - exponent));
  }
  return 1.0 + Kokkos::log(1.0 + Kokkos::exp(exponent - 1.0));
}

KOKKOS_INLINE_FUNCTION
double one_minus_exp_negative(const double value) {
  if (value < 1.0e-5) {
    return value * (1.0 - value * (0.5 - value / 6.0));
  }
  return 1.0 - Kokkos::exp(-value);
}

KOKKOS_INLINE_FUNCTION
double exp_minus_one(const double value) {
  if (value < 1.0e-5) {
    return value * (1.0 + value * (0.5 + value / 6.0));
  }
  return Kokkos::exp(value) - 1.0;
}

KOKKOS_INLINE_FUNCTION
double planck_source_function(const double frequency,
                              const double temperature) {
  const double x_nu = kPlanck * frequency / (kBoltzmann * temperature);
  const double planck_exponent =
      x_nu < 700.0 ? exp_minus_one(x_nu) : Kokkos::exp(700.0);
  return (2.0 * kPlanck * frequency * frequency * frequency /
          (kLightSpeed * kLightSpeed)) /
         planck_exponent;
}

struct Options {
  std::string input;
  std::string output;
  std::string mode = "spectrum";
  double number_density_scale = 0.0;
  double temperature_scale = 0.0;
  double length_scale_cm = 0.0;
  double frequency_min = 0.0;
  double frequency_max = 0.0;
  double relative_intensity_threshold = 1.0e-3;
  double integration_tolerance = 1.0e-3;
  std::size_t frequency_count = 64;
  double theta_deg = 0.0;
  double phi_deg = 0.0;
  bool adaptive_longitudinal = false;
  bool integration_tolerance_given = false;
};

struct Grid {
  std::size_t nx;
  std::size_t ny;
  std::size_t nz;
  std::array<double, 6> bounds;
  std::vector<double> fields;
  bool has_occultation = false;
};

struct GridFieldAccessor {
  const double* values;
  std::size_t ny;
  std::size_t nz;

  KOKKOS_INLINE_FUNCTION
  double operator()(const std::size_t i, const std::size_t j,
                    const std::size_t k, const std::size_t field) const {
    return values[((i * ny + j) * nz + k) * kFieldCount + field];
  }
};

struct Vector3 {
  double x;
  double y;
  double z;
};

struct ViewGeometry {
  std::size_t nx;
  std::size_t ny;
  std::size_t nlos;
  double u_min;
  double v_min;
  double s_min;
  double du;
  double dv;
  double ds;
  Vector3 u;
  Vector3 v;
  Vector3 los;
};

struct FlowSample {
  double temperature = 0.0;
  double emission_measure = 0.0;
  double ionic_charge = 0.0;
  bool opaque = false;
};

struct AdaptiveSegment {
  double lower = 0.0;
  double upper = 0.0;
  FlowSample sample;
};

struct SegmentOperator {
  double optical_depth = 0.0;
  double transmission = 1.0;
  double source_contribution = 0.0;
};

struct AdaptiveRaySamples {
  AdaptiveSegmentView segments;
  RayOffsetView offsets;
};

void usage(std::ostream& out) {
  out << "Usage: rff-transfer --input flow.rff --output result.csv"
         " --n0 cm^-3 --temperature0 K --length0 R_sun\n"
         "  Spectrum: [--mode spectrum] --frequency-min-hz Hz"
         " --frequency-max-hz Hz [--frequency-count N]\n"
         "  Stopping: [--relative-intensity-threshold (0,1)]"
         " (alias --relative_intensity_threshold; default 1e-3)\n"
         "  Adaptive: [--adaptive-longitudinal]"
         " [--integration-tolerance (0,1)] (default 1e-3)\n"
         "  Viewing:  [--view-theta-deg 0..180] [--view-phi-deg degrees]\n"
         "Theta is measured from +z; phi is measured from +x in the xy plane.\n";
}

double parse_finite(const std::string& value, const std::string& name) {
  std::size_t consumed = 0;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::invalid_argument(name + " must be a finite number");
  }
  return parsed;
}

double parse_positive(const std::string& value, const std::string& name) {
  const double parsed = parse_finite(value, name);
  if (parsed <= 0.0) {
    throw std::invalid_argument(name + " must be positive");
  }
  return parsed;
}

std::size_t parse_count(const std::string& value) {
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument("--frequency-count must be an integer >= 2");
  }
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size() ||
      parsed > std::numeric_limits<std::size_t>::max() || parsed < 2) {
    throw std::invalid_argument("--frequency-count must be an integer >= 2");
  }
  return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--help" || key == "-h") {
      usage(std::cout);
      std::exit(0);
    }
    if (key == "--adaptive-longitudinal") {
      options.adaptive_longitudinal = true;
      continue;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + key);
    }
    const std::string value(argv[++i]);
    if (key == "--input") {
      options.input = value;
    } else if (key == "--output") {
      options.output = value;
    } else if (key == "--mode") {
      options.mode = value;
    } else if (key == "--n0") {
      options.number_density_scale = parse_positive(value, key);
    } else if (key == "--temperature0") {
      options.temperature_scale = parse_positive(value, key);
    } else if (key == "--length0") {
      const double length0_solar_radii = parse_positive(value, key);
      if (length0_solar_radii >
          std::numeric_limits<double>::max() / kSolarRadiusCm) {
        throw std::invalid_argument("--length0 is too large");
      }
      options.length_scale_cm = length0_solar_radii * kSolarRadiusCm;
    } else if (key == "--frequency-min-hz") {
      options.frequency_min = parse_positive(value, key);
    } else if (key == "--frequency-max-hz") {
      options.frequency_max = parse_positive(value, key);
    } else if (key == "--frequency-count") {
      options.frequency_count = parse_count(value);
    } else if (key == "--relative-intensity-threshold" ||
               key == "--relative_intensity_threshold") {
      options.relative_intensity_threshold = parse_positive(value, key);
    } else if (key == "--integration-tolerance") {
      options.integration_tolerance = parse_positive(value, key);
      options.integration_tolerance_given = true;
    } else if (key == "--view-theta-deg") {
      options.theta_deg = parse_finite(value, key);
    } else if (key == "--view-phi-deg") {
      options.phi_deg = parse_finite(value, key);
    } else {
      throw std::invalid_argument("unknown option: " + key);
    }
  }

  if (options.input.empty() || options.output.empty() ||
      options.number_density_scale == 0.0 || options.temperature_scale == 0.0 ||
      options.length_scale_cm == 0.0) {
    throw std::invalid_argument(
        "--input, --output, --n0, --temperature0, and --length0 are required");
  }
  if (options.theta_deg < 0.0 || options.theta_deg > 180.0) {
    throw std::invalid_argument("--view-theta-deg must be between 0 and 180");
  }

  if (options.mode != "spectrum") {
    throw std::invalid_argument("--mode only supports 'spectrum'");
  }
  if (options.frequency_min == 0.0 || options.frequency_max == 0.0 ||
      options.frequency_min >= options.frequency_max) {
    throw std::invalid_argument(
        "spectrum mode requires --frequency-min-hz less than "
        "--frequency-max-hz");
  }
  if (options.relative_intensity_threshold >= 1.0) {
    throw std::invalid_argument(
        "--relative-intensity-threshold must be less than one");
  }
  if (options.integration_tolerance >= 1.0) {
    throw std::invalid_argument("--integration-tolerance must be less than one");
  }
  if (options.integration_tolerance_given && !options.adaptive_longitudinal) {
    throw std::invalid_argument(
        "--integration-tolerance requires --adaptive-longitudinal");
  }
  return options;
}

template <typename T>
T read_scalar(std::ifstream& stream, const std::string& description) {
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!stream) {
    throw std::runtime_error("truncated flow file while reading " + description);
  }
  return value;
}

Grid read_grid(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open flow file: " + path);
  }
  std::array<char, 8> magic{};
  stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!stream ||
      (magic != kMagic && magic != kPreviousMagic && magic != kLegacyMagic)) {
    throw std::runtime_error("invalid flow file magic/version");
  }
  const bool has_occultation_mask = magic == kMagic;
  const std::size_t input_field_count =
      magic == kPreviousMagic ? kPreviousFieldCount : kLegacyFieldCount;

  const auto nx = read_scalar<std::uint32_t>(stream, "nx");
  const auto ny = read_scalar<std::uint32_t>(stream, "ny");
  const auto nz = read_scalar<std::uint32_t>(stream, "nz");
  if (nx == 0 || ny == 0 || nz == 0) {
    throw std::runtime_error("flow grid dimensions must be non-zero");
  }
  Grid grid{nx, ny, nz, {}, {}};
  for (double& bound : grid.bounds) {
    bound = read_scalar<double>(stream, "grid bounds");
    if (!std::isfinite(bound)) {
      throw std::runtime_error("flow grid bounds must be finite");
    }
  }
  if (!(grid.bounds[0] < grid.bounds[1] && grid.bounds[2] < grid.bounds[3] &&
        grid.bounds[4] < grid.bounds[5])) {
    throw std::runtime_error("invalid flow grid bounds");
  }

  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  const std::size_t allocation_field_count =
      std::max(input_field_count, kFieldCount);
  if (static_cast<std::size_t>(nx) > max_size / ny ||
      static_cast<std::size_t>(nx) * ny >
          max_size / nz / allocation_field_count) {
    throw std::runtime_error("flow grid dimensions overflow addressable memory");
  }
  const std::size_t value_count =
      static_cast<std::size_t>(nx) * ny * nz * input_field_count;
  if (value_count > max_size / sizeof(double) ||
      value_count > static_cast<std::size_t>(
                        std::numeric_limits<std::streamsize>::max()) /
                        sizeof(double)) {
    throw std::runtime_error("flow payload is too large to read");
  }
  std::vector<double> input_fields(value_count);
  stream.read(reinterpret_cast<char*>(input_fields.data()),
              static_cast<std::streamsize>(value_count * sizeof(double)));
  if (!stream) {
    throw std::runtime_error("truncated flow field payload");
  }
  if (stream.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("unexpected trailing data in flow file");
  }
  const std::size_t cell_count = static_cast<std::size_t>(nx) * ny * nz;
  grid.fields.resize(cell_count * kFieldCount);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    std::copy_n(input_fields.begin() + cell * input_field_count,
                kFieldCount - 1, grid.fields.begin() + cell * kFieldCount);
    grid.fields[cell * kFieldCount + kFieldCount - 1] =
        has_occultation_mask
            ? input_fields[cell * input_field_count + kFieldCount - 1]
            : 0.0;
    grid.has_occultation |=
        grid.fields[cell * kFieldCount + kFieldCount - 1] > 0.0;
  }
  return grid;
}

Vector3 cross(const Vector3& a, const Vector3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

double dot(const Vector3& a, const Vector3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector3 normalized(const Vector3& value) {
  const double magnitude = std::sqrt(dot(value, value));
  if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
    throw std::runtime_error("cannot construct viewing coordinate basis");
  }
  return {value.x / magnitude, value.y / magnitude, value.z / magnitude};
}

std::size_t samples_for_span(const double span, const double spacing) {
  const double count = std::ceil(span / spacing);
  if (!std::isfinite(count) || count < 1.0 ||
      count > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("projected view grid is too large");
  }
  return static_cast<std::size_t>(count);
}

ViewGeometry make_view_geometry(const Grid& grid, const Options& options) {
  const double theta = options.theta_deg * kPi / 180.0;
  const double phi = options.phi_deg * kPi / 180.0;
  const Vector3 los{std::sin(theta) * std::cos(phi),
                    std::sin(theta) * std::sin(phi), std::cos(theta)};

  Vector3 reference{1.0, 0.0, 0.0};
  Vector3 horizontal{reference.x - dot(reference, los) * los.x,
                     reference.y - dot(reference, los) * los.y,
                     reference.z - dot(reference, los) * los.z};
  if (dot(horizontal, horizontal) < 1.0e-24) {
    reference = {0.0, 1.0, 0.0};
    horizontal = {reference.x - dot(reference, los) * los.x,
                  reference.y - dot(reference, los) * los.y,
                  reference.z - dot(reference, los) * los.z};
  }
  const Vector3 u = normalized(horizontal);
  const Vector3 v = normalized(cross(los, u));

  double u_min = std::numeric_limits<double>::infinity();
  double u_max = -u_min;
  double v_min = u_min;
  double v_max = -u_min;
  double s_min = u_min;
  double s_max = -u_min;
  for (int corner = 0; corner < 8; ++corner) {
    const Vector3 point{
        grid.bounds[(corner & 1) ? 1 : 0],
        grid.bounds[(corner & 2) ? 3 : 2],
        grid.bounds[(corner & 4) ? 5 : 4],
    };
    const double pu = dot(point, u);
    const double pv = dot(point, v);
    const double ps = dot(point, los);
    u_min = std::min(u_min, pu);
    u_max = std::max(u_max, pu);
    v_min = std::min(v_min, pv);
    v_max = std::max(v_max, pv);
    s_min = std::min(s_min, ps);
    s_max = std::max(s_max, ps);
  }

  const double dx = (grid.bounds[1] - grid.bounds[0]) / grid.nx;
  const double dy = (grid.bounds[3] - grid.bounds[2]) / grid.ny;
  const double dz = (grid.bounds[5] - grid.bounds[4]) / grid.nz;
  const double sampling_spacing = std::min({dx, dy, dz});
  const std::size_t nx = samples_for_span(u_max - u_min, sampling_spacing);
  const std::size_t ny = samples_for_span(v_max - v_min, sampling_spacing);
  const std::size_t nlos = samples_for_span(s_max - s_min, sampling_spacing);
  if (nx > std::numeric_limits<std::size_t>::max() / ny) {
    throw std::runtime_error("projected image dimensions overflow");
  }
  return {nx, ny, nlos, u_min, v_min, s_min,
          (u_max - u_min) / nx, (v_max - v_min) / ny,
          (s_max - s_min) / nlos, u, v, los};
}

template <typename FieldViewType>
KOKKOS_INLINE_FUNCTION
double sample_field(const FieldViewType& fields, const double x,
                    const double y, const double z, const std::size_t field,
                    const std::size_t nx, const std::size_t ny,
                    const std::size_t nz, const double x_min,
                    const double x_max, const double y_min,
                    const double y_max, const double z_min,
                    const double z_max, const double dx, const double dy,
                    const double dz) {
  if (x < x_min || x > x_max || y < y_min || y > y_max || z < z_min ||
      z > z_max) {
    return 0.0;
  }
  double gx = (x - (x_min + 0.5 * dx)) / dx;
  double gy = (y - (y_min + 0.5 * dy)) / dy;
  double gz = (z - (z_min + 0.5 * dz)) / dz;
  gx = gx < 0.0 ? 0.0 : (gx > nx - 1 ? nx - 1 : gx);
  gy = gy < 0.0 ? 0.0 : (gy > ny - 1 ? ny - 1 : gy);
  gz = gz < 0.0 ? 0.0 : (gz > nz - 1 ? nz - 1 : gz);

  const std::size_t ix0 = static_cast<std::size_t>(gx);
  const std::size_t iy0 = static_cast<std::size_t>(gy);
  const std::size_t iz0 = static_cast<std::size_t>(gz);
  const std::size_t ix1 = ix0 + 1 < nx ? ix0 + 1 : ix0;
  const std::size_t iy1 = iy0 + 1 < ny ? iy0 + 1 : iy0;
  const std::size_t iz1 = iz0 + 1 < nz ? iz0 + 1 : iz0;
  const double tx = gx - ix0;
  const double ty = gy - iy0;
  const double tz = gz - iz0;

  double result = 0.0;
  for (int i = 0; i < 2; ++i) {
    const std::size_t ix = i == 0 ? ix0 : ix1;
    const double wx = i == 0 ? 1.0 - tx : tx;
    for (int j = 0; j < 2; ++j) {
      const std::size_t iy = j == 0 ? iy0 : iy1;
      const double wy = j == 0 ? 1.0 - ty : ty;
      for (int k = 0; k < 2; ++k) {
        const std::size_t iz = k == 0 ? iz0 : iz1;
        const double wz = k == 0 ? 1.0 - tz : tz;
        result += wx * wy * wz * fields(ix, iy, iz, field);
      }
    }
  }
  return result;
}

template <typename FieldViewType>
FlowSample sample_flow(const FieldViewType& fields, const Grid& grid,
                       const Options& options, const double x,
                       const double y, const double z) {
  const std::size_t nx = grid.nx;
  const std::size_t ny = grid.ny;
  const std::size_t nz = grid.nz;
  const double x_min = grid.bounds[0];
  const double x_max = grid.bounds[1];
  const double y_min = grid.bounds[2];
  const double y_max = grid.bounds[3];
  const double z_min = grid.bounds[4];
  const double z_max = grid.bounds[5];
  const double dx = (x_max - x_min) / nx;
  const double dy = (y_max - y_min) / ny;
  const double dz = (z_max - z_min) / nz;
  const double mask =
      sample_field(fields, x, y, z, 4, nx, ny, nz, x_min, x_max, y_min,
                   y_max, z_min, z_max, dx, dy, dz);
  if (mask > 0.5) {
    return {0.0, 0.0, 0.0, true};
  }
  const double density_ratio =
      sample_field(fields, x, y, z, 0, nx, ny, nz, x_min, x_max, y_min,
                   y_max, z_min, z_max, dx, dy, dz);
  if (density_ratio == 0.0) {
    return {};
  }
  const double temperature_ratio =
      sample_field(fields, x, y, z, 1, nx, ny, nz, x_min, x_max, y_min,
                   y_max, z_min, z_max, dx, dy, dz);
  const double electron_abundance =
      sample_field(fields, x, y, z, 2, nx, ny, nz, x_min, x_max, y_min,
                   y_max, z_min, z_max, dx, dy, dz);
  const double ion_charge2_abundance =
      sample_field(fields, x, y, z, 3, nx, ny, nz, x_min, x_max, y_min,
                   y_max, z_min, z_max, dx, dy, dz);
  const double baryon_number_density =
      options.number_density_scale * density_ratio;
  const double electron_density = baryon_number_density * electron_abundance;
  const double ion_charge2_density =
      baryon_number_density * ion_charge2_abundance;
  return {options.temperature_scale * temperature_ratio,
          electron_density * ion_charge2_density,
          electron_abundance > 0.0
              ? ion_charge2_abundance / electron_abundance
              : 0.0,
          false};
}

SegmentOperator segment_operator(const FlowSample& sample,
                                 const double frequency,
                                 const double length_cm) {
  if (sample.opaque || sample.temperature <= 0.0 ||
      sample.emission_measure <= 0.0 || sample.ionic_charge <= 0.0) {
    return {};
  }
  const double x_nu =
      kPlanck * frequency / (kBoltzmann * sample.temperature);
  const double alpha =
      kFreeFreeCoefficient / (frequency * frequency * frequency) *
      Kokkos::sqrt(1.0 / sample.temperature) * sample.emission_measure *
      one_minus_exp_negative(x_nu) *
      free_free_gaunt_factor(sample.temperature, frequency,
                             sample.ionic_charge);
  const double optical_depth = alpha * length_cm;
  const double absorption = one_minus_exp_negative(optical_depth);
  return {optical_depth, Kokkos::exp(-optical_depth),
          planck_source_function(frequency, sample.temperature) * absorption};
}

double relative_difference(const double first, const double second) {
  if (first == second) {
    return 0.0;
  }
  const double scale = std::max(std::abs(first), std::abs(second));
  if (scale == 0.0) {
    return 0.0;
  }
  if (!std::isfinite(scale)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::abs(first - second) / scale;
}

bool same_opacity(const FlowSample& center, const FlowSample& left,
                  const FlowSample& right) {
  return center.opaque == left.opaque && center.opaque == right.opaque;
}

bool refinement_criteria_pass(const FlowSample& center,
                              const FlowSample& left,
                              const FlowSample& right,
                              const double interval_length,
                              const double length_scale_cm,
                              const std::vector<double>& frequencies,
                              const double tolerance) {
  if (!same_opacity(center, left, right)) {
    return false;
  }
  if (center.opaque) {
    return true;
  }
  for (const double frequency : frequencies) {
    const SegmentOperator coarse =
        segment_operator(center, frequency,
                         interval_length * length_scale_cm);
    const SegmentOperator left_half =
        segment_operator(left, frequency,
                         interval_length * length_scale_cm * 0.5);
    const SegmentOperator right_half =
        segment_operator(right, frequency,
                         interval_length * length_scale_cm * 0.5);
    const double fine_absorption =
        one_minus_exp_negative(left_half.optical_depth +
                               right_half.optical_depth);
    const double fine_source =
        right_half.transmission * left_half.source_contribution +
        right_half.source_contribution;
    if (relative_difference(
            one_minus_exp_negative(coarse.optical_depth), fine_absorption) >
            tolerance ||
        relative_difference(coarse.source_contribution, fine_source) >
            tolerance) {
      return false;
    }
  }
  return true;
}

void validate_and_copy_fields(const Grid& grid, FieldView* fields) {
  HostFieldView host_fields;
  if (fields != nullptr) {
    host_fields = Kokkos::create_mirror_view(*fields);
  }
  for (std::size_t i = 0; i < grid.nx; ++i) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t k = 0; k < grid.nz; ++k) {
        const std::size_t cell = ((i * grid.ny + j) * grid.nz + k) * kFieldCount;
        for (std::size_t f = 0; f < kFieldCount; ++f) {
          const double value = grid.fields[cell + f];
          if (!std::isfinite(value)) {
            throw std::runtime_error("flow fields must contain finite values");
          }
          if ((f == 0 || f == 2 || f == 3) && value < 0.0) {
            throw std::runtime_error("density and abundances must be non-negative");
          }
          if (f == 1 && value < 0.0) {
            throw std::runtime_error("temperature must be non-negative");
          }
          if (f == 4 && (value < 0.0 || value > 1.0)) {
            throw std::runtime_error(
                "occultation mask values must be between zero and one");
          }
          if (fields != nullptr) {
            host_fields(i, j, k, f) = value;
          }
        }
        if (grid.fields[cell] > 0.0 && grid.fields[cell + 1] <= 0.0) {
          throw std::runtime_error(
              "temperature must be positive where density is non-zero");
        }
      }
    }
  }
  if (fields != nullptr) {
    Kokkos::deep_copy(*fields, host_fields);
  }
}

RaySampleView precompute_ray_samples(const Grid& grid, const FieldView& fields,
                                     const ViewGeometry& view,
                                     const Options& options,
                                     const bool has_occultation) {
  RaySampleView samples("ray_samples", view.nx, view.ny, view.nlos, 5);
  const double number_density0 = options.number_density_scale;
  const double temperature0 = options.temperature_scale;
  const std::size_t nx = grid.nx;
  const std::size_t ny = grid.ny;
  const std::size_t nz = grid.nz;
  const std::size_t image_ny = view.ny;
  const std::size_t nlos = view.nlos;
  const double x_min = grid.bounds[0];
  const double x_max = grid.bounds[1];
  const double y_min = grid.bounds[2];
  const double y_max = grid.bounds[3];
  const double z_min = grid.bounds[4];
  const double z_max = grid.bounds[5];
  const double dx = (x_max - x_min) / nx;
  const double dy = (y_max - y_min) / ny;
  const double dz = (z_max - z_min) / nz;
  const double u_min = view.u_min;
  const double v_min = view.v_min;
  const double s_min = view.s_min;
  const double du = view.du;
  const double dv = view.dv;
  const double ds = view.ds;
  const double ux = view.u.x;
  const double uy = view.u.y;
  const double uz = view.u.z;
  const double vx = view.v.x;
  const double vy = view.v.y;
  const double vz = view.v.z;
  const double losx = view.los.x;
  const double losy = view.los.y;
  const double losz = view.los.z;

  Kokkos::parallel_for(
      "precompute_ray_samples",
      Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(
          0, view.nx * view.ny),
      KOKKOS_LAMBDA(const std::size_t pixel) {
        const std::size_t i = pixel / image_ny;
        const std::size_t j = pixel % image_ny;
        const double u_coord = u_min + (i + 0.5) * du;
        const double v_coord = v_min + (j + 0.5) * dv;
        for (std::size_t k = 0; k < nlos; ++k) {
          const double s_coord = s_min + (k + 0.5) * ds;
          const double x = ux * u_coord + vx * v_coord + losx * s_coord;
          const double y = uy * u_coord + vy * v_coord + losy * s_coord;
          const double z = uz * u_coord + vz * v_coord + losz * s_coord;
          const double occultation_mask =
              has_occultation
                  ? sample_field(fields, x, y, z, 4, nx, ny, nz, x_min, x_max,
                                 y_min, y_max, z_min, z_max, dx, dy, dz)
                  : 0.0;
          samples(i, j, k, 2) = occultation_mask;
          samples(i, j, k, 3) = 0.0;
          if (occultation_mask > 0.5) {
            samples(i, j, k, 0) = 0.0;
            samples(i, j, k, 1) = 0.0;
            continue;
          }
          const double density_ratio = sample_field(
              fields, x, y, z, 0, nx, ny, nz, x_min, x_max, y_min, y_max,
              z_min, z_max, dx, dy, dz);
          const double baryon_number_density =
              number_density0 * density_ratio;
          if (baryon_number_density == 0.0) {
            samples(i, j, k, 0) = 0.0;
            samples(i, j, k, 1) = 0.0;
            continue;
          }
          const double temperature_ratio = sample_field(
              fields, x, y, z, 1, nx, ny, nz, x_min, x_max, y_min, y_max,
              z_min, z_max, dx, dy, dz);
          const double electron_abundance = sample_field(
              fields, x, y, z, 2, nx, ny, nz, x_min, x_max, y_min, y_max,
              z_min, z_max, dx, dy, dz);
          const double ion_charge2_abundance = sample_field(
              fields, x, y, z, 3, nx, ny, nz, x_min, x_max, y_min, y_max,
              z_min, z_max, dx, dy, dz);
          const double temperature = temperature0 * temperature_ratio;
          const double electron_density =
              baryon_number_density * electron_abundance;
          const double ion_charge2_density =
              baryon_number_density * ion_charge2_abundance;
          samples(i, j, k, 0) = temperature;
          samples(i, j, k, 1) = electron_density * ion_charge2_density;
          samples(i, j, k, 3) =
              electron_abundance > 0.0
                  ? ion_charge2_abundance / electron_abundance
                  : 0.0;
        }
        double maximum_remaining_temperature = 0.0;
        for (std::size_t k = 0; k < nlos; ++k) {
          samples(i, j, k, 4) = maximum_remaining_temperature;
          const double temperature = samples(i, j, k, 0);
          if (temperature > maximum_remaining_temperature) {
            maximum_remaining_temperature = temperature;
          }
        }
      });
  Kokkos::fence();
  return samples;
}

std::vector<double> adaptive_reference_frequencies(const Options& options) {
  const double log_min = std::log(options.frequency_min);
  const double log_max = std::log(options.frequency_max);
  const double half_decade = 0.5 * std::log(10.0);
  std::vector<double> frequencies{options.frequency_min};
  for (std::size_t step = 1;; ++step) {
    const double log_frequency = log_min + step * half_decade;
    if (log_frequency >= log_max) {
      break;
    }
    const double frequency = std::exp(log_frequency);
    if (frequency > frequencies.back() && frequency < options.frequency_max) {
      frequencies.push_back(frequency);
    }
  }
  if (frequencies.back() != options.frequency_max) {
    frequencies.push_back(options.frequency_max);
  }
  return frequencies;
}

template <typename Sampler>
std::vector<AdaptiveSegment> refine_reference_ray(
    const double s_min, const double s_max, const double base_step,
    const std::vector<double>& frequencies, const double length_scale_cm,
    const double tolerance, Sampler&& sample_at) {
  std::vector<AdaptiveSegment> segments;
  const std::size_t interval_count = samples_for_span(s_max - s_min, base_step);
  const double interval_step = (s_max - s_min) / interval_count;
  auto refine = [&](auto&& self, const double lower, const double upper,
                    const int depth) -> void {
    const double middle = lower + (upper - lower) * 0.5;
    const FlowSample center_sample = sample_at(middle);
    const double left_middle = lower + (middle - lower) * 0.5;
    const double right_middle = middle + (upper - middle) * 0.5;
    const FlowSample lower_sample = sample_at(lower);
    const FlowSample left_sample = sample_at(left_middle);
    const FlowSample right_sample = sample_at(right_middle);
    const FlowSample upper_sample = sample_at(upper);
    const bool opacity_consistent =
        same_opacity(center_sample, left_sample, right_sample);
    if (!opacity_consistent) {
      const std::array<double, 5> positions{{
          lower, left_middle, middle, right_middle, upper}};
      const std::array<FlowSample, 5> samples{{
          lower_sample, left_sample, center_sample, right_sample,
          upper_sample}};
      std::vector<double> boundaries{lower};
      for (std::size_t i = 0; i + 1 < positions.size(); ++i) {
        if (samples[i].opaque == samples[i + 1].opaque) {
          continue;
        }
        double bracket_min = positions[i];
        double bracket_max = positions[i + 1];
        const bool lower_is_opaque = samples[i].opaque;
        for (int iteration = 0; iteration < 64; ++iteration) {
          const double bracket_middle =
              bracket_min + (bracket_max - bracket_min) * 0.5;
          if (!(bracket_middle > bracket_min &&
                bracket_middle < bracket_max)) {
            break;
          }
          if (sample_at(bracket_middle).opaque == lower_is_opaque) {
            bracket_min = bracket_middle;
          } else {
            bracket_max = bracket_middle;
          }
        }
        const double boundary =
            bracket_min + (bracket_max - bracket_min) * 0.5;
        if (boundary > boundaries.back() && boundary < upper) {
          boundaries.push_back(boundary);
        }
      }
      boundaries.push_back(upper);
      for (std::size_t i = 0; i + 1 < boundaries.size(); ++i) {
        if (boundaries[i + 1] > boundaries[i]) {
          self(self, boundaries[i], boundaries[i + 1], depth + 1);
        }
      }
      return;
    }
    if (refinement_criteria_pass(center_sample, left_sample, right_sample,
                                 upper - lower, length_scale_cm, frequencies,
                                 tolerance)) {
      if (segments.size() >= kMaximumAdaptiveSegmentsPerRay) {
        throw std::runtime_error(
            "adaptive longitudinal ray exceeds the segment limit");
      }
      segments.push_back({lower, upper, center_sample});
      return;
    }
    if (depth >= kMaximumAdaptiveDepth || !(middle > lower && middle < upper)) {
      throw std::runtime_error(
          "adaptive longitudinal refinement did not converge");
    }
    self(self, lower, middle, depth + 1);
    self(self, middle, upper, depth + 1);
  };

  for (std::size_t i = 0; i < interval_count; ++i) {
    const double lower = s_min + i * interval_step;
    const double upper = i + 1 == interval_count
                             ? s_max
                             : s_min + (i + 1) * interval_step;
    refine(refine, lower, upper, 0);
  }
  return segments;
}

std::vector<AdaptiveSegment> visible_ray_segments(
    std::vector<AdaptiveSegment> segments) {
  std::size_t nearest_opaque = segments.size();
  for (std::size_t i = 0; i < segments.size(); ++i) {
    if (segments[i].sample.opaque) {
      nearest_opaque = i;
    }
  }
  if (nearest_opaque < segments.size()) {
    segments.erase(segments.begin(), segments.begin() + nearest_opaque);
  }
  return segments;
}

bool merged_operator_matches(const std::vector<AdaptiveSegment>& reference,
                             const std::size_t begin, const std::size_t end,
                             const AdaptiveSegment& candidate,
                             const std::vector<double>& frequencies,
                             const double length_scale_cm,
                             const double tolerance) {
  for (const double frequency : frequencies) {
    double reference_transmission = 1.0;
    double reference_source = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      const SegmentOperator operation = segment_operator(
          reference[i].sample, frequency,
          (reference[i].upper - reference[i].lower) * length_scale_cm);
      reference_source =
          operation.transmission * reference_source +
          operation.source_contribution;
      reference_transmission *= operation.transmission;
    }
    const SegmentOperator candidate_operation = segment_operator(
        candidate.sample, frequency,
        (candidate.upper - candidate.lower) * length_scale_cm);
    if (relative_difference(reference_transmission,
                            candidate_operation.transmission) > tolerance ||
        relative_difference(reference_source,
                            candidate_operation.source_contribution) >
            tolerance) {
      return false;
    }
  }
  return true;
}

template <typename Sampler>
std::vector<AdaptiveSegment> compress_ray_segments(
    const std::vector<AdaptiveSegment>& reference,
    const std::vector<double>& frequencies, const double length_scale_cm,
    const double tolerance, Sampler&& sample_at) {
  std::vector<AdaptiveSegment> compressed;
  std::size_t begin = 0;
  while (begin < reference.size()) {
    if (reference[begin].sample.opaque) {
      compressed.push_back(reference[begin]);
      ++begin;
      continue;
    }
    std::size_t run_end = begin;
    while (run_end < reference.size() &&
           !reference[run_end].sample.opaque) {
      ++run_end;
    }
    std::size_t accepted_end = begin + 1;
    AdaptiveSegment accepted = reference[begin];
    for (std::size_t end = begin + 2; end <= run_end; ++end) {
      const double lower = reference[begin].lower;
      const double upper = reference[end - 1].upper;
      const AdaptiveSegment candidate{
          lower, upper, sample_at(lower + (upper - lower) * 0.5)};
      if (candidate.sample.opaque ||
          !merged_operator_matches(reference, begin, end, candidate,
                                   frequencies, length_scale_cm, tolerance)) {
        break;
      }
      accepted = candidate;
      accepted_end = end;
    }
    compressed.push_back(accepted);
    begin = accepted_end;
  }
  return compressed;
}

double integrate_adaptive_ray(const std::vector<AdaptiveSegment>& segments,
                              const double frequency,
                              const double length_scale_cm) {
  double intensity = 0.0;
  double foreground_optical_depth = 0.0;
  for (std::size_t i = segments.size(); i > 0; --i) {
    const AdaptiveSegment& segment = segments[i - 1];
    if (segment.sample.opaque) {
      break;
    }
    const SegmentOperator operation = segment_operator(
        segment.sample, frequency,
        (segment.upper - segment.lower) * length_scale_cm);
    intensity += Kokkos::exp(-foreground_optical_depth) *
                 operation.source_contribution;
    foreground_optical_depth += operation.optical_depth;
  }
  return intensity;
}

AdaptiveRaySamples build_adaptive_ray_samples(
    const Grid& grid, const ViewGeometry& view, const Options& options,
    const std::vector<double>& frequencies) {
  const std::size_t pixel_count = view.nx * view.ny;
  std::vector<std::array<double, kAdaptiveSegmentFieldCount>> packed_segments;
  std::vector<std::size_t> host_offsets(pixel_count + 1, 0);
  const GridFieldAccessor fields{grid.fields.data(), grid.ny, grid.nz};
  const double ray_s_min = view.s_min;
  const double ray_s_max = view.s_min + view.nlos * view.ds;
  const double losx = view.los.x;
  const double losy = view.los.y;
  const double losz = view.los.z;
  const double ux = view.u.x;
  const double uy = view.u.y;
  const double uz = view.u.z;
  const double vx = view.v.x;
  const double vy = view.v.y;
  const double vz = view.v.z;
  const double x_min = grid.bounds[0];
  const double x_max = grid.bounds[1];
  const double y_min = grid.bounds[2];
  const double y_max = grid.bounds[3];
  const double z_min = grid.bounds[4];
  const double z_max = grid.bounds[5];

  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    const std::size_t i = pixel / view.ny;
    const std::size_t j = pixel % view.ny;
    const double u_coord = view.u_min + (i + 0.5) * view.du;
    const double v_coord = view.v_min + (j + 0.5) * view.dv;
    const double origin_x = ux * u_coord + vx * v_coord;
    const double origin_y = uy * u_coord + vy * v_coord;
    const double origin_z = uz * u_coord + vz * v_coord;
    double clipped_min = ray_s_min;
    double clipped_max = ray_s_max;
    const std::array<double, 3> origins{{origin_x, origin_y, origin_z}};
    const std::array<double, 3> directions{{losx, losy, losz}};
    const std::array<double, 3> lower_bounds{{x_min, y_min, z_min}};
    const std::array<double, 3> upper_bounds{{x_max, y_max, z_max}};
    bool intersects_grid = true;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (std::abs(directions[axis]) < 1.0e-14) {
        if (origins[axis] < lower_bounds[axis] ||
            origins[axis] > upper_bounds[axis]) {
          intersects_grid = false;
          break;
        }
        continue;
      }
      double first =
          (lower_bounds[axis] - origins[axis]) / directions[axis];
      double second =
          (upper_bounds[axis] - origins[axis]) / directions[axis];
      if (first > second) {
        std::swap(first, second);
      }
      clipped_min = std::max(clipped_min, first);
      clipped_max = std::min(clipped_max, second);
      if (clipped_min >= clipped_max) {
        intersects_grid = false;
        break;
      }
    }
    if (intersects_grid) {
      const auto sample_at = [&](const double s) {
        return sample_flow(fields, grid, options, origin_x + losx * s,
                           origin_y + losy * s, origin_z + losz * s);
      };
      std::vector<AdaptiveSegment> reference = refine_reference_ray(
          clipped_min, clipped_max, view.ds, frequencies,
          options.length_scale_cm,
          options.integration_tolerance, sample_at);
      reference = visible_ray_segments(std::move(reference));
      const std::vector<AdaptiveSegment> compressed = compress_ray_segments(
          reference, frequencies, options.length_scale_cm,
          options.integration_tolerance, sample_at);
      bool compressed_is_accurate = true;
      for (const double frequency : frequencies) {
        const double reference_intensity = integrate_adaptive_ray(
            reference, frequency, options.length_scale_cm);
        const double compressed_intensity = integrate_adaptive_ray(
            compressed, frequency, options.length_scale_cm);
        if (relative_difference(reference_intensity, compressed_intensity) >
            options.integration_tolerance) {
          compressed_is_accurate = false;
          break;
        }
      }
      const std::vector<AdaptiveSegment>& selected =
          compressed_is_accurate ? compressed : reference;
      double maximum_remaining_temperature = 0.0;
      for (const AdaptiveSegment& segment : selected) {
        packed_segments.push_back(
            {{segment.sample.temperature, segment.sample.emission_measure,
              segment.sample.ionic_charge,
              segment.upper - segment.lower,
              segment.sample.opaque ? 1.0 : 0.0,
              maximum_remaining_temperature}});
        if (!segment.sample.opaque &&
            segment.sample.temperature > maximum_remaining_temperature) {
          maximum_remaining_temperature = segment.sample.temperature;
        }
      }
    }
    host_offsets[pixel + 1] = packed_segments.size();
  }

  AdaptiveSegmentView device_segments("adaptive_ray_segments",
                                      packed_segments.size(),
                                      kAdaptiveSegmentFieldCount);
  auto host_segments = Kokkos::create_mirror_view(device_segments);
  for (std::size_t i = 0; i < packed_segments.size(); ++i) {
    for (std::size_t field = 0; field < kAdaptiveSegmentFieldCount; ++field) {
      host_segments(i, field) = packed_segments[i][field];
    }
  }
  Kokkos::deep_copy(device_segments, host_segments);

  RayOffsetView device_offsets("adaptive_ray_offsets", pixel_count + 1);
  auto offsets = Kokkos::create_mirror_view(device_offsets);
  for (std::size_t i = 0; i < host_offsets.size(); ++i) {
    offsets(i) = host_offsets[i];
  }
  Kokkos::deep_copy(device_offsets, offsets);
  return {device_segments, device_offsets};
}

Kokkos::View<double**> compute_image(const RaySampleView& samples,
                                     const ViewGeometry& view,
                                     const double frequency,
                                     const double length_scale_cm,
                                     const double relative_intensity_threshold) {
  Kokkos::View<double**> image("intensity_image", view.nx, view.ny);
  const double prefactor =
      kFreeFreeCoefficient / (frequency * frequency * frequency);
  const std::size_t image_ny = view.ny;
  const std::size_t nlos = view.nlos;
  const double step = view.ds * length_scale_cm;

  Kokkos::parallel_for(
      "free_free_ray_transfer",
      Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(
          0, view.nx * view.ny),
      KOKKOS_LAMBDA(const std::size_t pixel) {
        const std::size_t i = pixel / image_ny;
        const std::size_t j = pixel % image_ny;
        double intensity = 0.0;
        double foreground_optical_depth = 0.0;
        for (std::size_t sample = nlos; sample > 0; --sample) {
          const std::size_t k = sample - 1;
          if (samples(i, j, k, 2) > 0.5) {
            // Stop at the first opaque sample, retaining only foreground light.
            break;
          }
          const double temperature = samples(i, j, k, 0);
          const double emission_measure = samples(i, j, k, 1);
          const double ionic_charge = samples(i, j, k, 3);
          if (temperature > 0.0 && emission_measure > 0.0 &&
              ionic_charge > 0.0) {
            const double gaunt_factor =
                free_free_gaunt_factor(temperature, frequency, ionic_charge);
            const double x_nu =
                kPlanck * frequency / (kBoltzmann * temperature);
            const double stimulated_emission = one_minus_exp_negative(x_nu);
            const double alpha =
                prefactor * Kokkos::sqrt(1.0 / temperature) *
                emission_measure * stimulated_emission *
                gaunt_factor;
            const double tau = alpha * step;
            const double emitted_fraction = one_minus_exp_negative(tau);
            const double source =
                planck_source_function(frequency, temperature);
            intensity += Kokkos::exp(-foreground_optical_depth) * source *
                         emitted_fraction;
            foreground_optical_depth += tau;
          }
          const double remaining_temperature_bound =
              samples(i, j, k, 4);
          if (remaining_temperature_bound == 0.0) {
            break;
          }
          const double remaining_source_bound = planck_source_function(
              frequency, remaining_temperature_bound);
          const double omitted_intensity_bound =
              Kokkos::exp(-foreground_optical_depth) *
              remaining_source_bound;
          if (omitted_intensity_bound <
              relative_intensity_threshold * remaining_source_bound) {
            break;
          }
        }
        image(i, j) = intensity;
      });
  Kokkos::fence();
  return image;
}

Kokkos::View<double**> compute_adaptive_image(
    const AdaptiveRaySamples& samples, const ViewGeometry& view,
    const double frequency, const double length_scale_cm,
    const double relative_intensity_threshold) {
  Kokkos::View<double**> image("adaptive_intensity_image", view.nx, view.ny);
  const double prefactor =
      kFreeFreeCoefficient / (frequency * frequency * frequency);
  const std::size_t image_ny = view.ny;

  Kokkos::parallel_for(
      "adaptive_free_free_ray_transfer",
      Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(
          0, view.nx * view.ny),
      KOKKOS_LAMBDA(const std::size_t pixel) {
        const std::size_t i = pixel / image_ny;
        const std::size_t j = pixel % image_ny;
        const std::size_t begin = samples.offsets(pixel);
        const std::size_t end = samples.offsets(pixel + 1);
        double intensity = 0.0;
        double foreground_optical_depth = 0.0;
        for (std::size_t position = end; position > begin; --position) {
          const std::size_t segment = position - 1;
          if (samples.segments(segment, 4) > 0.5) {
            break;
          }
          const double temperature = samples.segments(segment, 0);
          const double emission_measure = samples.segments(segment, 1);
          const double ionic_charge = samples.segments(segment, 2);
          if (temperature > 0.0 && emission_measure > 0.0 &&
              ionic_charge > 0.0) {
            const double gaunt_factor =
                free_free_gaunt_factor(temperature, frequency, ionic_charge);
            const double x_nu =
                kPlanck * frequency / (kBoltzmann * temperature);
            const double alpha =
                prefactor * Kokkos::sqrt(1.0 / temperature) *
                emission_measure * one_minus_exp_negative(x_nu) *
                gaunt_factor;
            const double tau =
                alpha * samples.segments(segment, 3) * length_scale_cm;
            const double source =
                planck_source_function(frequency, temperature);
            intensity += Kokkos::exp(-foreground_optical_depth) * source *
                         one_minus_exp_negative(tau);
            foreground_optical_depth += tau;
          }
          const double remaining_temperature_bound =
              samples.segments(segment, 5);
          if (remaining_temperature_bound == 0.0) {
            break;
          }
          const double remaining_source_bound = planck_source_function(
              frequency, remaining_temperature_bound);
          if (Kokkos::exp(-foreground_optical_depth) *
                  remaining_source_bound <
              relative_intensity_threshold * remaining_source_bound) {
            break;
          }
        }
        image(i, j) = intensity;
      });
  Kokkos::fence();
  return image;
}

void write_flow_grid_metadata(std::ostream& out, const Grid& grid) {
  out << "# flow_resolution: " << grid.nx << ',' << grid.ny << ',' << grid.nz
      << '\n';
  out << "# flow_bounds_r0: ";
  for (std::size_t i = 0; i < grid.bounds.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    out << grid.bounds[i];
  }
  out << '\n';
}

void create_output_parent_directories(const std::string& path) {
  const std::filesystem::path parent =
      std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

double integrate_image(const Kokkos::View<double**>& image,
                       const ViewGeometry& view,
                       const double length_scale) {
  auto host_image = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), image);
  const double pixel_area_cm2 =
      view.du * view.dv * length_scale * length_scale;
  double projected_intensity_area = 0.0;
  for (std::size_t i = 0; i < view.nx; ++i) {
    for (std::size_t j = 0; j < view.ny; ++j) {
      projected_intensity_area += host_image(i, j) * pixel_area_cm2;
    }
  }
  return projected_intensity_area;
}

std::vector<double> spectrum_frequencies(const Options& options) {
  std::vector<double> frequencies(options.frequency_count);
  const double log_min = std::log(options.frequency_min);
  const double log_max = std::log(options.frequency_max);
  for (std::size_t i = 0; i < options.frequency_count; ++i) {
    const double fraction =
        static_cast<double>(i) / (options.frequency_count - 1);
    frequencies[i] = std::exp(log_min + fraction * (log_max - log_min));
  }
  frequencies.front() = options.frequency_min;
  frequencies.back() = options.frequency_max;
  return frequencies;
}

void write_spectrum(const std::string& path, const std::vector<double>& frequencies,
                    const std::vector<double>& projected_intensity_areas,
                    const ViewGeometry& view, const Grid& grid,
                    const Options& options) {
  create_output_parent_directories(path);
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot open output spectrum: " + path);
  }
  out << std::setprecision(17);
  out << "# projected intensity-area units: erg s^-1 Hz^-1\n";
  out << "# n0_cm^-3: " << options.number_density_scale << '\n';
  out << "# temperature0_k: " << options.temperature_scale << '\n';
  out << "# length0_cm: " << options.length_scale_cm << '\n';
  out << "# longitudinal_integration: "
      << (options.adaptive_longitudinal ? "adaptive" : "fixed") << '\n';
  if (options.adaptive_longitudinal) {
    out << "# integration_tolerance: " << options.integration_tolerance << '\n';
  }
  write_flow_grid_metadata(out, grid);
  out << "# line_of_sight: " << view.los.x << ',' << view.los.y << ','
      << view.los.z << '\n';
  out << "frequency_hz,projected_intensity_area\n";
  for (std::size_t i = 0; i < frequencies.size(); ++i) {
    out << frequencies[i] << ',' << projected_intensity_areas[i] << '\n';
  }
  if (!out) {
    throw std::runtime_error("failed while writing output spectrum: " + path);
  }
}

void run(const Options& options) {
  const Grid grid = read_grid(options.input);
  FieldView fields;
  if (!options.adaptive_longitudinal) {
    fields = FieldView("flow_fields", grid.nx, grid.ny, grid.nz, kFieldCount);
  }
  validate_and_copy_fields(
      grid, options.adaptive_longitudinal ? nullptr : &fields);
  const ViewGeometry view = make_view_geometry(grid, options);
  const std::vector<double> frequencies = spectrum_frequencies(options);
  std::vector<double> projected_intensity_areas;
  projected_intensity_areas.reserve(frequencies.size());
  if (options.adaptive_longitudinal) {
    const std::vector<double> reference_frequencies =
        adaptive_reference_frequencies(options);
    const AdaptiveRaySamples ray_samples = build_adaptive_ray_samples(
        grid, view, options, reference_frequencies);
    for (const double frequency : frequencies) {
      const auto image = compute_adaptive_image(
          ray_samples, view, frequency, options.length_scale_cm,
          options.relative_intensity_threshold);
      projected_intensity_areas.push_back(
          integrate_image(image, view, options.length_scale_cm));
    }
  } else {
    const RaySampleView ray_samples = precompute_ray_samples(
        grid, fields, view, options, grid.has_occultation);
    for (const double frequency : frequencies) {
      const auto image =
          compute_image(ray_samples, view, frequency, options.length_scale_cm,
                        options.relative_intensity_threshold);
      projected_intensity_areas.push_back(
          integrate_image(image, view, options.length_scale_cm));
    }
  }
  write_spectrum(options.output, frequencies, projected_intensity_areas, view,
                 grid, options);
  std::cout << "Wrote " << frequencies.size() << " logarithmically spaced "
            << "spectrum samples to " << options.output << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Kokkos::initialize(argc, argv);
    try {
      run(options);
    } catch (...) {
      Kokkos::finalize();
      throw;
    }
    Kokkos::finalize();
  } catch (const std::exception& error) {
    std::cerr << "rff-transfer: " << error.what() << '\n';
    usage(std::cerr);
    return 1;
  }
  return 0;
}
