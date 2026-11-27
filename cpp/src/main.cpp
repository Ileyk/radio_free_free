#include <Kokkos_Core.hpp>
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
  std::size_t image_nu = 128;
  std::size_t image_nv = 128;
  std::size_t ray_samples = 128;
  std::array<double, 4> image_bounds{};
  std::array<double, 2> ray_bounds{};
  double theta_deg = 0.0;
  double phi_deg = 0.0;
  bool adaptive_longitudinal = false;
  bool integration_tolerance_given = false;
  bool image_resolution_given = false;
  bool ray_samples_given = false;
  bool image_bounds_given = false;
  bool ray_bounds_given = false;
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

struct ImageGeometry {
  std::size_t nx;
  std::size_t ny;
  double u_min;
  double v_min;
  double u_max;
  double v_max;
  double du;
  double dv;
  Vector3 u;
  Vector3 v;
  Vector3 los;
};

struct OpaqueObject {
  enum class Shape { Sphere, Cylinder };
  Shape shape;
  Vector3 center;
  double radius;
  double height = 0.0;
  Vector3 axis{};
};

struct AnalyticalModel {
  std::string filename;
  std::string module_name;
  std::string coordinates;
  std::vector<OpaqueObject> opaque_objects;
  bool owns_interpreter = false;
  PyObject* module = nullptr;
  PyObject* density = nullptr;
  PyObject* temperature = nullptr;
  PyObject* electron_per_baryon = nullptr;
  PyObject* ion_charge2_per_baryon = nullptr;
  PyObject* numpy = nullptr;
  PyObject* numpy_float64 = nullptr;

  explicit AnalyticalModel(const std::string& path);
  ~AnalyticalModel();
  AnalyticalModel(const AnalyticalModel&) = delete;
  AnalyticalModel& operator=(const AnalyticalModel&) = delete;
  void evaluate(const std::vector<Vector3>& positions,
                std::vector<double>& density_values,
                std::vector<double>& temperature_values,
                std::vector<double>& electron_values,
                std::vector<double>& ion_charge2_values) const;
};

struct FlowSample {
  double temperature = 0.0;
  double emission_measure = 0.0;
  double ionic_charge = 0.0;
  bool opaque = false;
};

std::vector<double> image_intensity_scales(
    const Grid& grid, const ViewGeometry& view, const Options& options,
    const std::vector<double>& frequencies) {
  const double dx = (grid.bounds[1] - grid.bounds[0]) / grid.nx;
  const double dy = (grid.bounds[3] - grid.bounds[2]) / grid.ny;
  const double dz = (grid.bounds[5] - grid.bounds[4]) / grid.nz;
  const double cell_volume_cm3 =
      dx * dy * dz * options.length_scale_cm * options.length_scale_cm *
      options.length_scale_cm;
  const double image_area_cm2 =
      view.nx * view.du * view.ny * view.dv * options.length_scale_cm *
      options.length_scale_cm;
  const double density_scale_squared =
      options.number_density_scale * options.number_density_scale;
  std::vector<double> scales;
  scales.reserve(frequencies.size());
  for (const double frequency : frequencies) {
    double optically_thin_flux = 0.0;
    double maximum_emitting_temperature = 0.0;
    for (std::size_t cell = 0; cell < grid.nx * grid.ny * grid.nz; ++cell) {
      const std::size_t offset = cell * kFieldCount;
      const double density_ratio = grid.fields[offset];
      const double temperature =
          options.temperature_scale * grid.fields[offset + 1];
      const double electron_abundance = grid.fields[offset + 2];
      const double ion_charge2_abundance = grid.fields[offset + 3];
      if (density_ratio <= 0.0 || temperature <= 0.0 ||
          electron_abundance <= 0.0 || ion_charge2_abundance <= 0.0) {
        continue;
      }
      const double emission_measure =
          density_scale_squared * density_ratio * density_ratio *
          electron_abundance * ion_charge2_abundance;
      maximum_emitting_temperature =
          std::max(maximum_emitting_temperature, temperature);
      const double ionic_charge =
          ion_charge2_abundance / electron_abundance;
      const double x_nu =
          kPlanck * frequency / (kBoltzmann * temperature);
      const double alpha =
          kFreeFreeCoefficient / (frequency * frequency * frequency) *
          std::sqrt(1.0 / temperature) * emission_measure *
          one_minus_exp_negative(x_nu) *
          free_free_gaunt_factor(temperature, frequency, ionic_charge);
      optically_thin_flux +=
          alpha * planck_source_function(frequency, temperature) *
          cell_volume_cm3;
    }
    const double blackbody_upper_bound =
        maximum_emitting_temperature > 0.0
            ? planck_source_function(frequency, maximum_emitting_temperature)
            : 0.0;
    const double scale = std::min(optically_thin_flux / image_area_cm2,
                                  blackbody_upper_bound);
    scales.push_back(
        scale > 0.0 && std::isfinite(scale)
            ? scale
            : std::numeric_limits<double>::min());
  }
  return scales;
}

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

struct AdaptiveContext {
  double ray_intensity = 0.0;
  double incoming_intensity = 0.0;
  double foreground_transmission = 1.0;
  double pilot_interval_source = 0.0;
  double pilot_interval_transmission = 1.0;
};

struct AdaptiveRaySamples {
  AdaptiveSegmentView segments;
  RayOffsetView offsets;
};

struct OpaqueIntersectionDetected {
  double position;
};

void usage(std::ostream& out) {
  out << "Usage: rff-transfer --input model.py --output result.csv"
         " --n0 cm^-3 --temperature0 K --length0 R_sun\n"
         "  Spectrum: [--mode spectrum] --frequency-min-hz Hz"
         " --frequency-max-hz Hz [--frequency-count N]\n"
         "  Stopping: [--relative-intensity-threshold (0,1)]"
         " (alias --relative_intensity_threshold; default 1e-3)\n"
         "  Analytical Python input: [--image-resolution NU NV]"
         " --image-bounds UMIN UMAX VMIN VMAX"
         " --ray-bounds SMIN SMAX [--ray-samples NS]\n"
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

std::size_t parse_integer_at_least(const std::string& value,
                                   const std::string& name,
                                   const std::size_t minimum) {
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument(name + " must be an integer >= " +
                                std::to_string(minimum));
  }
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size() ||
      parsed > std::numeric_limits<std::size_t>::max() || parsed < minimum) {
    throw std::invalid_argument(name + " must be an integer >= " +
                                std::to_string(minimum));
  }
  return static_cast<std::size_t>(parsed);
}

std::size_t parse_count(const std::string& value) {
  return parse_integer_at_least(value, "--frequency-count", 2);
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
    if (key == "--image-resolution") {
      if (i + 2 >= argc) {
        throw std::invalid_argument(
            "--image-resolution requires NU and NV");
      }
      options.image_nu =
          parse_integer_at_least(argv[++i], "--image-resolution NU", 1);
      options.image_nv =
          parse_integer_at_least(argv[++i], "--image-resolution NV", 1);
      options.image_resolution_given = true;
      continue;
    }
    if (key == "--image-bounds") {
      if (i + 4 >= argc) {
        throw std::invalid_argument(
            "--image-bounds requires UMIN UMAX VMIN VMAX");
      }
      for (double& bound : options.image_bounds) {
        bound = parse_finite(argv[++i], "--image-bounds");
      }
      options.image_bounds_given = true;
      continue;
    }
    if (key == "--ray-bounds") {
      if (i + 2 >= argc) {
        throw std::invalid_argument("--ray-bounds requires SMIN and SMAX");
      }
      for (double& bound : options.ray_bounds) {
        bound = parse_finite(argv[++i], "--ray-bounds");
      }
      options.ray_bounds_given = true;
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
    } else if (key == "--ray-samples") {
      options.ray_samples =
          parse_integer_at_least(value, "--ray-samples", 1);
      options.ray_samples_given = true;
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
  if (std::filesystem::path(options.input).extension() != ".py") {
    throw std::invalid_argument(
        "rff-transfer accepts only analytical Python input files (.py); "
        ".rff input is not supported");
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
  if (options.image_nu >
      std::numeric_limits<std::size_t>::max() / options.image_nv) {
    throw std::invalid_argument("--image-resolution dimensions overflow");
  }
  if (options.image_bounds_given &&
      !(options.image_bounds[0] < options.image_bounds[1] &&
        options.image_bounds[2] < options.image_bounds[3])) {
    throw std::invalid_argument(
        "--image-bounds must contain increasing U and V ranges");
  }
  if (options.ray_bounds_given &&
      !(options.ray_bounds[0] < options.ray_bounds[1])) {
    throw std::invalid_argument("--ray-bounds requires SMIN less than SMAX");
  }
  if (options.adaptive_longitudinal) {
    throw std::invalid_argument(
        "--adaptive-longitudinal is not supported for analytical input");
  }
  if (!options.image_bounds_given || !options.ray_bounds_given) {
    throw std::invalid_argument(
        "analytical input requires --image-bounds and --ray-bounds");
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

class PyObjectRef {
 public:
  explicit PyObjectRef(PyObject* value = nullptr) : value_(value) {}
  ~PyObjectRef() { Py_XDECREF(value_); }
  PyObjectRef(const PyObjectRef&) = delete;
  PyObjectRef& operator=(const PyObjectRef&) = delete;
  PyObjectRef(PyObjectRef&& other) noexcept : value_(other.release()) {}
  PyObjectRef& operator=(PyObjectRef&& other) noexcept {
    if (this != &other) {
      Py_XDECREF(value_);
      value_ = other.release();
    }
    return *this;
  }
  PyObject* get() const { return value_; }
  PyObject* release() {
    PyObject* value = value_;
    value_ = nullptr;
    return value;
  }

 private:
  PyObject* value_;
};

std::string python_error_message() {
  if (!PyErr_Occurred()) {
    return "unknown Python error";
  }
  PyObject* type = nullptr;
  PyObject* value = nullptr;
  PyObject* traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  PyObjectRef type_ref(type);
  PyObjectRef value_ref(value);
  PyObjectRef traceback_ref(traceback);
  PyObjectRef text(value != nullptr ? PyObject_Str(value) : nullptr);
  const char* utf8 =
      text.get() != nullptr ? PyUnicode_AsUTF8(text.get()) : nullptr;
  const std::string message =
      utf8 != nullptr ? utf8 : "failed to format Python exception";
  if (utf8 == nullptr) {
    PyErr_Clear();
  }
  return message;
}

[[noreturn]] void throw_python_error(const std::string& context) {
  throw std::runtime_error(context + ": " + python_error_message());
}

PyObjectRef python_attribute(PyObject* object, const char* name,
                             const std::string& context) {
  PyObjectRef value(PyObject_GetAttrString(object, name));
  if (value.get() == nullptr) {
    throw_python_error(context);
  }
  return value;
}

std::vector<double> python_number_sequence(PyObject* object,
                                           const std::size_t expected_size,
                                           const std::string& description) {
  PyObjectRef sequence(PySequence_Fast(
      object, (description + " must be a sequence").c_str()));
  if (sequence.get() == nullptr) {
    throw_python_error(description);
  }
  const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.get());
  if (size != static_cast<Py_ssize_t>(expected_size)) {
    throw std::runtime_error(description + " must contain " +
                             std::to_string(expected_size) + " values");
  }
  std::vector<double> values;
  values.reserve(expected_size);
  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (std::size_t i = 0; i < expected_size; ++i) {
    const double value = PyFloat_AsDouble(items[i]);
    if (PyErr_Occurred()) {
      throw_python_error(description + " values must be numeric");
    }
    if (!std::isfinite(value)) {
      throw std::runtime_error(description + " values must be finite");
    }
    values.push_back(value);
  }
  return values;
}

double python_dictionary_number(PyObject* dictionary, const char* key,
                                const std::string& description) {
  PyObject* value = PyDict_GetItemString(dictionary, key);
  if (value == nullptr) {
    throw std::runtime_error(description + " is missing '" + key + "'");
  }
  const double number = PyFloat_AsDouble(value);
  if (PyErr_Occurred()) {
    throw_python_error(description + " field '" + key + "' must be numeric");
  }
  if (!std::isfinite(number)) {
    throw std::runtime_error(description + " field '" + key +
                             "' must be finite");
  }
  return number;
}

std::vector<OpaqueObject> parse_opaque_objects(PyObject* objects) {
  PyObjectRef sequence(
      PySequence_Fast(objects, "opaque_objects must be a sequence"));
  if (sequence.get() == nullptr) {
    throw_python_error("opaque_objects");
  }
  std::vector<OpaqueObject> parsed;
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence.get());
  parsed.reserve(static_cast<std::size_t>(count));
  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (Py_ssize_t i = 0; i < count; ++i) {
    PyObject* item = items[i];
    if (!PyDict_Check(item)) {
      throw std::runtime_error("each opaque_objects entry must be a dictionary");
    }
    const std::string description =
        "opaque_objects[" + std::to_string(i) + "]";
    PyObject* shape_value = PyDict_GetItemString(item, "shape");
    if (shape_value == nullptr || !PyUnicode_Check(shape_value)) {
      throw std::runtime_error(description + " requires a string 'shape'");
    }
    const char* shape_text = PyUnicode_AsUTF8(shape_value);
    if (shape_text == nullptr) {
      throw_python_error(description + " shape");
    }
    PyObject* center_value = PyDict_GetItemString(item, "center");
    if (center_value == nullptr) {
      throw std::runtime_error(description + " is missing 'center'");
    }
    const std::vector<double> center =
        python_number_sequence(center_value, 3, description + " center");
    OpaqueObject object{};
    object.center = {center[0], center[1], center[2]};
    object.radius = python_dictionary_number(item, "radius", description);
    if (!(object.radius > 0.0)) {
      throw std::runtime_error(description + " radius must be positive");
    }
    if (std::string(shape_text) == "sphere") {
      object.shape = OpaqueObject::Shape::Sphere;
    } else if (std::string(shape_text) == "cylinder") {
      object.shape = OpaqueObject::Shape::Cylinder;
      object.height =
          python_dictionary_number(item, "height", description);
      if (!(object.height > 0.0)) {
        throw std::runtime_error(description + " height must be positive");
      }
      PyObject* axis_value = PyDict_GetItemString(item, "axis");
      if (axis_value == nullptr) {
        throw std::runtime_error(description + " is missing 'axis'");
      }
      const std::vector<double> axis =
          python_number_sequence(axis_value, 3, description + " axis");
      object.axis = normalized({axis[0], axis[1], axis[2]});
    } else {
      throw std::runtime_error(
          description + " shape must be 'sphere' or 'cylinder'");
    }

    parsed.push_back(object);
  }
  return parsed;
}

PyObjectRef numpy_array_from_vector(PyObject* numpy, PyObject* float64,
                                    const std::vector<double>& values) {
  if (values.size() >
      static_cast<std::size_t>(PY_SSIZE_T_MAX) / sizeof(double)) {
    throw std::runtime_error("analytical ray batch is too large for Python");
  }
  const Py_ssize_t byte_count =
      static_cast<Py_ssize_t>(values.size() * sizeof(double));
  const char* data = values.empty()
                         ? ""
                         : reinterpret_cast<const char*>(values.data());
  PyObjectRef bytes(PyBytes_FromStringAndSize(data, byte_count));
  if (bytes.get() == nullptr) {
    throw_python_error("creating analytical coordinate array");
  }
  PyObjectRef frombuffer(PyObject_GetAttrString(numpy, "frombuffer"));
  if (frombuffer.get() == nullptr) {
    throw_python_error("loading numpy.frombuffer");
  }
  PyObjectRef array(PyObject_CallFunctionObjArgs(
      frombuffer.get(), bytes.get(), float64, nullptr));
  if (array.get() == nullptr) {
    throw_python_error("creating analytical NumPy array");
  }
  return array;
}

void evaluate_python_field(PyObject* numpy, PyObject* float64,
                           PyObject* function, PyObject* arg1, PyObject* arg2,
                           PyObject* arg3, const std::size_t count,
                           const std::string& field_name,
                           std::vector<double>& output) {
  PyObjectRef value(PyObject_CallFunctionObjArgs(function, arg1, arg2, arg3,
                                                 nullptr));
  if (value.get() == nullptr) {
    throw_python_error("evaluating " + field_name + "()");
  }
  PyObjectRef shape(PyTuple_New(1));
  if (shape.get() == nullptr) {
    throw_python_error("constructing analytical output shape");
  }
  PyObject* dimension = PyLong_FromSize_t(count);
  if (dimension == nullptr) {
    throw_python_error("constructing analytical output shape");
  }
  PyTuple_SET_ITEM(shape.get(), 0, dimension);

  PyObjectRef broadcast_to(
      python_attribute(numpy, "broadcast_to", "loading numpy.broadcast_to"));
  PyObjectRef broadcast(PyObject_CallFunctionObjArgs(
      broadcast_to.get(), value.get(), shape.get(), nullptr));
  if (broadcast.get() == nullptr) {
    throw_python_error(field_name + "() result is not broadcastable to the "
                                   "ray batch shape");
  }
  PyObjectRef asarray(
      python_attribute(numpy, "asarray", "loading numpy.asarray"));
  PyObjectRef kwargs(PyDict_New());
  if (kwargs.get() == nullptr ||
      PyDict_SetItemString(kwargs.get(), "dtype", float64) != 0) {
    throw_python_error("converting " + field_name + "() result");
  }
  PyObjectRef arguments(PyTuple_Pack(1, broadcast.get()));
  if (arguments.get() == nullptr) {
    throw_python_error("converting " + field_name + "() result");
  }
  PyObjectRef array(PyObject_Call(asarray.get(), arguments.get(), kwargs.get()));
  if (array.get() == nullptr) {
    throw_python_error("converting " + field_name + "() result to float64");
  }
  PyObjectRef contiguous_function(python_attribute(
      numpy, "ascontiguousarray", "loading numpy.ascontiguousarray"));
  PyObjectRef contiguous(PyObject_CallFunctionObjArgs(
      contiguous_function.get(), array.get(), nullptr));
  if (contiguous.get() == nullptr) {
    throw_python_error("making " + field_name + "() result contiguous");
  }

  Py_buffer buffer{};
  if (PyObject_GetBuffer(contiguous.get(), &buffer, PyBUF_CONTIG_RO) != 0) {
    throw_python_error("reading " + field_name + "() result");
  }
  const std::size_t expected_bytes = count * sizeof(double);
  if (buffer.len < 0 ||
      static_cast<std::size_t>(buffer.len) != expected_bytes) {
    PyBuffer_Release(&buffer);
    throw std::runtime_error(field_name +
                             "() returned an incompatible array shape");
  }
  output.resize(count);
  if (expected_bytes > 0) {
    std::memcpy(output.data(), buffer.buf, expected_bytes);
  }
  PyBuffer_Release(&buffer);
}

AnalyticalModel::AnalyticalModel(const std::string& path)
    : filename(std::filesystem::absolute(path).lexically_normal().string()),
      module_name("_rff_transfer_analytical_model") {
  if (!std::filesystem::is_regular_file(filename)) {
    throw std::runtime_error("cannot open analytical input file: " + filename);
  }
  if (!Py_IsInitialized()) {
    Py_Initialize();
    owns_interpreter = true;
  }
  try {
    PyObjectRef import_util(PyImport_ImportModule("importlib.util"));
    if (import_util.get() == nullptr) {
      throw_python_error("importing importlib.util");
    }
    PyObjectRef spec_from_file_location(python_attribute(
        import_util.get(), "spec_from_file_location",
        "loading importlib.util.spec_from_file_location"));
    PyObjectRef module_from_spec(python_attribute(
        import_util.get(), "module_from_spec",
        "loading importlib.util.module_from_spec"));
    PyObjectRef module_name_object(
        PyUnicode_FromString(module_name.c_str()));
    PyObjectRef filename_object(PyUnicode_FromString(filename.c_str()));
    PyObjectRef spec(PyObject_CallFunctionObjArgs(
        spec_from_file_location.get(), module_name_object.get(),
        filename_object.get(), nullptr));
    if (spec.get() == nullptr) {
      throw_python_error("loading analytical input file");
    }
    PyObjectRef loaded_module(
        PyObject_CallFunctionObjArgs(module_from_spec.get(), spec.get(),
                                     nullptr));
    if (loaded_module.get() == nullptr) {
      throw_python_error("creating analytical input module");
    }
    PyObjectRef system(PyImport_ImportModule("sys"));
    PyObjectRef modules(PyObject_GetAttrString(system.get(), "modules"));
    if (modules.get() == nullptr ||
        PyDict_SetItemString(modules.get(), module_name.c_str(),
                             loaded_module.get()) != 0) {
      throw_python_error("registering analytical input module");
    }
    PyObjectRef system_path(PyObject_GetAttrString(system.get(), "path"));
    PyObjectRef parent_path(PyUnicode_FromString(
        std::filesystem::path(filename).parent_path().string().c_str()));
    if (PyList_Insert(system_path.get(), 0, parent_path.get()) != 0) {
      throw_python_error("adding analytical input directory to sys.path");
    }
    PyObjectRef loader(PyObject_GetAttrString(spec.get(), "loader"));
    PyObjectRef execute(PyObject_GetAttrString(loader.get(), "exec_module"));
    PyObjectRef executed(PyObject_CallFunctionObjArgs(
        execute.get(), loaded_module.get(), nullptr));
    if (executed.get() == nullptr) {
      throw_python_error("executing analytical input file");
    }

    PyObjectRef coordinates_object(
        PyObject_GetAttrString(loaded_module.get(), "coordinates"));
    if (coordinates_object.get() == nullptr) {
      throw_python_error("analytical input must define 'coordinates'");
    }
    if (!PyUnicode_Check(coordinates_object.get())) {
      throw std::runtime_error(
          "analytical input must define coordinates as a string");
    }
    const char* coordinate_name =
        PyUnicode_AsUTF8(coordinates_object.get());
    if (coordinate_name == nullptr) {
      throw_python_error("reading analytical coordinates");
    }
    coordinates = coordinate_name;
    if (coordinates != "spherical" && coordinates != "cylindrical" &&
        coordinates != "cartesian") {
      throw std::runtime_error(
          "coordinates must be 'spherical', 'cylindrical', or 'cartesian'");
    }

    PyObjectRef objects_object(
        PyObject_GetAttrString(loaded_module.get(), "opaque_objects"));
    if (objects_object.get() == nullptr) {
      if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        objects_object = PyObjectRef(PyList_New(0));
      } else {
        throw_python_error("reading analytical opaque_objects");
      }
    }
    if (objects_object.get() == nullptr) {
      throw_python_error("creating an empty opaque_objects list");
    }
    if (objects_object.get() != Py_None) {
      opaque_objects = parse_opaque_objects(objects_object.get());
    }

    PyObjectRef density_object(
        PyObject_GetAttrString(loaded_module.get(), "density"));
    PyObjectRef temperature_object(
        PyObject_GetAttrString(loaded_module.get(), "temperature"));
    PyObjectRef electron_object(
        PyObject_GetAttrString(loaded_module.get(), "electron_per_baryon"));
    PyObjectRef ion_charge2_object(
        PyObject_GetAttrString(loaded_module.get(),
                               "ion_charge2_per_baryon"));
    const std::array<PyObject*, 4> functions{{
        density_object.get(), temperature_object.get(), electron_object.get(),
        ion_charge2_object.get()}};
    const std::array<const char*, 4> function_names{{
        "density", "temperature", "electron_per_baryon",
        "ion_charge2_per_baryon"}};
    for (std::size_t i = 0; i < functions.size(); ++i) {
      if (functions[i] == nullptr) {
        throw_python_error(std::string("analytical input must define ") +
                           function_names[i] + "()");
      }
      if (!PyCallable_Check(functions[i])) {
        throw std::runtime_error(std::string("analytical ") +
                                 function_names[i] + " must be callable");
      }
    }

    PyObjectRef numpy_module(PyImport_ImportModule("numpy"));
    if (numpy_module.get() == nullptr) {
      throw_python_error(
          "importing NumPy; install NumPy for the embedded Python interpreter");
    }
    PyObjectRef float64_object(
        PyObject_GetAttrString(numpy_module.get(), "float64"));
    if (float64_object.get() == nullptr) {
      throw_python_error("reading numpy.float64");
    }

    module = loaded_module.release();
    density = density_object.release();
    temperature = temperature_object.release();
    electron_per_baryon = electron_object.release();
    ion_charge2_per_baryon = ion_charge2_object.release();
    numpy = numpy_module.release();
    numpy_float64 = float64_object.release();
  } catch (...) {
    if (owns_interpreter && Py_IsInitialized()) {
      PyObject* system = PyImport_ImportModule("sys");
      if (system != nullptr) {
        PyObject* modules = PyObject_GetAttrString(system, "modules");
        if (modules != nullptr) {
          if (PyDict_DelItemString(modules, module_name.c_str()) != 0) {
            PyErr_Clear();
          }
          Py_DECREF(modules);
        } else {
          PyErr_Clear();
        }
        Py_DECREF(system);
      } else {
        PyErr_Clear();
      }
      Py_FinalizeEx();
      owns_interpreter = false;
    }
    throw;
  }
}

AnalyticalModel::~AnalyticalModel() {
  if (Py_IsInitialized()) {
    if (module != nullptr) {
      PyObject* system = PyImport_ImportModule("sys");
      if (system != nullptr) {
        PyObject* modules = PyObject_GetAttrString(system, "modules");
        if (modules != nullptr) {
          if (PyDict_DelItemString(modules, module_name.c_str()) != 0) {
            PyErr_Clear();
          }
          Py_DECREF(modules);
        } else {
          PyErr_Clear();
        }
        Py_DECREF(system);
      } else {
        PyErr_Clear();
      }
    }
    Py_XDECREF(density);
    Py_XDECREF(temperature);
    Py_XDECREF(electron_per_baryon);
    Py_XDECREF(ion_charge2_per_baryon);
    Py_XDECREF(numpy_float64);
    Py_XDECREF(numpy);
    Py_XDECREF(module);
    if (owns_interpreter) {
      Py_FinalizeEx();
    }
  }
}

void AnalyticalModel::evaluate(
    const std::vector<Vector3>& positions,
    std::vector<double>& density_values,
    std::vector<double>& temperature_values,
    std::vector<double>& electron_values,
    std::vector<double>& ion_charge2_values) const {
  std::array<std::vector<double>, 3> coordinates_values;
  for (auto& values : coordinates_values) {
    values.reserve(positions.size());
  }
  for (const Vector3& position : positions) {
    if (coordinates == "cartesian") {
      coordinates_values[0].push_back(position.x);
      coordinates_values[1].push_back(position.y);
      coordinates_values[2].push_back(position.z);
    } else {
      const double cylindrical_radius = std::hypot(position.x, position.y);
      double azimuth = std::atan2(position.y, position.x);
      if (azimuth < 0.0) {
        azimuth += 2.0 * kPi;
      }
      if (coordinates == "cylindrical") {
        coordinates_values[0].push_back(cylindrical_radius);
        coordinates_values[1].push_back(azimuth);
        coordinates_values[2].push_back(position.z);
      } else {
        coordinates_values[0].push_back(
            std::hypot(cylindrical_radius, position.z));
        coordinates_values[1].push_back(
            std::atan2(cylindrical_radius, position.z));
        coordinates_values[2].push_back(azimuth);
      }
    }
  }
  PyObjectRef arg1(numpy_array_from_vector(
      numpy, numpy_float64, coordinates_values[0]));
  PyObjectRef arg2(numpy_array_from_vector(
      numpy, numpy_float64, coordinates_values[1]));
  PyObjectRef arg3(numpy_array_from_vector(
      numpy, numpy_float64, coordinates_values[2]));
  evaluate_python_field(numpy, numpy_float64, density, arg1.get(), arg2.get(),
                        arg3.get(), positions.size(), "density",
                        density_values);
  evaluate_python_field(numpy, numpy_float64, temperature, arg1.get(),
                        arg2.get(), arg3.get(), positions.size(),
                        "temperature", temperature_values);
  evaluate_python_field(numpy, numpy_float64, electron_per_baryon, arg1.get(),
                        arg2.get(), arg3.get(), positions.size(),
                        "electron_per_baryon", electron_values);
  evaluate_python_field(numpy, numpy_float64, ion_charge2_per_baryon,
                        arg1.get(), arg2.get(), arg3.get(), positions.size(),
                        "ion_charge2_per_baryon", ion_charge2_values);
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

ImageGeometry make_image_geometry(const Options& options) {
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
  const double u_min = options.image_bounds[0];
  const double u_max = options.image_bounds[1];
  const double v_min = options.image_bounds[2];
  const double v_max = options.image_bounds[3];
  return {options.image_nu, options.image_nv, u_min, v_min, u_max, v_max,
          (u_max - u_min) / options.image_nu,
          (v_max - v_min) / options.image_nv, u, v, los};
}

double opaque_object_extent(const OpaqueObject& object,
                            const Vector3& direction) {
  if (object.shape == OpaqueObject::Shape::Sphere) {
    return object.radius;
  }
  const double axis_projection = dot(object.axis, direction);
  return 0.5 * object.height * std::abs(axis_projection) +
         object.radius *
             std::sqrt(std::max(0.0, 1.0 - axis_projection * axis_projection));
}

void validate_opaque_objects(const std::vector<OpaqueObject>& objects,
                             const ImageGeometry& image,
                             const std::array<double, 2>& ray_bounds) {
  for (std::size_t i = 0; i < objects.size(); ++i) {
    const OpaqueObject& object = objects[i];
    const auto validate_axis = [&](const Vector3& direction,
                                   const double lower_bound,
                                   const double upper_bound,
                                   const char* axis_name) {
      const double center_projection = dot(object.center, direction);
      const double extent = opaque_object_extent(object, direction);
      const double tolerance =
          1.0e-12 *
          std::max({1.0, std::abs(lower_bound), std::abs(upper_bound),
                    std::abs(center_projection) + extent});
      if (center_projection - extent < lower_bound - tolerance ||
          center_projection + extent > upper_bound + tolerance) {
        throw std::runtime_error(
            "opaque_objects[" + std::to_string(i) +
            "] must lie entirely inside the observer domain on " + axis_name);
      }
    };
    validate_axis(image.u, image.u_min, image.u_max, "u");
    validate_axis(image.v, image.v_min, image.v_max, "v");
    validate_axis(image.los, ray_bounds[0], ray_bounds[1], "s");
  }
}

bool ray_object_interval(const Vector3& origin, const Vector3& direction,
                         const OpaqueObject& object, double& s_min,
                         double& s_max) {
  const Vector3 offset{origin.x - object.center.x,
                       origin.y - object.center.y,
                       origin.z - object.center.z};
  if (object.shape == OpaqueObject::Shape::Sphere) {
    const double linear = dot(offset, direction);
    const double constant = dot(offset, offset) - object.radius * object.radius;
    const double discriminant = linear * linear - constant;
    if (discriminant < 0.0) {
      return false;
    }
    const double root = std::sqrt(std::max(0.0, discriminant));
    s_min = -linear - root;
    s_max = -linear + root;
    return true;
  }

  const double axial_origin = dot(offset, object.axis);
  const double axial_direction = dot(direction, object.axis);
  const Vector3 radial_origin{
      offset.x - axial_origin * object.axis.x,
      offset.y - axial_origin * object.axis.y,
      offset.z - axial_origin * object.axis.z};
  const double radial_direction_projection =
      dot(direction, object.axis);
  const Vector3 radial_direction{
      direction.x - radial_direction_projection * object.axis.x,
      direction.y - radial_direction_projection * object.axis.y,
      direction.z - radial_direction_projection * object.axis.z};
  const double radial_a = dot(radial_direction, radial_direction);
  const double radial_b = 2.0 * dot(radial_origin, radial_direction);
  const double radial_c =
      dot(radial_origin, radial_origin) - object.radius * object.radius;
  double radial_min = -std::numeric_limits<double>::infinity();
  double radial_max = std::numeric_limits<double>::infinity();
  if (radial_a < 1.0e-28) {
    if (radial_c > 0.0) {
      return false;
    }
  } else {
    const double discriminant =
        radial_b * radial_b - 4.0 * radial_a * radial_c;
    if (discriminant < 0.0) {
      return false;
    }
    const double root = std::sqrt(std::max(0.0, discriminant));
    radial_min = (-radial_b - root) / (2.0 * radial_a);
    radial_max = (-radial_b + root) / (2.0 * radial_a);
  }

  double cap_min = -std::numeric_limits<double>::infinity();
  double cap_max = std::numeric_limits<double>::infinity();
  const double half_height = object.height * 0.5;
  if (std::abs(axial_direction) < 1.0e-14) {
    if (std::abs(axial_origin) > half_height) {
      return false;
    }
  } else {
    cap_min = (-half_height - axial_origin) / axial_direction;
    cap_max = (half_height - axial_origin) / axial_direction;
    if (cap_min > cap_max) {
      std::swap(cap_min, cap_max);
    }
  }
  s_min = std::max(radial_min, cap_min);
  s_max = std::min(radial_max, cap_max);
  return s_min <= s_max;
}

bool visible_ray_interval(const Vector3& origin, const Vector3& direction,
                          const std::vector<OpaqueObject>& objects,
                          const double domain_min, const double domain_max,
                          double& visible_min) {
  visible_min = domain_min;
  const double boundary_epsilon =
      1.0e-12 * std::max(1.0, std::abs(domain_max));
  for (const OpaqueObject& object : objects) {
    double object_min = 0.0;
    double object_max = 0.0;
    if (!ray_object_interval(origin, direction, object, object_min,
                             object_max) ||
        object_max < domain_min || object_min > domain_max) {
      continue;
    }
    if (object_min <= domain_max + boundary_epsilon &&
        object_max >= domain_max - boundary_epsilon) {
      return false;
    }
    if (object_max > domain_min) {
      visible_min = std::max(visible_min, object_max);
    }
  }
  return visible_min < domain_max;
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

template <typename Sampler>
bool clip_to_nearest_opaque_boundary(const double clipped_min,
                                     const double clipped_max,
                                     const double base_step,
                                     Sampler&& sample_at,
                                     double& visible_min) {
  const std::size_t scan_count =
      samples_for_span(clipped_max - clipped_min, base_step * 0.5);
  const double scan_step = (clipped_max - clipped_min) / scan_count;
  double transparent_s = clipped_max;
  if (sample_at(transparent_s).opaque) {
    return false;
  }
  for (std::size_t i = 1; i <= scan_count; ++i) {
    const double sample_s =
        i == scan_count ? clipped_min : clipped_max - i * scan_step;
    if (sample_at(sample_s).opaque) {
      double opaque_s = sample_s;
      double clear_s = transparent_s;
      for (int iteration = 0; iteration < 64; ++iteration) {
        const double middle = opaque_s + (clear_s - opaque_s) * 0.5;
        if (!(middle > opaque_s && middle < clear_s)) {
          break;
        }
        if (sample_at(middle).opaque) {
          opaque_s = middle;
        } else {
          clear_s = middle;
        }
      }
      visible_min = clear_s;
      return visible_min < clipped_max;
    }
    transparent_s = sample_s;
  }
  visible_min = clipped_min;
  return true;
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

std::vector<std::vector<AdaptiveContext>> build_pilot_contexts(
    const std::vector<FlowSample>& samples,
    const std::vector<double>& frequencies, const double interval_length_cm,
    const std::vector<double>& image_scales) {
  std::vector<std::vector<AdaptiveContext>> contexts(
      samples.size(), std::vector<AdaptiveContext>(frequencies.size()));
  for (std::size_t frequency_index = 0;
       frequency_index < frequencies.size(); ++frequency_index) {
    const double frequency = frequencies[frequency_index];
    std::vector<SegmentOperator> operations;
    operations.reserve(samples.size());
    double pilot_intensity = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
      contexts[i][frequency_index].incoming_intensity = pilot_intensity;
      const SegmentOperator operation =
          segment_operator(samples[i], frequency, interval_length_cm);
      operations.push_back(operation);
      contexts[i][frequency_index].pilot_interval_source =
          operation.source_contribution;
      contexts[i][frequency_index].pilot_interval_transmission =
          operation.transmission;
      pilot_intensity =
          operation.transmission * pilot_intensity +
          operation.source_contribution;
    }
    double foreground_transmission = 1.0;
    for (std::size_t i = samples.size(); i > 0; --i) {
      AdaptiveContext& context = contexts[i - 1][frequency_index];
      context.ray_intensity =
          std::max(pilot_intensity, image_scales[frequency_index]);
      context.foreground_transmission = foreground_transmission;
      foreground_transmission *= operations[i - 1].transmission;
    }
  }
  return contexts;
}

std::vector<std::vector<AdaptiveContext>> build_segment_pilot_contexts(
    const std::vector<AdaptiveSegment>& segments,
    const std::vector<double>& frequencies, const double length_scale_cm,
    const std::vector<double>& image_scales) {
  std::vector<std::vector<AdaptiveContext>> contexts(
      segments.size(), std::vector<AdaptiveContext>(frequencies.size()));
  for (std::size_t frequency_index = 0;
       frequency_index < frequencies.size(); ++frequency_index) {
    const double frequency = frequencies[frequency_index];
    std::vector<SegmentOperator> operations;
    operations.reserve(segments.size());
    double pilot_intensity = 0.0;
    for (std::size_t i = 0; i < segments.size(); ++i) {
      contexts[i][frequency_index].incoming_intensity = pilot_intensity;
      const SegmentOperator operation = segment_operator(
          segments[i].sample, frequency,
          (segments[i].upper - segments[i].lower) * length_scale_cm);
      operations.push_back(operation);
      contexts[i][frequency_index].pilot_interval_source =
          operation.source_contribution;
      contexts[i][frequency_index].pilot_interval_transmission =
          operation.transmission;
      pilot_intensity =
          operation.transmission * pilot_intensity +
          operation.source_contribution;
    }
    double foreground_transmission = 1.0;
    for (std::size_t i = segments.size(); i > 0; --i) {
      AdaptiveContext& context = contexts[i - 1][frequency_index];
      context.ray_intensity =
          std::max(pilot_intensity, image_scales[frequency_index]);
      context.foreground_transmission = foreground_transmission;
      foreground_transmission *= operations[i - 1].transmission;
    }
  }
  return contexts;
}

bool refinement_criteria_pass(
    const FlowSample& center, const FlowSample& left,
    const FlowSample& right, const double interval_length,
    const double total_ray_length_cm, const double length_scale_cm,
    const std::vector<double>& frequencies,
    const std::vector<AdaptiveContext>& contexts, const double tolerance,
    std::vector<SegmentOperator>& left_operations,
    std::vector<SegmentOperator>& right_operations,
    bool& pilot_missed_emission) {
  bool passes = true;
  left_operations.clear();
  right_operations.clear();
  left_operations.reserve(frequencies.size());
  right_operations.reserve(frequencies.size());
  const double half_length_cm = interval_length * length_scale_cm * 0.5;
  const double interval_fraction =
      interval_length * length_scale_cm / total_ray_length_cm;
  for (std::size_t i = 0; i < frequencies.size(); ++i) {
    const double frequency = frequencies[i];
    const SegmentOperator coarse =
        segment_operator(center, frequency,
                         interval_length * length_scale_cm);
    const SegmentOperator left_half =
        segment_operator(left, frequency, half_length_cm);
    const SegmentOperator right_half =
        segment_operator(right, frequency, half_length_cm);
    left_operations.push_back(left_half);
    right_operations.push_back(right_half);

    const double fine_transmission =
        right_half.transmission * left_half.transmission;
    const double fine_source =
        right_half.transmission * left_half.source_contribution +
        right_half.source_contribution;
    const AdaptiveContext& context = contexts[i];
    if (context.pilot_interval_source == 0.0 &&
        (coarse.source_contribution > 0.0 || fine_source > 0.0)) {
      pilot_missed_emission = true;
      return false;
    }
    const double local_error =
        context.foreground_transmission *
        (std::abs(coarse.source_contribution - fine_source) +
         std::abs(coarse.transmission - fine_transmission) *
             std::abs(context.incoming_intensity));
    const double local_budget =
        tolerance * context.ray_intensity * interval_fraction;
    if (local_error > local_budget) {
      passes = false;
    }
  }
  return passes;
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
    const std::vector<double>& image_scales, const double tolerance,
    const std::size_t pilot_subdivisions, bool& pilot_missed_emission,
    Sampler&& sample_at) {
  std::vector<AdaptiveSegment> segments;
  const std::size_t interval_count = samples_for_span(s_max - s_min, base_step);
  const double interval_step = (s_max - s_min) / interval_count;
  const double ray_length_cm = (s_max - s_min) * length_scale_cm;
  if (interval_count >
      kMaximumAdaptiveSegmentsPerRay / pilot_subdivisions) {
    throw std::runtime_error(
        "adaptive pilot integration exceeds the segment limit");
  }
  const std::size_t pilot_count = interval_count * pilot_subdivisions;
  const double pilot_step = (s_max - s_min) / pilot_count;
  std::vector<FlowSample> pilot_samples;
  pilot_samples.reserve(pilot_count);
  double nearest_opaque = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < pilot_count; ++i) {
    const double midpoint = s_min + (i + 0.5) * pilot_step;
    const FlowSample sample = sample_at(midpoint);
    if (sample.opaque) {
      nearest_opaque = midpoint;
    }
    pilot_samples.push_back(sample);
  }
  if (std::isfinite(nearest_opaque)) {
    throw OpaqueIntersectionDetected{nearest_opaque};
  }
  const std::vector<std::vector<AdaptiveContext>> fine_pilot_contexts =
      build_pilot_contexts(pilot_samples, frequencies,
                           pilot_step * length_scale_cm, image_scales);
  std::vector<std::vector<AdaptiveContext>> pilot_contexts(
      interval_count, std::vector<AdaptiveContext>(frequencies.size()));
  for (std::size_t interval = 0; interval < interval_count; ++interval) {
    const std::size_t first = interval * pilot_subdivisions;
    const std::size_t last = first + pilot_subdivisions - 1;
    for (std::size_t frequency_index = 0;
         frequency_index < frequencies.size(); ++frequency_index) {
      AdaptiveContext context = fine_pilot_contexts[first][frequency_index];
      context.foreground_transmission =
          fine_pilot_contexts[last][frequency_index].foreground_transmission;
      double interval_source = 0.0;
      double interval_transmission = 1.0;
      for (std::size_t sample = first; sample <= last; ++sample) {
        const AdaptiveContext& sample_context =
            fine_pilot_contexts[sample][frequency_index];
        interval_source =
            sample_context.pilot_interval_transmission * interval_source +
            sample_context.pilot_interval_source;
        interval_transmission *=
            sample_context.pilot_interval_transmission;
      }
      context.pilot_interval_source = interval_source;
      context.pilot_interval_transmission = interval_transmission;
      pilot_contexts[interval][frequency_index] = context;
    }
  }

  auto append_segment = [&](const double lower, const double upper,
                            const FlowSample& sample) {
    if (segments.size() >= kMaximumAdaptiveSegmentsPerRay) {
      throw std::runtime_error(
          "adaptive longitudinal ray exceeds the segment limit");
    }
    segments.push_back({lower, upper, sample});
  };
  auto refine = [&](auto&& self, const double lower, const double upper,
                    const int depth,
                    const std::vector<AdaptiveContext>& contexts) -> void {
    if (pilot_missed_emission) {
      return;
    }
    const double middle = lower + (upper - lower) * 0.5;
    const FlowSample center_sample = sample_at(middle);
    const double left_middle = lower + (middle - lower) * 0.5;
    const double right_middle = middle + (upper - middle) * 0.5;
    const FlowSample left_sample = sample_at(left_middle);
    const FlowSample right_sample = sample_at(right_middle);
    if (center_sample.opaque || left_sample.opaque || right_sample.opaque) {
      double nearest_opaque = -std::numeric_limits<double>::infinity();
      if (center_sample.opaque) {
        nearest_opaque = middle;
      }
      if (left_sample.opaque) {
        nearest_opaque = std::max(nearest_opaque, left_middle);
      }
      if (right_sample.opaque) {
        nearest_opaque = std::max(nearest_opaque, right_middle);
      }
      throw OpaqueIntersectionDetected{nearest_opaque};
    }
    std::vector<SegmentOperator> left_operations;
    std::vector<SegmentOperator> right_operations;
    const bool passes = refinement_criteria_pass(
        center_sample, left_sample, right_sample, upper - lower,
        ray_length_cm, length_scale_cm, frequencies, contexts, tolerance,
        left_operations, right_operations, pilot_missed_emission);
    if (passes) {
      append_segment(lower, middle, left_sample);
      append_segment(middle, upper, right_sample);
      return;
    }
    if (pilot_missed_emission) {
      return;
    }
    if (depth >= kMaximumAdaptiveDepth || !(middle > lower && middle < upper)) {
      throw std::runtime_error(
          "adaptive longitudinal refinement did not converge");
    }
    std::vector<AdaptiveContext> left_contexts(contexts.size());
    std::vector<AdaptiveContext> right_contexts(contexts.size());
    for (std::size_t i = 0; i < contexts.size(); ++i) {
      left_contexts[i] = contexts[i];
      left_contexts[i].foreground_transmission =
          contexts[i].foreground_transmission *
          right_operations[i].transmission;

      right_contexts[i] = contexts[i];
      right_contexts[i].incoming_intensity =
          left_operations[i].transmission *
              contexts[i].incoming_intensity +
          left_operations[i].source_contribution;
      right_contexts[i].foreground_transmission =
          contexts[i].foreground_transmission;
    }
    self(self, lower, middle, depth + 1, left_contexts);
    self(self, middle, upper, depth + 1, right_contexts);
  };

  for (std::size_t i = 0; i < interval_count; ++i) {
    const double lower = s_min + i * interval_step;
    const double upper = i + 1 == interval_count
                             ? s_max
                             : s_min + (i + 1) * interval_step;
    refine(refine, lower, upper, 0, pilot_contexts[i]);
    if (pilot_missed_emission) {
      return {};
    }
  }
  return segments;
}

double integrate_adaptive_ray(const std::vector<AdaptiveSegment>& segments,
                              double frequency, double length_scale_cm);

template <typename Sampler>
void verify_reference_ray(std::vector<AdaptiveSegment>& segments,
                          const double ray_length_cm,
                          const std::vector<double>& frequencies,
                          const std::vector<double>& image_scales,
                          const double length_scale_cm,
                          const double tolerance, Sampler&& sample_at) {
  for (int iteration = 0; iteration <= kMaximumAdaptiveDepth; ++iteration) {
    const std::vector<std::vector<AdaptiveContext>> contexts =
        build_segment_pilot_contexts(segments, frequencies, length_scale_cm,
                                     image_scales);
    std::vector<AdaptiveSegment> refined;
    std::vector<AdaptiveSegment> probe_refined;
    refined.reserve(segments.size());
    probe_refined.reserve(segments.size() * 2);
    bool needs_refinement = false;
    for (std::size_t i = 0; i < segments.size(); ++i) {
      const AdaptiveSegment& segment = segments[i];
      const double middle =
          segment.lower + (segment.upper - segment.lower) * 0.5;
      const double left_middle = segment.lower + (middle - segment.lower) * 0.5;
      const double right_middle = middle + (segment.upper - middle) * 0.5;
      const FlowSample left_sample = sample_at(left_middle);
      const FlowSample right_sample = sample_at(right_middle);
      if (segment.sample.opaque || left_sample.opaque || right_sample.opaque) {
        double nearest_opaque = -std::numeric_limits<double>::infinity();
        if (segment.sample.opaque) {
          nearest_opaque = middle;
        }
        if (left_sample.opaque) {
          nearest_opaque = std::max(nearest_opaque, left_middle);
        }
        if (right_sample.opaque) {
          nearest_opaque = std::max(nearest_opaque, right_middle);
        }
        throw OpaqueIntersectionDetected{nearest_opaque};
      }
      probe_refined.push_back({segment.lower, middle, left_sample});
      probe_refined.push_back({middle, segment.upper, right_sample});

      const FlowSample& center_sample = segment.sample;
      std::vector<SegmentOperator> left_operations;
      std::vector<SegmentOperator> right_operations;
      bool pilot_missed_emission = false;
      if (refinement_criteria_pass(
              center_sample, left_sample, right_sample,
              segment.upper - segment.lower, ray_length_cm, length_scale_cm,
              frequencies, contexts[i], tolerance, left_operations,
              right_operations, pilot_missed_emission)) {
        refined.push_back(segment);
        continue;
      }

      const double split = middle;
      if (iteration == kMaximumAdaptiveDepth ||
          !(split > segment.lower && split < segment.upper)) {
        throw std::runtime_error(
            "adaptive longitudinal refinement did not converge");
      }
      if (refined.size() + (segments.size() - i) + 1 >
          kMaximumAdaptiveSegmentsPerRay) {
        throw std::runtime_error(
            "adaptive longitudinal ray exceeds the segment limit");
      }
      refined.push_back({segment.lower, split, left_sample});
      refined.push_back({split, segment.upper, right_sample});
      needs_refinement = true;
    }

    bool globally_converged = true;
    for (std::size_t i = 0; i < frequencies.size(); ++i) {
      const double current_intensity = integrate_adaptive_ray(
          segments, frequencies[i], length_scale_cm);
      const double refined_intensity = integrate_adaptive_ray(
          probe_refined, frequencies[i], length_scale_cm);
      const double absolute_budget =
          tolerance * contexts.front()[i].ray_intensity;
      const double intensity_scale =
          std::max(std::abs(current_intensity), std::abs(refined_intensity));
      if (std::abs(current_intensity - refined_intensity) > absolute_budget ||
          (intensity_scale > absolute_budget &&
           relative_difference(current_intensity, refined_intensity) >
               tolerance)) {
        globally_converged = false;
        break;
      }
    }
    if (!needs_refinement && globally_converged) {
      return;
    }
    if (needs_refinement) {
      segments = std::move(refined);
    } else {
      if (probe_refined.size() > kMaximumAdaptiveSegmentsPerRay) {
        throw std::runtime_error(
            "adaptive longitudinal ray exceeds the segment limit");
      }
      segments = std::move(probe_refined);
    }
  }
  throw std::runtime_error(
      "adaptive longitudinal refinement did not converge");
}

bool merged_operator_matches(const std::vector<AdaptiveSegment>& reference,
                             const std::size_t begin, const std::size_t end,
                             const AdaptiveSegment& candidate,
                             const std::vector<double>& frequencies,
                             const std::vector<std::vector<AdaptiveContext>>&
                                 contexts,
                             const double total_ray_length_cm,
                             const double length_scale_cm,
                             const double tolerance) {
  const double interval_fraction =
      (reference[end - 1].upper - reference[begin].lower) *
      length_scale_cm / total_ray_length_cm;
  for (std::size_t frequency_index = 0;
       frequency_index < frequencies.size(); ++frequency_index) {
    const double frequency = frequencies[frequency_index];
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
    const AdaptiveContext& incoming =
        contexts[begin][frequency_index];
    const AdaptiveContext& outgoing =
        contexts[end - 1][frequency_index];
    const double local_budget =
        tolerance * incoming.ray_intensity * interval_fraction;
    const double local_error =
        outgoing.foreground_transmission *
        (std::abs(reference_source -
                  candidate_operation.source_contribution) +
         std::abs(reference_transmission -
                  candidate_operation.transmission) *
             std::abs(incoming.incoming_intensity));
    if (local_error > local_budget) {
      return false;
    }
  }
  return true;
}

template <typename Sampler>
std::vector<AdaptiveSegment> compress_ray_segments(
    const std::vector<AdaptiveSegment>& reference,
    const std::vector<double>& frequencies,
    const std::vector<std::vector<AdaptiveContext>>& contexts,
    const double total_ray_length_cm, const double length_scale_cm,
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
                                   frequencies, contexts,
                                   total_ray_length_cm, length_scale_cm,
                                   tolerance)) {
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
  const std::vector<double> image_scales =
      image_intensity_scales(grid, view, options, frequencies);

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
      double visible_min = clipped_min;
      bool has_visible_material = clip_to_nearest_opaque_boundary(
          clipped_min, clipped_max, view.ds, sample_at, visible_min);
      if (has_visible_material) {
        std::vector<AdaptiveSegment> reference;
        int occultation_recoveries = 0;
        while (true) {
          try {
            std::size_t pilot_subdivisions = 1;
            const std::size_t interval_count =
                samples_for_span(clipped_max - visible_min, view.ds);
            const std::size_t maximum_pilot_subdivisions =
                kMaximumAdaptiveSegmentsPerRay / interval_count;
            while (true) {
              bool pilot_missed_emission = false;
              reference = refine_reference_ray(
                  visible_min, clipped_max, view.ds, frequencies,
                  options.length_scale_cm, image_scales,
                  options.integration_tolerance, pilot_subdivisions,
                  pilot_missed_emission, sample_at);
              if (!pilot_missed_emission) {
                break;
              }
              if (pilot_subdivisions > maximum_pilot_subdivisions / 2) {
                throw std::runtime_error(
                    "adaptive pilot refinement could not resolve emission "
                    "missed by the initial pilot");
              }
              pilot_subdivisions *= 2;
            }
            verify_reference_ray(
                reference, (clipped_max - visible_min) *
                               options.length_scale_cm,
                frequencies, image_scales, options.length_scale_cm,
                options.integration_tolerance, sample_at);
            break;
          } catch (const OpaqueIntersectionDetected& intersection) {
            if (++occultation_recoveries > 8) {
              throw std::runtime_error(
                  "could not recover the nearest opaque boundary on an "
                  "adaptive ray");
            }
            double recovered_min = visible_min;
            const double recovery_spacing =
                view.ds / std::pow(2.0, occultation_recoveries);
            has_visible_material = clip_to_nearest_opaque_boundary(
                intersection.position, clipped_max, recovery_spacing,
                sample_at, recovered_min);
            if (!has_visible_material) {
              break;
            }
            visible_min = recovered_min;
          }
        }
        if (!has_visible_material) {
          host_offsets[pixel + 1] = packed_segments.size();
          continue;
        }
        const std::vector<std::vector<AdaptiveContext>> reference_contexts =
            build_segment_pilot_contexts(reference, frequencies,
                                         options.length_scale_cm,
                                         image_scales);
        const std::vector<AdaptiveSegment> compressed =
            compress_ray_segments(
                reference, frequencies, reference_contexts,
                (clipped_max - visible_min) * options.length_scale_cm,
                options.length_scale_cm,
                options.integration_tolerance, sample_at);
        bool compressed_is_accurate = true;
        for (std::size_t frequency_index = 0;
             frequency_index < frequencies.size(); ++frequency_index) {
          const double frequency = frequencies[frequency_index];
          const double reference_intensity = integrate_adaptive_ray(
              reference, frequency, options.length_scale_cm);
          const double compressed_intensity = integrate_adaptive_ray(
              compressed, frequency, options.length_scale_cm);
          const double verification_budget =
              options.integration_tolerance *
              reference_contexts.front()[frequency_index].ray_intensity;
          const double verification_scale =
              std::max(std::abs(reference_intensity),
                       std::abs(compressed_intensity));
          if (std::abs(reference_intensity - compressed_intensity) >
                  verification_budget ||
              (verification_scale > verification_budget &&
               relative_difference(reference_intensity,
                                    compressed_intensity) >
                   options.integration_tolerance)) {
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

AdaptiveRaySamples build_analytical_ray_samples(
    const AnalyticalModel& model, const ImageGeometry& image,
    const Options& options,
    const std::vector<std::array<double, 2>>& ray_coordinates) {
  const std::size_t ray_count = ray_coordinates.size();
  if (ray_count >
      std::numeric_limits<std::size_t>::max() / options.ray_samples) {
    throw std::runtime_error("analytical ray segment count overflows");
  }
  const std::size_t maximum_segments = ray_count * options.ray_samples;
  std::vector<std::size_t> host_offsets(ray_count + 1, 0);
  std::vector<Vector3> positions;
  std::vector<double> segment_lengths;
  positions.reserve(maximum_segments);
  segment_lengths.reserve(maximum_segments);

  for (std::size_t ray = 0; ray < ray_count; ++ray) {
    const double u_coordinate = ray_coordinates[ray][0];
    const double v_coordinate = ray_coordinates[ray][1];
    const Vector3 origin{
        image.u.x * u_coordinate + image.v.x * v_coordinate,
        image.u.y * u_coordinate + image.v.y * v_coordinate,
        image.u.z * u_coordinate + image.v.z * v_coordinate};
    const double domain_min = options.ray_bounds[0];
    const double domain_max = options.ray_bounds[1];
    double visible_min = domain_min;
    if (visible_ray_interval(origin, image.los, model.opaque_objects,
                             domain_min, domain_max, visible_min)) {
      const double segment_length =
          (domain_max - visible_min) / options.ray_samples;
      if (segment_length > 0.0 && std::isfinite(segment_length)) {
        for (std::size_t sample = 0; sample < options.ray_samples; ++sample) {
          const double s = visible_min + (sample + 0.5) * segment_length;
          positions.push_back(
              {origin.x + image.los.x * s,
               origin.y + image.los.y * s,
               origin.z + image.los.z * s});
          segment_lengths.push_back(segment_length);
        }
      }
    }
    host_offsets[ray + 1] = positions.size();
  }

  std::vector<double> density_values;
  std::vector<double> temperature_values;
  std::vector<double> electron_values;
  std::vector<double> ion_charge2_values;
  model.evaluate(positions, density_values, temperature_values, electron_values,
                 ion_charge2_values);

  std::vector<std::array<double, kAdaptiveSegmentFieldCount>> host_segments(
      positions.size());
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const double density_ratio = density_values[i];
    const double temperature_ratio = temperature_values[i];
    const double electron_abundance = electron_values[i];
    const double ion_charge2_abundance = ion_charge2_values[i];
    if (!std::isfinite(density_ratio) || density_ratio < 0.0 ||
        !std::isfinite(temperature_ratio) || temperature_ratio < 0.0 ||
        !std::isfinite(electron_abundance) || electron_abundance < 0.0 ||
        !std::isfinite(ion_charge2_abundance) ||
        ion_charge2_abundance < 0.0) {
      throw std::runtime_error(
          "analytical flow functions must return finite, non-negative values");
    }
    if (density_ratio > 0.0 && temperature_ratio <= 0.0) {
      throw std::runtime_error(
          "temperature must be positive where density is non-zero");
    }
    const double temperature =
        options.temperature_scale * temperature_ratio;
    const double baryon_density =
        options.number_density_scale * density_ratio;
    const double electron_density = baryon_density * electron_abundance;
    const double ion_charge2_density =
        baryon_density * ion_charge2_abundance;
    const double emission_measure = electron_density * ion_charge2_density;
    const double ionic_charge =
        electron_abundance > 0.0
            ? ion_charge2_abundance / electron_abundance
            : 0.0;
    if (!std::isfinite(temperature) || !std::isfinite(emission_measure) ||
        !std::isfinite(ionic_charge)) {
      throw std::runtime_error(
          "analytical flow values overflow physical normalization");
    }
    host_segments[i] = {{
        temperature,
        emission_measure,
        ionic_charge,
        segment_lengths[i],
        0.0,
        0.0,
    }};
  }
  for (std::size_t ray = 0; ray < ray_count; ++ray) {
    double maximum_remaining_temperature = 0.0;
    for (std::size_t segment = host_offsets[ray];
         segment < host_offsets[ray + 1]; ++segment) {
      host_segments[segment][5] = maximum_remaining_temperature;
      maximum_remaining_temperature =
          std::max(maximum_remaining_temperature,
                   host_segments[segment][0]);
    }
  }

  AdaptiveSegmentView device_segments("analytical_ray_segments",
                                      host_segments.size(),
                                      kAdaptiveSegmentFieldCount);
  auto device_segment_host = Kokkos::create_mirror_view(device_segments);
  for (std::size_t i = 0; i < host_segments.size(); ++i) {
    for (std::size_t field = 0; field < kAdaptiveSegmentFieldCount; ++field) {
      device_segment_host(i, field) = host_segments[i][field];
    }
  }
  Kokkos::deep_copy(device_segments, device_segment_host);

  RayOffsetView device_offsets("analytical_ray_offsets", ray_count + 1);
  auto device_offsets_host = Kokkos::create_mirror_view(device_offsets);
  for (std::size_t i = 0; i < host_offsets.size(); ++i) {
    device_offsets_host(i) = host_offsets[i];
  }
  Kokkos::deep_copy(device_offsets, device_offsets_host);
  return {device_segments, device_offsets};
}

std::vector<std::array<double, 2>> regular_image_ray_coordinates(
    const ImageGeometry& image) {
  std::vector<std::array<double, 2>> coordinates;
  coordinates.reserve(image.nx * image.ny);
  for (std::size_t i = 0; i < image.nx; ++i) {
    for (std::size_t j = 0; j < image.ny; ++j) {
      coordinates.push_back({{
          image.u_min + (i + 0.5) * image.du,
          image.v_min + (j + 0.5) * image.dv,
      }});
    }
  }
  return coordinates;
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
              relative_intensity_threshold * intensity) {
            break;
          }
        }
        image(i, j) = intensity;
      });
  Kokkos::fence();
  return image;
}

template <typename ImageGeometryType>
Kokkos::View<double**> compute_adaptive_image(
    const AdaptiveRaySamples& samples, const ImageGeometryType& view,
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
              relative_intensity_threshold * intensity) {
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

template <typename ImageGeometryType>
double integrate_image(const Kokkos::View<double**>& image,
                       const ImageGeometryType& view,
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

void write_analytical_spectrum(
    const std::string& path, const std::vector<double>& frequencies,
    const std::vector<double>& projected_intensity_areas,
    const ImageGeometry& image, const AnalyticalModel& model,
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
  out << "# analytical_input: " << model.filename << '\n';
  out << "# coordinates: " << model.coordinates << '\n';
  out << "# image_resolution: " << image.nx << ',' << image.ny << '\n';
  out << "# ray_samples: " << options.ray_samples << '\n';
  out << "# ray_segment_length_units: R0\n";
  out << "# image_bounds_r0: " << image.u_min << ',' << image.u_max << ','
      << image.v_min << ',' << image.v_max << '\n';
  out << "# ray_bounds_r0: " << options.ray_bounds[0] << ','
      << options.ray_bounds[1] << '\n';
  out << "# longitudinal_integration: uniform_midpoint\n";
  out << "# line_of_sight: " << image.los.x << ',' << image.los.y << ','
      << image.los.z << '\n';
  out << "frequency_hz,projected_intensity_area\n";
  for (std::size_t i = 0; i < frequencies.size(); ++i) {
    out << frequencies[i] << ',' << projected_intensity_areas[i] << '\n';
  }
  if (!out) {
    throw std::runtime_error("failed while writing output spectrum: " + path);
  }
}

void run_analytical(const Options& options) {
  const AnalyticalModel model(options.input);
  const ImageGeometry image = make_image_geometry(options);
  validate_opaque_objects(model.opaque_objects, image, options.ray_bounds);
  const std::vector<std::array<double, 2>> ray_coordinates =
      regular_image_ray_coordinates(image);
  const AdaptiveRaySamples ray_samples = build_analytical_ray_samples(
      model, image, options, ray_coordinates);
  const std::vector<double> frequencies = spectrum_frequencies(options);
  std::vector<double> projected_intensity_areas;
  projected_intensity_areas.reserve(frequencies.size());
  for (const double frequency : frequencies) {
    const auto intensity_image = compute_adaptive_image(
        ray_samples, image, frequency, options.length_scale_cm,
        options.relative_intensity_threshold);
    projected_intensity_areas.push_back(
        integrate_image(intensity_image, image, options.length_scale_cm));
  }
  write_analytical_spectrum(options.output, frequencies,
                            projected_intensity_areas, image, model, options);
  std::cout << "Wrote " << frequencies.size()
            << " logarithmically spaced spectrum samples to " << options.output
            << '\n';
}

void run(const Options& options) {
  if (std::filesystem::path(options.input).extension() == ".py") {
    run_analytical(options);
    return;
  }
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
