/***********************************************************
 *  Unit and Physics Verification Suite for MCML
 *
 *  Validates:
 *  1. PRNG bounds and uniformity (0 <= xi < 1)
 *  2. Analytical specular reflectance (Fresnel normal incidence)
 *  3. Snell/Fresnel boundary reflection (matched, normal, TIR)
 *  4. Henyey-Greenstein phase function first moment (<cos theta> ~ g)
 *  5. End-to-end Monte Carlo radiative energy conservation
 *     (Rsp + Rd + A + Tt = 1.0 within statistical tolerance)
 ****/

#include "mcml.h"
#include "conv.h"
#include <assert.h>

#define ASSERT_NEAR(actual, expected, tol, msg)                                \
  do {                                                                         \
    double _a = (actual);                                                      \
    double _e = (expected);                                                    \
    double _diff = fabs(_a - _e);                                              \
    if (_diff > (tol)) {                                                       \
      fprintf(stderr, "[FAIL] %s: expected %.6f +/- %.6f, got %.6f\n", (msg),  \
              _e, (double)(tol), _a);                                          \
      exit(1);                                                                 \
    } else {                                                                   \
      printf("[PASS] %-48s (val = %.6f)\n", (msg), _a);                        \
    }                                                                          \
  } while (0)

static void test_rng_bounds(void)
{
  long i;
  double sum = 0.0;
  const long N = 200000;

  for (i = 0; i < N; i++) {
    double r = RandomNum();
    if (r < 0.0 || r > 1.0) {
      fprintf(stderr, "[FAIL] RNG out of [0, 1] bounds: %f\n", r);
      exit(1);
    }
    sum += r;
  }
  ASSERT_NEAR(sum / (double)N, 0.5, 0.01, "RNG mean converges to 0.5");
}

static void test_specular_reflectance(void)
{
  LayerStruct layers[3];
  double rsp;

  /* Air (n=1.0) to turbid medium (n=1.5) -> Rsp = ((1.0 - 1.5)/(1.0 + 1.5))^2 = 0.04 */
  memset(layers, 0, sizeof(layers));
  layers[0].n = 1.0;
  layers[1].n = 1.5;
  layers[1].mua = 10.0;
  layers[1].mus = 90.0;
  layers[2].n = 1.0;

  rsp = Rspecular(layers);
  ASSERT_NEAR(rsp, 0.040000, 1e-7, "Specular reflectance (n=1.0 -> n=1.5)");
}

static void test_fresnel_boundary(void)
{
  double ca2 = 0.0;
  double r;

  /* 1. Matched refractive index boundary (n1 = n2 = 1.37) -> R = 0 */
  r = RFresnel(1.37, 1.37, 0.8, &ca2);
  ASSERT_NEAR(r, 0.0, 1e-9, "Fresnel matched boundary (n1 == n2)");
  ASSERT_NEAR(ca2, 0.8, 1e-9, "Fresnel matched transmission cosine");

  /* 2. Normal incidence (ca1 = 1.0, n1 = 1.0, n2 = 1.5) -> R = 0.04 */
  r = RFresnel(1.0, 1.5, 1.0, &ca2);
  ASSERT_NEAR(r, 0.04, 1e-7, "Fresnel normal incidence (n1=1.0, n2=1.5)");

  /* 3. Total internal reflection (n1 = 1.5, n2 = 1.0, sin(a1) > 1/1.5) */
  r = RFresnel(1.5, 1.0, 0.5, &ca2);
  ASSERT_NEAR(r, 1.0, 1e-9, "Fresnel total internal reflection (TIR)");
}

static void test_henyey_greenstein_moment(void)
{
  long i;
  const long N = 200000;
  double sum_iso = 0.0;
  double sum_fwd = 0.0;
  const double g_target = 0.85;

  for (i = 0; i < N; i++) {
    sum_iso += SpinTheta(0.0);
    sum_fwd += SpinTheta(g_target);
  }

  ASSERT_NEAR(sum_iso / (double)N, 0.0, 0.01,
              "Henyey-Greenstein isotropic <cos theta> ~ 0.0");
  ASSERT_NEAR(sum_fwd / (double)N, g_target, 0.01,
              "Henyey-Greenstein anisotropic <cos theta> ~ g");
}

static void test_energy_conservation(void)
{
  InputStruct in_parm;
  OutStruct out_parm;
  PhotonStruct photon;
  long i;
  double total_energy;

  memset(&in_parm, 0, sizeof(in_parm));
  in_parm.num_photons = 50000;
  in_parm.Wth = WEIGHT;
  in_parm.dz = 0.01;
  in_parm.dr = 0.01;
  in_parm.nz = 40;
  in_parm.nr = 50;
  in_parm.na = 10;
  in_parm.da = 0.5 * PI / in_parm.na;
  in_parm.num_layers = 1;

  in_parm.layerspecs =
      (LayerStruct *)malloc(3 * sizeof(LayerStruct));
  assert(in_parm.layerspecs != NULL);

  /* Ambient top */
  in_parm.layerspecs[0].n = 1.0;

  /* Single turbid slab (Wang et al. 1995 test parameters) */
  in_parm.layerspecs[1].n = 1.5;
  in_parm.layerspecs[1].mua = 10.0;
  in_parm.layerspecs[1].mus = 90.0;
  in_parm.layerspecs[1].g = 0.75;
  in_parm.layerspecs[1].z0 = 0.0;
  in_parm.layerspecs[1].z1 = 0.02;
  in_parm.layerspecs[1].cos_crit0 = sqrt(1.0 - 1.0 / (1.5 * 1.5));
  in_parm.layerspecs[1].cos_crit1 = sqrt(1.0 - 1.0 / (1.5 * 1.5));

  /* Ambient bottom */
  in_parm.layerspecs[2].n = 1.0;

  InitOutputData(in_parm, &out_parm);
  out_parm.Rsp = Rspecular(in_parm.layerspecs);

  for (i = 0; i < in_parm.num_photons; i++) {
    LaunchPhoton(out_parm.Rsp, in_parm.layerspecs, &photon);
    do {
      HopDropSpin(&in_parm, &photon, &out_parm);
    } while (!photon.dead);
  }

  SumScaleResult(in_parm, &out_parm);
  total_energy = out_parm.Rsp + out_parm.Rd + out_parm.A + out_parm.Tt;

  ASSERT_NEAR(total_energy, 1.000000, 0.002,
              "Radiative energy conservation (Rsp+Rd+A+Tt=1)");

  FreeData(in_parm, &out_parm);
}

static void test_32bit_grid_indexing(void)
{
  /* Verify 32-bit grid index support beyond the legacy 16-bit short ceiling
   * (32,767) by allocating and indexing a 40,000-element vector. */
  const int large_nz = 40000;
  double *vec = AllocVector(0, large_nz - 1);
  assert(vec != NULL);
  vec[large_nz - 1] = 1.0;
  ASSERT_NEAR(vec[large_nz - 1], 1.0, 1e-12,
              "32-bit grid index allocation (nz = 40,000 > 32,767)");
  FreeVector(vec, 0, large_nz - 1);
}

#ifdef ENABLE_CUDA
static void test_cuda_hotspot_large_grid(void)
{
  InputStruct in_parm;
  OutStruct out_parm;
  double total_energy;

  if (!InitDeviceCUDA()) {
    printf("[SKIP] No CUDA GPU detected for large-grid hot-spot test\n");
    return;
  }

  /* Basal Cell Carcinoma 630nm optical parameters on a 100x100 grid (10,000
   * bins > 4,096 shared-memory limit), exercising SHARED_MODE=2 (64x64
   * hot-spot sub-grid privatization) + persistent in-loop photon respawning
   * across both Mixed FP32/FP64 and Full FP64 modes. */
  memset(&in_parm, 0, sizeof(in_parm));
  in_parm.num_photons = 100000;
  in_parm.Wth = WEIGHT;
  in_parm.dz = 0.002;
  in_parm.dr = 0.004;
  in_parm.nz = 100;
  in_parm.nr = 100;
  in_parm.na = 1;
  in_parm.da = 0.5 * PI / in_parm.na;
  in_parm.num_layers = 1;

  in_parm.layerspecs = (LayerStruct *)malloc(3 * sizeof(LayerStruct));
  assert(in_parm.layerspecs != NULL);

  in_parm.layerspecs[0].n = 1.0;
  in_parm.layerspecs[1].n = 1.37;
  in_parm.layerspecs[1].mua = 1.7;
  in_parm.layerspecs[1].mus = 125.0;
  in_parm.layerspecs[1].g = 0.80;
  in_parm.layerspecs[1].z0 = 0.0;
  in_parm.layerspecs[1].z1 = 0.20;
  in_parm.layerspecs[1].cos_crit0 = sqrt(1.0 - 1.0 / (1.37 * 1.37));
  in_parm.layerspecs[1].cos_crit1 = sqrt(1.0 - 1.0 / (1.37 * 1.37));
  in_parm.layerspecs[2].n = 1.0;

  /* 1. Test Default Mixed FP32/FP64 Mode */
  SetPrecisionCUDA(0);
  InitOutputData(in_parm, &out_parm);
  out_parm.Rsp = Rspecular(in_parm.layerspecs);
  assert(RunSimulationCUDA(&in_parm, &out_parm) == 1);
  SumScaleResult(in_parm, &out_parm);
  total_energy = out_parm.Rsp + out_parm.Rd + out_parm.A + out_parm.Tt;
  ASSERT_NEAR(total_energy, 1.000000, 0.002,
              "CUDA Mixed FP32/FP64 energy (Rsp+Rd+A+Tt=1)");
  ASSERT_NEAR(out_parm.A, 0.575, 0.01,
              "CUDA Mixed FP32/FP64 BCC 630nm absorption (A~0.575)");
  FreeData(in_parm, &out_parm);

  /* 2. Test Full FP64 Mode */
  in_parm.layerspecs = (LayerStruct *)malloc(3 * sizeof(LayerStruct));
  assert(in_parm.layerspecs != NULL);
  in_parm.layerspecs[0].n = 1.0;
  in_parm.layerspecs[1].n = 1.37;
  in_parm.layerspecs[1].mua = 1.7;
  in_parm.layerspecs[1].mus = 125.0;
  in_parm.layerspecs[1].g = 0.80;
  in_parm.layerspecs[1].z0 = 0.0;
  in_parm.layerspecs[1].z1 = 0.20;
  in_parm.layerspecs[1].cos_crit0 = sqrt(1.0 - 1.0 / (1.37 * 1.37));
  in_parm.layerspecs[1].cos_crit1 = sqrt(1.0 - 1.0 / (1.37 * 1.37));
  in_parm.layerspecs[2].n = 1.0;

  SetPrecisionCUDA(1);
  InitOutputData(in_parm, &out_parm);
  out_parm.Rsp = Rspecular(in_parm.layerspecs);
  assert(RunSimulationCUDA(&in_parm, &out_parm) == 1);
  SumScaleResult(in_parm, &out_parm);
  total_energy = out_parm.Rsp + out_parm.Rd + out_parm.A + out_parm.Tt;
  ASSERT_NEAR(total_energy, 1.000000, 0.002,
              "CUDA Full FP64 energy (Rsp+Rd+A+Tt=1)");
  SetPrecisionCUDA(0);
  FreeData(in_parm, &out_parm);
}
#endif

static void test_finite_beam_convolution(void)
{
  /* 1. Modified Bessel function exp(-0)*I0(0) == 1.0 */
  ASSERT_NEAR(BessI0(0.0), 1.0, 1e-7,
              "Modified Bessel exp(-x)*I0(x) at x = 0");

  /* 2. Flat-top circle overlap ITheta(r, r2, R):
   *    When R >= r + r2 (full interior overlap), ITheta == 1.0;
   *    when R < |r - r2| (disjoint), ITheta == 0.0. */
  ASSERT_NEAR(ITheta(0.02, 0.03, 0.10), 1.0, 1e-9,
              "Flat-beam overlap kernel ITheta (interior = 1.0)");
  ASSERT_NEAR(ITheta(0.25, 0.05, 0.10), 0.0, 1e-9,
              "Flat-beam overlap kernel ITheta (disjoint = 0.0)");

  /* 3. Convolve a uniform diffuse reflectance sheet Rd(r) = 1.0 [1/cm^2]
   *    with a flat-top circular beam (P = 1.0 J, R = 0.20 cm):
   *    Near r = 0 (inside the flat-top radius), the convolved energy density
   *    must equal the incident irradiance P / (pi * R^2) * (pi * Rd) -> P/R^2 * R^2 = P / (R^2)?
   *    Specifically: ConvRd_r computes (2*P/R^2) * int_0^R 1.0 * 1.0 * r2 dr2
   *    = (2*P/R^2) * (R^2 / 2) = P = 1.0 J/cm^2. */
  InputConvStruct in_conv;
  OutConvStruct out_conv;
  memset(&in_conv, 0, sizeof(in_conv));
  memset(&out_conv, 0, sizeof(out_conv));

  in_conv.nz = 10;
  in_conv.nr = 100;
  in_conv.na = 1;
  in_conv.nrc = 10;
  in_conv.dz = 0.01;
  in_conv.dr = 0.01;
  in_conv.drc = 0.01;
  in_conv.da = 0.5 * PI;
  in_conv.eps = 1e-4f;
  in_conv.beam.type = flat;
  in_conv.beam.P = 1.0;
  in_conv.beam.R = 0.20;

  out_conv.Rd_r = AllocVector(0, in_conv.nr - 1);
  for (int ir = 0; ir < in_conv.nr; ir++) {
    out_conv.Rd_r[ir] = 1.0;
  }
  AllocConvData(&in_conv, &out_conv);
  out_conv.allocated = 1;

  ConvRd_r(&in_conv, &out_conv);
  ASSERT_NEAR(out_conv.Rd_rc[0], 1.0, 0.01,
              "Flat-beam radial convolution on uniform sheet");

  FreeVector(out_conv.Rd_r, 0, in_conv.nr - 1);
  FreeConvData(&in_conv, &out_conv);
}

int main(void)
{
  printf("=== Running MCML Physics & Unit Test Suite ===\n");
  test_rng_bounds();
  test_specular_reflectance();
  test_fresnel_boundary();
  test_henyey_greenstein_moment();
  test_energy_conservation();
  test_32bit_grid_indexing();
  test_finite_beam_convolution();
#ifdef ENABLE_CUDA
  test_cuda_hotspot_large_grid();
#endif
  printf("=== All Physics Verification Tests Passed ===\n");
  return 0;
}
