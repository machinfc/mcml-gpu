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

int main(void)
{
  printf("=== Running MCML Physics & Unit Test Suite ===\n");
  test_rng_bounds();
  test_specular_reflectance();
  test_fresnel_boundary();
  test_henyey_greenstein_moment();
  test_energy_conservation();
  printf("=== All Physics Verification Tests Passed ===\n");
  return 0;
}
