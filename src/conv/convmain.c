/****************************************************************
 *	CLI Entry Point for the Finite-Beam Laser Convolution
 *	Post-Processor (build/conv).
 *
 *	Usage:
 *	  ./build/conv <input.mco> <output.dat> [--beam gaussian|flat]
 *	               [--power <J>] [--radius <cm>]
 ****/

#include "conv.h"
#include "extra.h"

int main(int argc, char *argv[])
{
  const char *in_mco = NULL;
  const char *out_dat = NULL;
  BeamType beam_type = gaussian;
  double beam_power = 1.0;  /* 1.0 Joule default */
  double beam_radius = 0.1; /* 0.1 cm (1 mm) default radius */

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--beam") == 0 && i + 1 < argc) {
      i++;
      if (strcmp(argv[i], "flat") == 0 || strcmp(argv[i], "FLAT") == 0)
        beam_type = flat;
      else
        beam_type = gaussian;
    } else if (strcmp(argv[i], "--power") == 0 && i + 1 < argc) {
      beam_power = atof(argv[++i]);
    } else if (strcmp(argv[i], "--radius") == 0 && i + 1 < argc) {
      beam_radius = atof(argv[++i]);
    } else if (argv[i][0] != '-') {
      if (!in_mco)
        in_mco = argv[i];
      else if (!out_dat)
        out_dat = argv[i];
    }
  }

  if (!in_mco) {
    fprintf(stderr,
            "Usage: %s <input.mco> [output.dat] [--beam gaussian|flat] "
            "[--power <J>] [--radius <cm>]\n",
            argv[0]);
    return 1;
  }

  InputConvStruct in_conv;
  OutConvStruct out_conv;

  if (!ReadMcoFile(in_mco, &in_conv, &out_conv)) {
    fprintf(stderr, "[ERROR] Failed to read MCML output file: %s\n", in_mco);
    return 1;
  }

  in_conv.beam.type = beam_type;
  in_conv.beam.P = (beam_power > 0.0) ? beam_power : 1.0;
  in_conv.beam.R = (beam_radius > 0.0) ? beam_radius : 0.1;

  printf("[CONV] Convolving %s (%s beam, P = %.3f J, R = %.4f cm, grid %dx%d)\n",
         in_mco, (beam_type == flat) ? "Flat-Top" : "Gaussian",
         in_conv.beam.P, in_conv.beam.R, in_conv.nrc, in_conv.nz);

  ConvA_rz(&in_conv, &out_conv);
  ConvRd_r(&in_conv, &out_conv);
  ConvTt_r(&in_conv, &out_conv);

  const char *target_out = out_dat ? out_dat : "convolved_Arzc.dat";
  FILE *fp = fopen(target_out, "w");
  if (!fp) {
    fprintf(stderr, "[ERROR] Cannot open output file: %s\n", target_out);
    FreeMcoConvData(&in_conv, &out_conv);
    return 1;
  }

  fprintf(fp, "# Finite-Beam Convolved Absorbed Energy Density A_rzc [J/cm^3]\n");
  fprintf(fp, "# Source: %s | Beam: %s | Power: %.4f J | Radius: %.4f cm\n",
          in_mco, (beam_type == flat) ? "Flat-Top" : "Gaussian",
          in_conv.beam.P, in_conv.beam.R);
  fprintf(fp, "%-12s\t%-12s\t%-12s\n", "r[cm]", "z[cm]", "A_rzc[J/cm3]");

  for (int ir = 0; ir < in_conv.nrc; ir++) {
    double r = (ir + 0.5) * in_conv.drc;
    for (int iz = 0; iz < in_conv.nz; iz++) {
      double z = (iz + 0.5) * in_conv.dz;
      fprintf(fp, "%-12.4E\t%-12.4E\t%-12.4E\n", r, z,
              out_conv.A_rzc[ir][iz]);
    }
  }

  fclose(fp);
  printf("[CONV] Wrote convolved 2D distribution to %s\n", target_out);
  FreeMcoConvData(&in_conv, &out_conv);
  return 0;
}
