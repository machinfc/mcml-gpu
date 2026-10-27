/***********************************************************
 *	Program Name: conv.
 *
 *	Finite-beam convolution post-processor for MCML
 *	(Flat-top circular and Gaussian laser beams).
 ****/

#ifndef CONV_H
#define CONV_H

#include "mcml.h"
#include <float.h>
#include <limits.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define NULLCONVSTRUCT {0, 0, 0, 0, 0}
#define NULLOUTSTRUCT                                                          \
  {0.0,  NULL, NULL, NULL, 0.0,  NULL, NULL, NULL, 0.0,           NULL,        \
   NULL, NULL, 0.0,  NULL, NULL, NULL, NULL, NULL, 0,   NULLCONVSTRUCT}

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { pencil, flat, gaussian } BeamType;

typedef struct {
  BeamType type;
  double P; /* Total incident laser beam energy [J] */
  double R; /* Beam radius (flat-top radius or 1/e^2 Gaussian radius) [cm] */
} BeamStruct;

typedef struct {
  char in_fname[STRLEN];
  char out_fname[STRLEN];
  char out_fformat;
  long num_photons;
  double Wth;
  double dz;
  double dr;
  double da;
  int nz;
  int nr;
  int na;
  int nrc;    /* Number of radial grid points after convolution */
  double drc; /* Radial grid separation after convolution [cm] */
  float eps;  /* Relative integration tolerance */
  BeamStruct beam;
  short num_layers;
  LayerStruct *layerspecs;
} InputConvStruct;

typedef struct {
  Boolean Rd_ra;
  Boolean Rd_r;
  Boolean A_rz;
  Boolean Tt_ra;
  Boolean Tt_r;
} ConvStruct;

typedef struct {
  double Rsp;
  double **Rd_ra;
  double *Rd_r;
  double *Rd_a;
  double Rd;

  double **A_rz;
  double *A_z;
  double *A_l;
  double A;

  double **Tt_ra;
  double *Tt_r;
  double *Tt_a;
  double Tt;

  double **Rd_rac; /* Convolved diffuse reflectance [J/(cm2 sr)] */
  double *Rd_rc;   /* Convolved radial reflectance [J/cm2] */
  double **A_rzc;  /* Convolved absorbed energy density [J/cm3] */
  double **Tt_rac; /* Convolved transmittance [J/(cm2 sr)] */
  double *Tt_rc;   /* Convolved radial transmittance [J/cm2] */

  char allocated;
  ConvStruct conved;
} OutConvStruct;

void IsoPlot(double **Z, long int IXmax, long int IYmax, double Dx, double Dy);
short GetShort(short Lo, short Hi);
float GetFloat(float Lo, float Hi);
FILE *GetWriteFile(char *Ext);
Boolean AllocConvData(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr);
void FreeConvData(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr);
float trapzd(float (*func)(float), float a, float b, int n);
float qtrap(float (*func)(float), float a, float b, float EPS);
double BessI0(double x);
double ITheta(double r, double r2, double R);
double ExpBessI0(double r, double r2, double R);

void ConvA_rz(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr);
void ConvRd_ra(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr);
void ConvRd_r(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr);
void ConvTt_ra(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr);
void ConvTt_r(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr);
Boolean ReadMcoFile(const char *mco_fname, InputConvStruct *In_Ptr,
                    OutConvStruct *Out_Ptr);
void FreeMcoConvData(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr);

#ifdef __cplusplus
}
#endif

#endif /* CONV_H */
