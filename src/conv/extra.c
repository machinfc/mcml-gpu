/***********************************************************
 *	Numerical integration (trapezoidal quadrature), modified
 *	Bessel function exp(-x)*I0(x), and safe C99 I/O helpers
 *	for finite-beam convolution.
 ****/

#include "extra.h"

static void ReadLineSafe(char *buf, size_t size)
{
  if (fgets(buf, (int)size, stdin) == NULL) {
    buf[0] = '\0';
    return;
  }
  size_t len = strlen(buf);
  while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) {
    buf[--len] = '\0';
  }
}

double ZMin(double **Z, long IXmax, long IYmax, double Dx, double Dy,
            struct XY *Pmin_Ptr)
{
  double zmin = Z[0][0];
  long i, j;

  Pmin_Ptr->x = 0.0;
  Pmin_Ptr->y = 0.0;
  for (i = 0; i < IXmax; i++)
    for (j = 0; j < IYmax; j++)
      if (Z[i][j] < zmin) {
        zmin = Z[i][j];
        Pmin_Ptr->x = (double)i * Dx;
        Pmin_Ptr->y = (double)j * Dy;
      }
  return zmin;
}

double ZMax(double **Z, long IXmax, long IYmax, double Dx, double Dy,
            struct XY *Pmax_Ptr)
{
  double zmax = Z[0][0];
  long i, j;

  Pmax_Ptr->x = 0.0;
  Pmax_Ptr->y = 0.0;
  for (i = 0; i < IXmax; i++)
    for (j = 0; j < IYmax; j++)
      if (Z[i][j] > zmax) {
        zmax = Z[i][j];
        Pmax_Ptr->x = (double)i * Dx;
        Pmax_Ptr->y = (double)j * Dy;
      }
  return zmax;
}

void IsoPlot(double **Z, long int IXmax, long int IYmax, double Dx, double Dy)
{
  (void)Z;
  (void)IXmax;
  (void)IYmax;
  (void)Dx;
  (void)Dy;
}

short GetShort(short Lo, short Hi)
{
  char in_str[STRLEN];
  short x = Lo;

  ReadLineSafe(in_str, sizeof(in_str));
  sscanf(in_str, "%hd", &x);
  while (x < Lo || x > Hi) {
    printf("...Wrong parameter. Input again: ");
    ReadLineSafe(in_str, sizeof(in_str));
    if (in_str[0] == '\0')
      break;
    sscanf(in_str, "%hd", &x);
  }
  return x;
}

float GetFloat(float Lo, float Hi)
{
  char in_str[STRLEN];
  float x = Lo;

  ReadLineSafe(in_str, sizeof(in_str));
  sscanf(in_str, "%f", &x);
  while (x < Lo || x > Hi) {
    printf("...Wrong parameter. Input again: ");
    ReadLineSafe(in_str, sizeof(in_str));
    if (in_str[0] == '\0')
      break;
    sscanf(in_str, "%f", &x);
  }
  return x;
}

FILE *GetWriteFile(char *Ext)
{
  FILE *file = NULL;
  char fname[STRLEN];

  if (Ext != NULL && Ext[0] != '\0') {
    snprintf(fname, sizeof(fname), "out.%s", Ext);
  } else {
    snprintf(fname, sizeof(fname), "out.Arzc");
  }
  file = fopen(fname, "w");
  return file;
}

Boolean AllocConvData(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  int nz = In_Ptr->nz;
  int nr = In_Ptr->nrc;
  int na = In_Ptr->na;

  if (nz <= 0 || nr <= 0 || na <= 0) {
    printf("Wrong grid parameters.\n");
    return 0;
  }
  Out_Ptr->Rd_rac = AllocMatrix(0, nr - 1, 0, na - 1);
  Out_Ptr->Rd_rc = AllocVector(0, nr - 1);
  Out_Ptr->A_rzc = AllocMatrix(0, nr - 1, 0, nz - 1);
  Out_Ptr->Tt_rac = AllocMatrix(0, nr - 1, 0, na - 1);
  Out_Ptr->Tt_rc = AllocVector(0, nr - 1);

  return 1;
}

void FreeConvData(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  int nz = In_Ptr->nz;
  int nr = In_Ptr->nrc;
  int na = In_Ptr->na;

  if (nz <= 0 || nr <= 0 || na <= 0)
    return;

  if (Out_Ptr->Rd_rac != NULL) {
    FreeMatrix(Out_Ptr->Rd_rac, 0, nr - 1, 0, na - 1);
    Out_Ptr->Rd_rac = NULL;
  }
  if (Out_Ptr->Rd_rc != NULL) {
    FreeVector(Out_Ptr->Rd_rc, 0, nr - 1);
    Out_Ptr->Rd_rc = NULL;
  }
  if (Out_Ptr->A_rzc != NULL) {
    FreeMatrix(Out_Ptr->A_rzc, 0, nr - 1, 0, nz - 1);
    Out_Ptr->A_rzc = NULL;
  }
  if (Out_Ptr->Tt_rac != NULL) {
    FreeMatrix(Out_Ptr->Tt_rac, 0, nr - 1, 0, na - 1);
    Out_Ptr->Tt_rac = NULL;
  }
  if (Out_Ptr->Tt_rc != NULL) {
    FreeVector(Out_Ptr->Tt_rc, 0, nr - 1);
    Out_Ptr->Tt_rc = NULL;
  }
}

#define FUNC(x) ((*func)(x))
float trapzd(float (*func)(float), float a, float b, int n)
{
  float x, tnm, sum, del;
  static float s;
  static int it;
  int j;

  if (n == 1) {
    it = 1;
    s = 0.5f * (b - a) * (FUNC(a) + FUNC(b));
  } else {
    tnm = (float)it;
    del = (b - a) / tnm;
    x = a + 0.5f * del;
    for (sum = 0.0f, j = 1; j <= it; j++, x += del)
      sum += FUNC(x);
    it *= 2;
    s = 0.5f * (s + (b - a) * sum / tnm);
  }
  return s;
}
#undef FUNC

#define JMAX 20
float qtrap(float (*func)(float), float a, float b, float EPS)
{
  int j;
  float s = 0.0f, s_old = 0.0f;

  for (j = 1; j <= JMAX; j++) {
    s = trapzd(func, a, b, j);
    if (j <= 3 || fabsf(s - s_old) > EPS * fabsf(s_old))
      s_old = s;
    else
      break;
  }
  return s;
}
#undef JMAX

/***********************************************************
 *	Modified Bessel function exp(-x) * I0(x), for x >= 0.
 ****/
double BessI0(double x)
{
  double ax, ans;
  double y;

  if ((ax = fabs(x)) < 3.75) {
    y = x / 3.75;
    y *= y;
    ans = exp(-ax) *
          (1.0 +
           y * (3.5156229 +
                y * (3.0899424 +
                     y * (1.2067492 +
                          y * (0.2659732 +
                               y * (0.360768e-1 + y * 0.45813e-2))))));
  } else {
    y = 3.75 / ax;
    ans = (1.0 / sqrt(ax)) *
          (0.39894228 +
           y * (0.1328592e-1 +
                y * (0.225319e-2 +
                     y * (-0.157565e-2 +
                          y * (0.916281e-2 +
                               y * (-0.2057706e-1 +
                                    y * (0.2635537e-1 +
                                         y * (-0.1647633e-1 +
                                              y * 0.392377e-2))))))));
  }
  return ans;
}
