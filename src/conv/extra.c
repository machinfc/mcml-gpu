#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <float.h>
#include <limits.h>


void 
IsoPlot(double **Z,		/* the 2D array Z[i][j]. */
	long int IXmax,
	long int IYmax,		/* the 0<=i<=IXmax, 0<=j<=IYmax. */
	double Dx,
	double Dy)
{				/* the gridline separations. */
  FILE       *isofile;		/* send isolines to this file. */
  struct XY   pmin, pmax;	/* the xy positions of the min & max. */
  double      zmin, zmax;
  IsoList     isos;
  char        fname[128] = "iso";

  /* Locate the min & max. */
  zmin = ZMin(Z, IXmax, IYmax, Dx, Dy, &pmin);
  zmax = ZMax(Z, IXmax, IYmax, Dx, Dy, &pmax);

  if (zmin == zmax || IXmax * IYmax < 4) {
    printf("...Not enough data for contour plot\n");
    return;
  }
  isofile = GetWriteFile(fname);
  if (isofile == NULL)
    return;

  printf("The range of the value is %lf to %lf.\n", zmin, zmax);
  isos = GetIsoValues(zmin, zmax);

  GetIsoLines(isos, Z, IXmax, IYmax, Dx, Dy, pmax);
  WriteIsoLines(isofile, isos);

  FreeIsoLines(isos);

  fclose(isofile);
}

/****************************************************************
 ****/
short
GetShort(short Lo, short Hi)
{
  char        in_str[STRLEN];
  short       x;

  gets(in_str);
  sscanf(in_str, "%hd", &x);
  while (x < Lo || x > Hi) {
    printf("...Wrong paramter.  Input again: ");
    gets(in_str);
    sscanf(in_str, "%hd", &x);
  }
  return (x);
}


/***********************************************************
 *	Trapzoidal integration.
 ****/

#define FUNC(x) ((*func)(x))

float
trapzd(
       float (*func) (float),
       float a, float b, int n)
{
  float       x, tnm, sum, del;
  static float s;
  static int  it;
  int         j;

  if (n == 1) {
    it = 1;
    s = 0.5 * (b - a) * (FUNC(a) + FUNC(b));
  } else {
    tnm = it;
    del = (b - a) / tnm;
    x = a + 0.5 * del;
    for (sum = 0.0, j = 1; j <= it; j++, x += del)
      sum += FUNC(x);
    it *= 2;
    s = 0.5 * (s + (b - a) * sum / tnm);
  }
  return (s);
}
#undef FUNC



/****************************************************************
 ****/
float
GetFloat(float Lo, float Hi)
{
  char        in_str[STRLEN];
  float       x;

  gets(in_str);
  sscanf(in_str, "%f", &x);
  while (x < Lo || x > Hi) {
    printf("...Wrong paramter.  Input again: ");
    gets(in_str);
    sscanf(in_str, "%f", &x);
  }
  return (x);
}

/****************************************************************
 *	Open a file for output with extension Ext.  If file exists,
 *	ask whether to overwrite or append or change filename.
 *
 *	Return file pointer, which could be NULL.
 *	Return the full filename as Ext.
 ****/
FILE       *
GetWriteFile(char *Ext)
{
  FILE       *file;
  char        fname[STRLEN], fmode[STRLEN];

  do {
    printf("Enter output filename with extension .%s (or . to quit): ", Ext);
    gets(fname);
    if (strlen(fname) == 1 && fname[0] == '.') {
      fmode[0] = 'q';
      break;
    } else
      fmode[0] = 'w';

    if ((file = fopen(fname, "r")) != NULL) {	/* file exists. */
      printf("File %s exists, %s",
	     fname, "w=overwrite, a=append, n=new filename, q=quit: ");
      do
	gets(fmode);
      while (!strlen(fmode));	/* avoid null line. */
      fclose(file);
    }
  } while (fmode[0] != 'w' && fmode[0] != 'a' && fmode[0] != 'q');

  if (fmode[0] != 'q')
    file = fopen(fname, fmode);
  else
    file = NULL;

  strcpy(Ext, fname);
  return (file);		/* could be NULL. */
}

/****************************************************************
 *  Allocate the arrays for convolution in OutStruct for each
 *	run, and they are automatically initialized to zeros.
 *
 *  Returns 1 if successful, otherwise 0.
 ****/
Boolean 
AllocConvData(InputStruct * In_Ptr,
	      OutStruct * Out_Ptr)
{
  short       nz = In_Ptr->nz;
  short       nr = In_Ptr->nrc;	/* use nrc instead of nr. */
  short       na = In_Ptr->na;

  if (nz <= 0 || nr <= 0 || na <= 0) {
    printf("Wrong grid parameters.\n");
    return (0);
  }
  Out_Ptr->Rd_rac = AllocMatrix(0, nr - 1, 0, na - 1);
  Out_Ptr->Rd_rc = AllocVector(0, nr - 1);
  Out_Ptr->A_rzc = AllocMatrix(0, nr - 1, 0, nz - 1);
  Out_Ptr->Tt_rac = AllocMatrix(0, nr - 1, 0, na - 1);
  Out_Ptr->Tt_rc = AllocVector(0, nr - 1);

  return (1);
}

/****************************************************************
 *  Free the arrays in OutStruct.
 ****/
void 
FreeConvData(InputStruct * In_Ptr,
	     OutStruct * Out_Ptr)
{
  short       nz = In_Ptr->nz;
  short       nr = In_Ptr->nrc;
  short       na = In_Ptr->na;

  if (nz <= 0 || nr <= 0 || na <= 0) {
    printf("Wrong grid parameters.\n");
    return;
  }
  if (Out_Ptr->Rd_rac != NULL)
    FreeMatrix(Out_Ptr->Rd_rac, 0, nr - 1, 0, na - 1);
  if (Out_Ptr->Rd_rc != NULL)
    FreeVector(Out_Ptr->Rd_rc, 0, nr - 1);
  if (Out_Ptr->A_rzc != NULL)
    FreeMatrix(Out_Ptr->A_rzc, 0, nr - 1, 0, nz - 1);
  if (Out_Ptr->Tt_rac != NULL)
    FreeMatrix(Out_Ptr->Tt_rac, 0, nr - 1, 0, na - 1);
  if (Out_Ptr->Tt_rc != NULL)
    FreeVector(Out_Ptr->Tt_rc, 0, nr - 1);
}
/***********************************************************
 *  Allow user to change EPS.
 *  Modifed to do at least three trapzd() in case data is
 *  noisy.  9/26/1994 Lihong Wang.
 ****/

#define JMAX 30
float
qtrap(
      float (*func) (float),
      float a,
      float b,
      float EPS)
{
  int         j;
  float       s, s_old = 0;

  for (j = 1; j <= JMAX; j++) {
    s = trapzd(func, a, b, j);
    if (j <= 3 || fabs(s - s_old) > EPS * fabs(s_old))
      s_old = s;
    else
      break;
  }
  return (s);
}
#undef JMAX

/***********************************************************
 *	Modified Bessel function exp(-x) I0(x), for x >=0.
 *	We modified from the original bessi0(). Instead of
 *	I0(x) itself, it returns I0(x) exp(-x).
 ****/
double
BessI0(double x)
{
  double      ax, ans;
  double      y;

  if ((ax = fabs(x)) < 3.75) {
    y = x / 3.75;
    y *= y;
    ans = exp(-ax) * (1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
	      + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2))))));
  } else {
    y = 3.75 / ax;
    ans = (1 / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
		+ y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
	    + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
					       + y * 0.392377e-2))))))));
  }
  return ans;
}
