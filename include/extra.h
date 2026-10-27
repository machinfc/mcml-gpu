/***********************************************************
 *	Numerical quadrature, modified Bessel function I0(x)*exp(-x),
 *	and memory management helpers for finite-beam convolution.
 ****/

#ifndef EXTRA_H
#define EXTRA_H

#include "conv.h"

struct XY {
  double x, y;
};

struct XYpairs {
  double x, y;
  struct XYpairs *next;
};

typedef struct XYpairs *PairList;

double ZMin(double **Z, long IXmax, long IYmax, double Dx, double Dy,
            struct XY *Pmin_Ptr);
double ZMax(double **Z, long IXmax, long IYmax, double Dx, double Dy,
            struct XY *Pmax_Ptr);

#endif /* EXTRA_H */
