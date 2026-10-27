/****************************************************************
 *	Finite-Beam Laser Convolution Engine (Flat-Top & Gaussian)
 *	for MCML Cylindrical Impulse-Response Outputs (.mco).
 *
 *	Refactored to C99 (fgets safety, 32-bit grid indices, and
 *	corrected A_rz interpolation & Gaussian beam integration).
 ****/

#include "conv.h"
#include "extra.h"

#define IBMPC 0
#define GAUSSLIMIT 4.0

struct Node {
  float x, y;
  struct Node *left, *right;
};

typedef struct Node *LINK;
typedef LINK TREE;

struct {
  double r;
  int iz, ia;
  InputConvStruct *in_ptr;
  OutConvStruct *out_ptr;
  TREE tree;
} ConvVar;

static void ReadConvLine(char *buf, size_t size)
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

void GetFlatBeam(BeamStruct *Beam_Ptr)
{
  Beam_Ptr->type = flat;
  printf("Total energy of the flat beam [J]: ");
  Beam_Ptr->P = GetFloat(FLT_MIN, FLT_MAX);
  printf("Radius of the flat beam [cm]: ");
  Beam_Ptr->R = GetFloat(FLT_MIN, FLT_MAX);
  printf("Total power: %8.2lg J, and radius: %8.2lg cm.\n", Beam_Ptr->P,
         Beam_Ptr->R);
}

void GetGaussianBeam(BeamStruct *Beam_Ptr)
{
  Beam_Ptr->type = gaussian;
  printf("Total energy of the Gaussian beam [J]: ");
  Beam_Ptr->P = GetFloat(FLT_MIN, FLT_MAX);
  printf("1/e2 Radius of the Gaussian beam [cm]: ");
  Beam_Ptr->R = GetFloat(FLT_MIN, FLT_MAX);
  printf("Total power: %8.2lg J, and radius: %8.2lg cm.\n", Beam_Ptr->P,
         Beam_Ptr->R);
}

void LaserBeam(BeamStruct *Beam_Ptr, OutConvStruct *Out_Ptr)
{
  ConvStruct null_conved = NULLCONVSTRUCT;
  char cmd_str[STRLEN];

  printf("Beam profile: f=flat, g=Gaussian, q=quit: ");
  do {
    ReadConvLine(cmd_str, sizeof(cmd_str));
  } while (!strlen(cmd_str));

  switch (toupper((unsigned char)cmd_str[0])) {
  case 'F':
    Out_Ptr->conved = null_conved;
    GetFlatBeam(Beam_Ptr);
    break;
  case 'G':
    Out_Ptr->conved = null_conved;
    GetGaussianBeam(Beam_Ptr);
    break;
  default:
    break;
  }
}

void ConvResolution(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  ConvStruct null_conved = NULLCONVSTRUCT;

  if (!Out_Ptr->allocated) {
    puts("...No mcml output data to work with");
    return;
  }
  printf("Current resolution: %8.2lg cm and number of points: %d\n",
         In_Ptr->drc, In_Ptr->nrc);
  printf("Input resolution in r direction [cm]: ");
  In_Ptr->drc = GetFloat(FLT_MIN, FLT_MAX);
  printf("Input number of points in r direction: ");
  In_Ptr->nrc = (int)GetShort(1, SHRT_MAX);

  Out_Ptr->conved = null_conved;
  FreeConvData(In_Ptr, Out_Ptr);
  AllocConvData(In_Ptr, Out_Ptr);
}

void ConvError(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  ConvStruct null_conved = NULLCONVSTRUCT;

  printf("Current relative convolution error is %8.2g (0.001-0.1 recommended): ",
         In_Ptr->eps);
  In_Ptr->eps = GetFloat(FLT_MIN, 1.0f);
  Out_Ptr->conved = null_conved;
}

/*********************** Binary Tree *****************************/
LINK FillNode(float x, float y)
{
  LINK l = (LINK)malloc(sizeof(struct Node));
  if (l) {
    l->x = x;
    l->y = y;
    l->left = l->right = NULL;
  }
  return l;
}

void InsertNode(TREE *TreePtr, float x, float y)
{
  LINK l1 = NULL;
  LINK l2 = *TreePtr;
  while (l2 != NULL) {
    l1 = l2;
    if (x < l2->x)
      l2 = l2->left;
    else
      l2 = l2->right;
  }

  if (l1 == NULL)
    *TreePtr = FillNode(x, y);
  else if (x < l1->x)
    l1->left = FillNode(x, y);
  else
    l1->right = FillNode(x, y);
}

LINK SearchNode(TREE Tree, float x)
{
  LINK l = Tree;
  while (l != NULL) {
    if (x < l->x)
      l = l->left;
    else if (x > l->x)
      l = l->right;
    else
      return l;
  }
  return NULL;
}

void FreeTree(TREE Tree)
{
  if (Tree) {
    FreeTree(Tree->left);
    FreeTree(Tree->right);
    free(Tree);
  }
}

/*********************** Integration *****************************/
double ITheta(double r, double r2, double R)
{
  double temp;

  if (R >= r + r2)
    temp = 1.0;
  else if (fabs(r - r2) <= R) {
    temp = (r * r + r2 * r2 - R * R) / (2.0 * r * r2);
    if (fabs(temp) > 1.0)
      temp = SIGN(temp);
    temp = acos(temp) / PI;
  } else
    temp = 0.0;

  return temp;
}

double ExpBessI0(double r, double r2, double R)
{
  double _RR = 1.0 / (R * R);
  double x = 4.0 * r * r2 * _RR;
  double y = 2.0 * (r2 * r2 + r * r) * _RR;

  return exp(-y + x) * BessI0(x);
}

double A_rzInterp(double **A_rz, double r2)
{
  double ir2, A_lo, A_hi, A_at_r2;
  int iz, ir2lo, nr = ConvVar.in_ptr->nr;

  ir2 = r2 / ConvVar.in_ptr->dr;
  iz = ConvVar.iz;
  if (nr < 3)
    A_at_r2 = A_rz[0][iz];
  else if (ir2 < nr - 1.5) {
    ir2lo = MAX(0, (int)(ir2 - 0.5));
    A_lo = A_rz[ir2lo][iz];
    A_hi = A_rz[ir2lo + 1][iz];
    A_at_r2 = A_lo + (A_hi - A_lo) * (ir2 - ir2lo - 0.5);
  } else {
    ir2lo = nr - 3;
    A_lo = A_rz[ir2lo][iz];
    A_hi = A_rz[ir2lo + 1][iz];
    if (A_lo >= A_hi)
      A_at_r2 = A_lo + (A_hi - A_lo) * (ir2 - ir2lo - 0.5);
    else
      A_at_r2 = 0.0;
  }

  return MAX(0.0, A_at_r2);
}

double RT_raInterp(double **RT_ra, double r2)
{
  double ir2, RT_lo, RT_hi, RT_at_r2;
  int ia, ir2lo, nr = ConvVar.in_ptr->nr;

  ir2 = r2 / ConvVar.in_ptr->dr;
  ia = ConvVar.ia;
  if (nr < 3)
    RT_at_r2 = RT_ra[0][ia];
  else if (ir2 < nr - 1.5) {
    ir2lo = MAX(0, (int)(ir2 - 0.5));
    RT_lo = RT_ra[ir2lo][ia];
    RT_hi = RT_ra[ir2lo + 1][ia];
    RT_at_r2 = RT_lo + (RT_hi - RT_lo) * (ir2 - ir2lo - 0.5);
  } else {
    ir2lo = nr - 3;
    RT_lo = RT_ra[ir2lo][ia];
    RT_hi = RT_ra[ir2lo + 1][ia];
    if (RT_lo >= RT_hi)
      RT_at_r2 = RT_lo + (RT_hi - RT_lo) * (ir2 - ir2lo - 0.5);
    else
      RT_at_r2 = 0.0;
  }

  return MAX(0.0, RT_at_r2);
}

double RT_rInterp(double *RT_r, double r2)
{
  double ir2, RT_lo, RT_hi, RT_at_r2;
  int ir2lo, nr = ConvVar.in_ptr->nr;

  ir2 = r2 / ConvVar.in_ptr->dr;
  if (nr < 3)
    RT_at_r2 = RT_r[0];
  else if (ir2 < nr - 1.5) {
    ir2lo = MAX(0, (int)(ir2 - 0.5));
    RT_lo = RT_r[ir2lo];
    RT_hi = RT_r[ir2lo + 1];
    RT_at_r2 = RT_lo + (RT_hi - RT_lo) * (ir2 - ir2lo - 0.5);
  } else {
    ir2lo = nr - 3;
    RT_lo = RT_r[ir2lo];
    RT_hi = RT_r[ir2lo + 1];
    if (RT_lo >= RT_hi)
      RT_at_r2 = RT_lo + (RT_hi - RT_lo) * (ir2 - ir2lo - 0.5);
    else
      RT_at_r2 = 0.0;
  }

  return MAX(0.0, RT_at_r2);
}

float A_rzFGIntegrand(float r2)
{
  float f;
  double R, r, A_at_r2;
  LINK link;

  A_at_r2 = A_rzInterp(ConvVar.out_ptr->A_rz, r2);
  R = ConvVar.in_ptr->beam.R;
  r = ConvVar.r;
  if ((link = SearchNode(ConvVar.tree, r2)) != NULL)
    f = link->y;
  else {
    if (ConvVar.in_ptr->beam.type == flat)
      f = (float)ITheta(r, r2, R);
    else
      f = (float)ExpBessI0(r, r2, R);
    InsertNode(&ConvVar.tree, r2, f);
  }

  f *= (float)(A_at_r2 * r2);
  return f;
}

float Rd_raFGIntegrand(float r2)
{
  float f;
  double R, r, Rd_at_r2;
  LINK link;

  Rd_at_r2 = RT_raInterp(ConvVar.out_ptr->Rd_ra, r2);
  R = ConvVar.in_ptr->beam.R;
  r = ConvVar.r;
  if ((link = SearchNode(ConvVar.tree, r2)) != NULL)
    f = link->y;
  else {
    if (ConvVar.in_ptr->beam.type == flat)
      f = (float)ITheta(r, r2, R);
    else
      f = (float)ExpBessI0(r, r2, R);
    InsertNode(&ConvVar.tree, r2, f);
  }

  f *= (float)(Rd_at_r2 * r2);
  return f;
}

float Rd_rFGIntegrand(float r2)
{
  float f;
  double R, r, Rd_at_r2;

  Rd_at_r2 = RT_rInterp(ConvVar.out_ptr->Rd_r, r2);
  R = ConvVar.in_ptr->beam.R;
  r = ConvVar.r;
  if (ConvVar.in_ptr->beam.type == flat)
    f = (float)(Rd_at_r2 * ITheta(r, r2, R) * r2);
  else
    f = (float)(Rd_at_r2 * ExpBessI0(r, r2, R) * r2);

  return f;
}

float Tt_raFGIntegrand(float r2)
{
  float f;
  double R, r, Tt_at_r2;
  LINK link;

  Tt_at_r2 = RT_raInterp(ConvVar.out_ptr->Tt_ra, r2);
  R = ConvVar.in_ptr->beam.R;
  r = ConvVar.r;
  if ((link = SearchNode(ConvVar.tree, r2)) != NULL)
    f = link->y;
  else {
    if (ConvVar.in_ptr->beam.type == flat)
      f = (float)ITheta(r, r2, R);
    else
      f = (float)ExpBessI0(r, r2, R);
    InsertNode(&ConvVar.tree, r2, f);
  }

  f *= (float)(Tt_at_r2 * r2);
  return f;
}

float Tt_rFGIntegrand(float r2)
{
  float f;
  double R, r, Tt_at_r2;

  Tt_at_r2 = RT_rInterp(ConvVar.out_ptr->Tt_r, r2);
  R = ConvVar.in_ptr->beam.R;
  r = ConvVar.r;
  if (ConvVar.in_ptr->beam.type == flat)
    f = (float)(Tt_at_r2 * ITheta(r, r2, R) * r2);
  else
    f = (float)(Tt_at_r2 * ExpBessI0(r, r2, R) * r2);

  return f;
}

double FlatIntegration(float (*Func)(float))
{
  double rc = ConvVar.r;
  double R = ConvVar.in_ptr->beam.R;
  double b_max = (ConvVar.in_ptr->nr - 0.5) * ConvVar.in_ptr->dr;
  double a = MAX(0.0, rc - R);
  double b = MIN(b_max, rc + R);

  if (a >= b)
    return 0.0;
  else
    return (double)qtrap(Func, (float)a, (float)b, ConvVar.in_ptr->eps);
}

double GaussIntegration(float (*Func)(float))
{
  double rc = ConvVar.r;
  double R = ConvVar.in_ptr->beam.R;
  double b_max = (ConvVar.in_ptr->nr - 0.5) * ConvVar.in_ptr->dr;
  double a = MAX(0.0, rc - GAUSSLIMIT * R);
  double b = MIN(b_max, rc + GAUSSLIMIT * R);

  if (a >= b)
    return 0.0;
  else
    return (double)qtrap(Func, (float)a, (float)b, ConvVar.in_ptr->eps);
}

void ConvA_rz(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  int irc, iz;
  double rc, P = In_Ptr->beam.P, R = In_Ptr->beam.R;

  ConvVar.in_ptr = In_Ptr;
  ConvVar.out_ptr = Out_Ptr;

  for (irc = 0; irc < In_Ptr->nrc; irc++) {
    rc = (irc + 0.5) * In_Ptr->drc;
    ConvVar.r = rc;
    ConvVar.tree = NULL;
    for (iz = 0; iz < In_Ptr->nz; iz++) {
      ConvVar.iz = iz;
      if (In_Ptr->beam.type == flat)
        Out_Ptr->A_rzc[irc][iz] =
            2.0 * P / (R * R) * FlatIntegration(A_rzFGIntegrand);
      else
        Out_Ptr->A_rzc[irc][iz] =
            4.0 * P / (R * R) * GaussIntegration(A_rzFGIntegrand);
    }
    FreeTree(ConvVar.tree);
  }

  Out_Ptr->conved.A_rz = 1;
}

void ConvRd_ra(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  int irc, ia;
  double rc, P = In_Ptr->beam.P, R = In_Ptr->beam.R;

  ConvVar.in_ptr = In_Ptr;
  ConvVar.out_ptr = Out_Ptr;

  for (irc = 0; irc < In_Ptr->nrc; irc++) {
    rc = (irc + 0.5) * In_Ptr->drc;
    ConvVar.r = rc;
    ConvVar.tree = NULL;
    for (ia = 0; ia < In_Ptr->na; ia++) {
      ConvVar.ia = ia;
      if (In_Ptr->beam.type == flat)
        Out_Ptr->Rd_rac[irc][ia] =
            2.0 * P / (R * R) * FlatIntegration(Rd_raFGIntegrand);
      else
        Out_Ptr->Rd_rac[irc][ia] =
            4.0 * P / (R * R) * GaussIntegration(Rd_raFGIntegrand);
    }
    FreeTree(ConvVar.tree);
  }

  Out_Ptr->conved.Rd_ra = 1;
}

void ConvRd_r(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  int irc;
  double rc, P = In_Ptr->beam.P, R = In_Ptr->beam.R;

  ConvVar.in_ptr = In_Ptr;
  ConvVar.out_ptr = Out_Ptr;

  for (irc = 0; irc < In_Ptr->nrc; irc++) {
    rc = (irc + 0.5) * In_Ptr->drc;
    ConvVar.r = rc;
    if (In_Ptr->beam.type == flat)
      Out_Ptr->Rd_rc[irc] =
          2.0 * P / (R * R) * FlatIntegration(Rd_rFGIntegrand);
    else
      Out_Ptr->Rd_rc[irc] =
          4.0 * P / (R * R) * GaussIntegration(Rd_rFGIntegrand);
  }

  Out_Ptr->conved.Rd_r = 1;
}

void ConvTt_ra(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  int irc, ia;
  double rc, P = In_Ptr->beam.P, R = In_Ptr->beam.R;

  ConvVar.in_ptr = In_Ptr;
  ConvVar.out_ptr = Out_Ptr;

  for (irc = 0; irc < In_Ptr->nrc; irc++) {
    rc = (irc + 0.5) * In_Ptr->drc;
    ConvVar.r = rc;
    ConvVar.tree = NULL;
    for (ia = 0; ia < In_Ptr->na; ia++) {
      ConvVar.ia = ia;
      if (In_Ptr->beam.type == flat)
        Out_Ptr->Tt_rac[irc][ia] =
            2.0 * P / (R * R) * FlatIntegration(Tt_raFGIntegrand);
      else
        Out_Ptr->Tt_rac[irc][ia] =
            4.0 * P / (R * R) * GaussIntegration(Tt_raFGIntegrand);
    }
    FreeTree(ConvVar.tree);
  }

  Out_Ptr->conved.Tt_ra = 1;
}

void ConvTt_r(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  int irc;
  double rc, P = In_Ptr->beam.P, R = In_Ptr->beam.R;

  ConvVar.in_ptr = In_Ptr;
  ConvVar.out_ptr = Out_Ptr;

  for (irc = 0; irc < In_Ptr->nrc; irc++) {
    rc = (irc + 0.5) * In_Ptr->drc;
    ConvVar.r = rc;
    if (In_Ptr->beam.type == flat)
      Out_Ptr->Tt_rc[irc] =
          2.0 * P / (R * R) * FlatIntegration(Tt_rFGIntegrand);
    else
      Out_Ptr->Tt_rc[irc] =
          4.0 * P / (R * R) * GaussIntegration(Tt_rFGIntegrand);
  }

  Out_Ptr->conved.Tt_r = 1;
}

static char *NextDataLine(FILE *fp, char *buf, size_t size)
{
  while (fgets(buf, (int)size, fp) != NULL) {
    char *p = buf;
    while (*p && isspace((unsigned char)*p))
      p++;
    if (*p == '\0' || *p == '#')
      continue;
    return p;
  }
  return NULL;
}

Boolean ReadMcoFile(const char *mco_fname, InputConvStruct *In_Ptr,
                    OutConvStruct *Out_Ptr)
{
  FILE *fp = fopen(mco_fname, "r");
  if (!fp)
    return 0;

  char buf[1024];
  char *line;

  memset(In_Ptr, 0, sizeof(InputConvStruct));
  memset(Out_Ptr, 0, sizeof(OutConvStruct));
  strncpy(In_Ptr->in_fname, mco_fname, STRLEN - 1);

  /* 1. Version (A1) */
  if (!NextDataLine(fp, buf, sizeof(buf))) {
    fclose(fp);
    return 0;
  }

  /* 2. Seek InParm section */
  while ((line = NextDataLine(fp, buf, sizeof(buf))) != NULL) {
    if (strncmp(line, "InParm", 6) == 0)
      break;
  }
  if (!line) {
    fclose(fp);
    return 0;
  }

  line = NextDataLine(fp, buf, sizeof(buf));
  if (!line) {
    fclose(fp);
    return 0;
  }
  sscanf(line, "%255s %c", In_Ptr->out_fname, &In_Ptr->out_fformat);

  line = NextDataLine(fp, buf, sizeof(buf));
  sscanf(line, "%ld", &In_Ptr->num_photons);

  line = NextDataLine(fp, buf, sizeof(buf));
  sscanf(line, "%lf %lf", &In_Ptr->dz, &In_Ptr->dr);

  line = NextDataLine(fp, buf, sizeof(buf));
  sscanf(line, "%d %d %d", &In_Ptr->nz, &In_Ptr->nr, &In_Ptr->na);
  In_Ptr->da = 0.5 * PI / In_Ptr->na;
  In_Ptr->nrc = In_Ptr->nr;
  In_Ptr->drc = In_Ptr->dr;
  In_Ptr->eps = 0.01f;
  In_Ptr->beam.type = gaussian;
  In_Ptr->beam.P = 1.0;
  In_Ptr->beam.R = 0.1;

  line = NextDataLine(fp, buf, sizeof(buf));
  sscanf(line, "%hd", &In_Ptr->num_layers);

  In_Ptr->layerspecs = (LayerStruct *)malloc(
      (size_t)(In_Ptr->num_layers + 2) * sizeof(LayerStruct));
  memset(In_Ptr->layerspecs, 0,
         (size_t)(In_Ptr->num_layers + 2) * sizeof(LayerStruct));

  line = NextDataLine(fp, buf, sizeof(buf));
  sscanf(line, "%lf", &In_Ptr->layerspecs[0].n);

  double z = 0.0;
  for (int i = 1; i <= In_Ptr->num_layers; i++) {
    double d = 0.0;
    line = NextDataLine(fp, buf, sizeof(buf));
    sscanf(line, "%lf %lf %lf %lf %lf", &In_Ptr->layerspecs[i].n,
           &In_Ptr->layerspecs[i].mua, &In_Ptr->layerspecs[i].mus,
           &In_Ptr->layerspecs[i].g, &d);
    In_Ptr->layerspecs[i].z0 = z;
    z += d;
    In_Ptr->layerspecs[i].z1 = z;
  }
  line = NextDataLine(fp, buf, sizeof(buf));
  sscanf(line, "%lf", &In_Ptr->layerspecs[In_Ptr->num_layers + 1].n);

  /* Allocate raw & convolved arrays */
  int nz = In_Ptr->nz, nr = In_Ptr->nr, na = In_Ptr->na;
  short nl = In_Ptr->num_layers;
  Out_Ptr->Rd_ra = AllocMatrix(0, nr - 1, 0, na - 1);
  Out_Ptr->Rd_r = AllocVector(0, nr - 1);
  Out_Ptr->Rd_a = AllocVector(0, na - 1);
  Out_Ptr->A_rz = AllocMatrix(0, nr - 1, 0, nz - 1);
  Out_Ptr->A_z = AllocVector(0, nz - 1);
  Out_Ptr->A_l = AllocVector(0, nl + 1);
  Out_Ptr->Tt_ra = AllocMatrix(0, nr - 1, 0, na - 1);
  Out_Ptr->Tt_r = AllocVector(0, nr - 1);
  Out_Ptr->Tt_a = AllocVector(0, na - 1);
  AllocConvData(In_Ptr, Out_Ptr);
  Out_Ptr->allocated = 1;

  /* Parse data sections */
  while ((line = NextDataLine(fp, buf, sizeof(buf))) != NULL) {
    if (strncmp(line, "RAT", 3) == 0) {
      sscanf(NextDataLine(fp, buf, sizeof(buf)), "%lf", &Out_Ptr->Rsp);
      sscanf(NextDataLine(fp, buf, sizeof(buf)), "%lf", &Out_Ptr->Rd);
      sscanf(NextDataLine(fp, buf, sizeof(buf)), "%lf", &Out_Ptr->A);
      sscanf(NextDataLine(fp, buf, sizeof(buf)), "%lf", &Out_Ptr->Tt);
    } else if (strncmp(line, "A_l", 3) == 0) {
      for (int i = 1; i <= nl; i++)
        if (fscanf(fp, "%lf", &Out_Ptr->A_l[i]) != 1) break;
    } else if (strncmp(line, "A_z", 3) == 0 && strncmp(line, "A_rz", 4) != 0) {
      for (int iz = 0; iz < nz; iz++)
        if (fscanf(fp, "%lf", &Out_Ptr->A_z[iz]) != 1) break;
    } else if (strncmp(line, "Rd_r", 4) == 0 &&
               strncmp(line, "Rd_ra", 5) != 0) {
      for (int ir = 0; ir < nr; ir++)
        if (fscanf(fp, "%lf", &Out_Ptr->Rd_r[ir]) != 1) break;
    } else if (strncmp(line, "Rd_a", 4) == 0) {
      for (int ia = 0; ia < na; ia++)
        if (fscanf(fp, "%lf", &Out_Ptr->Rd_a[ia]) != 1) break;
    } else if (strncmp(line, "Tt_r", 4) == 0 &&
               strncmp(line, "Tt_ra", 5) != 0) {
      for (int ir = 0; ir < nr; ir++)
        if (fscanf(fp, "%lf", &Out_Ptr->Tt_r[ir]) != 1) break;
    } else if (strncmp(line, "Tt_a", 4) == 0) {
      for (int ia = 0; ia < na; ia++)
        if (fscanf(fp, "%lf", &Out_Ptr->Tt_a[ia]) != 1) break;
    } else if (strncmp(line, "A_rz", 4) == 0) {
      for (int ir = 0; ir < nr; ir++)
        for (int iz = 0; iz < nz; iz++)
          if (fscanf(fp, "%lf", &Out_Ptr->A_rz[ir][iz]) != 1) break;
    } else if (strncmp(line, "Rd_ra", 5) == 0) {
      for (int ir = 0; ir < nr; ir++)
        for (int ia = 0; ia < na; ia++)
          if (fscanf(fp, "%lf", &Out_Ptr->Rd_ra[ir][ia]) != 1) break;
    } else if (strncmp(line, "Tt_ra", 5) == 0) {
      for (int ir = 0; ir < nr; ir++)
        for (int ia = 0; ia < na; ia++)
          if (fscanf(fp, "%lf", &Out_Ptr->Tt_ra[ir][ia]) != 1) break;
    }
  }

  fclose(fp);
  return 1;
}

void FreeMcoConvData(InputConvStruct *In_Ptr, OutConvStruct *Out_Ptr)
{
  if (!Out_Ptr->allocated)
    return;

  int nz = In_Ptr->nz, nr = In_Ptr->nr, na = In_Ptr->na;
  short nl = In_Ptr->num_layers;

  free(In_Ptr->layerspecs);
  In_Ptr->layerspecs = NULL;

  FreeMatrix(Out_Ptr->Rd_ra, 0, nr - 1, 0, na - 1);
  FreeVector(Out_Ptr->Rd_r, 0, nr - 1);
  FreeVector(Out_Ptr->Rd_a, 0, na - 1);
  FreeMatrix(Out_Ptr->A_rz, 0, nr - 1, 0, nz - 1);
  FreeVector(Out_Ptr->A_z, 0, nz - 1);
  FreeVector(Out_Ptr->A_l, 0, nl + 1);
  FreeMatrix(Out_Ptr->Tt_ra, 0, nr - 1, 0, na - 1);
  FreeVector(Out_Ptr->Tt_r, 0, nr - 1);
  FreeVector(Out_Ptr->Tt_a, 0, na - 1);
  FreeConvData(In_Ptr, Out_Ptr);
  Out_Ptr->allocated = 0;
}
