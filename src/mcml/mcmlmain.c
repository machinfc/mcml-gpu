/***********************************************************
 *  Copyright Univ. of Texas M.D. Anderson Cancer Center
 *  1992.
 *
 *	main program for Monte Carlo simulation of photon
 *	distribution in multi-layered turbid media.
 *
 ****/

/****
 *	THINKCPROFILER is defined to generate profiler calls in 
 *	Think C. If 1, remember to turn on "Generate profiler 
 *	calls" in the options menu. 
 ****/
#define THINKCPROFILER 0	

/* GNU cc does not support difftime() and CLOCKS_PER_SEC.*/
#define GNUCC 0

#if THINKCPROFILER
#include <profile.h>
#include <console.h>
#endif

#include "mcml.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/*	Declare before they are used in main(). */
FILE *GetFile(char *);
short ReadNumRuns(FILE* );
void ReadParm(FILE* , InputStruct * );
void CheckParm(FILE* , InputStruct * );
void InitOutputData(InputStruct, OutStruct *);
void FreeData(InputStruct, OutStruct *);
double Rspecular(LayerStruct * );
void LaunchPhoton(double, LayerStruct *, PhotonStruct *);
void HopDropSpin(InputStruct  *,PhotonStruct *,OutStruct *);
void SumScaleResult(InputStruct, OutStruct *);
void WriteResult(InputStruct, OutStruct, char *);


/***********************************************************
 *	If F = 0, reset the clock and return 0.
 *
 *	If F = 1, pass the user time to Msg and print Msg on 
 *	screen, return the real time since F=0. 
 *
 *	If F = 2, same as F=1 except no printing.  
 *
 *	Note that clock() and time() return user time and real 
 *	time respectively.
 *	User time is whatever the system allocates to the 
 *	running of the program; 
 *	real time is wall-clock time.  In a time-shared system,
 *	they need not be the same.
 *	
 *	clock() only hold 16 bit integer, which is about 32768 
 *	clock ticks.
 ****/
time_t PunchTime(char F, char *Msg)
{
#if GNUCC
  return(0);
#else
#ifdef _OPENMP
  static double  wt0;	/* high-resolution wall-clock reference. */
#else
  static clock_t ut0;	/* user time reference. */
#endif
  static time_t  rt0;	/* real time reference. */
  double secs;
  char s[STRLEN];
  
  if(F==0) {
    rt0 = time(NULL);
#ifdef _OPENMP
    wt0 = omp_get_wtime();
#else
    ut0 = clock();
#endif
    return(0);
  }
  else if(F==1)  {
#ifdef _OPENMP
    secs = omp_get_wtime() - wt0;
#else
    secs = (clock() - ut0)/(double)CLOCKS_PER_SEC;
#endif
    if (secs<0) secs=0;	/* clock() can overflow. */
    sprintf(s, "Wall/User time: %8.3lf sec = %8.2lf hr.  %s\n", 
	    secs, secs/3600.0, Msg);
    puts(s);
    strcpy(Msg, s);
    return(difftime(time(NULL), rt0));
  }
  else if(F==2) return(difftime(time(NULL), rt0));
  else return(0);
#endif
}

/***********************************************************
 *	Print the current time and the estimated finishing time.
 *
 *	P1 is the number of computed photon packets.
 *	Pt is the total number of photon packets.
 ****/
void PredictDoneTime(long P1, long Pt)	
{
  time_t now, done_time;
  struct tm *date;
  char s[80];
  
  now = time(NULL);
  date = localtime(&now);
  strftime(s, 80, "%H:%M %x", date);
  printf("Now %s, ", s);
  
  done_time = now + 
			(time_t) (PunchTime(2,"")/(double)P1*(Pt-P1));
  date = localtime(&done_time);
  strftime(s, 80, "%H:%M %x", date);
  printf("End %s\n", s);
}

/***********************************************************
 *	Report time and write results. 
 ****/
void ReportResult(InputStruct In_Parm, OutStruct Out_Parm)
{
  char time_report[STRLEN];
  
  strcpy(time_report, " Simulation time of this run.");
  PunchTime(1, time_report);

  SumScaleResult(In_Parm, &Out_Parm);
  WriteResult(In_Parm, Out_Parm, time_report);
}

static Boolean UseGPU = 0;

/***********************************************************
 *	Get the file name of the input data file from the 
 *	argument to the command line, along with optional
 *	execution flags (--gpu, --cpu, --threads <N> / -t <N>).
 ****/
void GetFnameFromArgv(int argc,
					  char * argv[],
					  char * input_filename)
{
  int i;
  const char *env_backend = getenv("MCML_BACKEND");
  input_filename[0] = '\0';

  if (env_backend != NULL &&
      (strcmp(env_backend, "cuda") == 0 || strcmp(env_backend, "gpu") == 0 ||
       strcmp(env_backend, "CUDA") == 0 || strcmp(env_backend, "GPU") == 0)) {
    UseGPU = 1;
  }

  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--gpu") == 0 || strcmp(argv[i], "--cuda") == 0) {
      UseGPU = 1;
    } else if (strcmp(argv[i], "--cpu") == 0 || strcmp(argv[i], "--omp") == 0) {
      UseGPU = 0;
    } else if ((strcmp(argv[i], "--threads") == 0 ||
                strcmp(argv[i], "-t") == 0) &&
               i + 1 < argc) {
      int nthreads = atoi(argv[++i]);
#ifdef _OPENMP
      if (nthreads > 0)
        omp_set_num_threads(nthreads);
#else
      (void)nthreads;
#endif
    } else if (argv[i][0] != '-') {
      strcpy(input_filename, argv[i]);
    }
  }
} 

    
/***********************************************************
 *	Execute Monte Carlo simulation for one independent run.
 ****/
void DoOneRun(short NumRuns, InputStruct *In_Ptr)
{
  long i_photon;	
	/* index to photon. register for speed.*/
  OutStruct out_parm;		/* distribution of photons.*/
  PhotonStruct photon;
  long num_photons = In_Ptr->num_photons, photon_rep=10;
  Boolean ran_on_gpu = 0;

#if THINKCPROFILER
  InitProfile(200,200); cecho2file("prof.rpt",0, stdout);
#endif
    
  InitOutputData(*In_Ptr, &out_parm);
  out_parm.Rsp = Rspecular(In_Ptr->layerspecs);	
  i_photon = num_photons;

#ifdef ENABLE_CUDA
  if (UseGPU) {
    InitDeviceCUDA();
  }
#endif

  PunchTime(0, "");

#ifdef ENABLE_CUDA
  if (UseGPU) {
    ran_on_gpu = RunSimulationCUDA(In_Ptr, &out_parm);
    if (!ran_on_gpu) {
      fprintf(stderr, "[WARN] CUDA device unavailable; falling back to CPU.\n");
    }
  }
#endif

  if (!ran_on_gpu) {
#ifdef _OPENMP
    (void)NumRuns;
    (void)photon_rep;
    printf("[Backend] OpenMP CPU (%d threads)\n", omp_get_max_threads());
    #pragma omp parallel for schedule(dynamic, 1024) private(photon)
    for(i_photon = 0; i_photon < num_photons; i_photon++) {
      LaunchPhoton(out_parm.Rsp, In_Ptr->layerspecs, &photon);
      do  HopDropSpin(In_Ptr, &photon, &out_parm);
      while (!photon.dead);
    }
#else
    printf("[Backend] Serial CPU (1 thread)\n");
    do {
      if(num_photons - i_photon == photon_rep) {
        printf("%ld photons & %hd runs left, ", i_photon, NumRuns);
        PredictDoneTime(num_photons - i_photon, num_photons);
        photon_rep *= 10;
      }
      LaunchPhoton(out_parm.Rsp, In_Ptr->layerspecs, &photon);
      do  HopDropSpin(In_Ptr, &photon, &out_parm);
      while (!photon.dead);
    } while(--i_photon);
#endif
  }
    
#if THINKCPROFILER
  exit(0);
#endif
    
  ReportResult(*In_Ptr, out_parm);
  FreeData(*In_Ptr, &out_parm);
}

/***********************************************************
 *	The argument to the command line is filename, if any.
 *	Macintosh does not support command line.
 ****/
int
main(int argc, char *argv[]) 
{
  char input_filename[STRLEN];
  FILE *input_file_ptr;
  short num_runs;	/* number of independent runs. */
  InputStruct in_parm;

  ShowVersion("Version 1.2.2, 2000");
  GetFnameFromArgv(argc, argv, input_filename);
  input_file_ptr = GetFile(input_filename);
  CheckParm(input_file_ptr, &in_parm);	
  num_runs = ReadNumRuns(input_file_ptr);
  
  while(num_runs--)  {
    ReadParm(input_file_ptr, &in_parm);
	DoOneRun(num_runs, &in_parm);
  }
  
  fclose(input_file_ptr);
  return(0);
}
