#include "mpi.h"
#include "fem.h"

void AcousticRun3d(Mesh *mesh, double FinalTime, double dt){
  
  double time = 0;
  int    INTRK, tstep=0;

  double mpitime0 = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  
  int Nsteps = FinalTime/dt;
  dt = FinalTime/Nsteps;

    /* unload data from GPU */
  void gpu_get_data3d(int K,
		      double *d_velX, double *d_velY, double *d_velZ,
		      double *d_pres);
  
  #if 0
  double *velXEA = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *velYEA = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *velZEA = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *presEA = (double*) calloc(mesh->K*p_Np, sizeof(double));

  double * presRecvPoint = (double*) calloc(Nsteps, sizeof(double));
  int kEA = 4000; /* [EA] Choose element of receiver point */
  int nEA = 0; /* [EA] Choose point in that element to be receiver point*/
  #endif
  
  /* outer time step loop  */
  for(tstep=0;tstep<Nsteps;++tstep){

    for (INTRK=1; INTRK<=5; ++INTRK) {
      
      /* compute rhs of AcousticGPU's equations */
      const float fdt = dt;
      const float fa = (float)mesh->rk4a[INTRK-1];
      const float fb = (float)mesh->rk4b[INTRK-1];

      AcousticKernel3d(mesh, fa, fb, fdt);
    }

    /* [EA] If the solution at receiver point for every time-step is wanted this is where one would implement this. */
    /* [EA]
      gpu_get_data3d(mesh->K, velXEA, velYEA, velZEA, presEA); to get velX, velY, velZ and pres. 
      These are all 1D arrays with (K x Np). 
      See code below for example of pulling out point nEA from element kEA.
      (Currently only done through hard-coding).
      
      Note:
      This will severly slow down the solver! The data has to be copied from GPU to CPU after each time-step!
      
      
      Pseudo-code for receiver point:
      REQUIRE TO BE COMPUTED EARLIER: 
       	- k_Recv(Element containing receiver point)
        - PointWiseInterpolator (Vector containing basis functions for interpolation)
      
      1) Transfer from GPU to CPU:
      gpu_get_data3d(mesh->K, velXEA, velYEA, velZEA, presEA); 
      
      2) Interpolate:
      interpolatedPoint = 0
      for n = 1:p_Np
      	int idEA = n + p_Np*k_Recv;
      	interpolatedPoint += PointWiseInterpolator[n]*presEA[idEA]
      end
      3) Save interpolated point for time-step tstep:
      presRecvPoint[tstep] = interpolatedPoint
     */

    #if 0
    gpu_get_data3d(mesh->K, velXEA, velYEA, velZEA, presEA);
   
    int idEA = nEA + p_Np*kEA;
    presRecvPoint[tstep] = presEA[idEA];
    #endif
    /* [EA] Prints every 100 time-steps */
    if(tstep % 100 == 0 ){
      printf("Time-step %d out of %d - Current time:%lf\n", tstep, Nsteps, time);
      
    }
    time += dt;     /* increment current time */
  }
  

  int Kloc = mesh->K;
  
  double mpitime1 = MPI_Wtime();
  
  double time_total = mpitime1-mpitime0;
  
  MPI_Barrier(MPI_COMM_WORLD);

  #if 0
  free(velXEA);
  free(velYEA);
  free(velZEA);
  free(presEA);


  FILE * fileEA = fopen("data/presRecvPoint.txt","w");
  fprintf(fileEA,"%.15lf %.15lf %.15lf\n",mesh->x[kEA][nEA],mesh->y[kEA][nEA],mesh->z[kEA][nEA]);
  for(int i = 0; i < Nsteps; i++){
    fprintf(fileEA, "%.15lf ",presRecvPoint[i]);
  }
  fclose(fileEA);

  free(presRecvPoint);
  #endif
	 
}

