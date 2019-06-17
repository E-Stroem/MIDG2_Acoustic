#include <math.h>
#include "mpi.h"
#include "fem.h"


int main(int argc, char **argv){

  Mesh *mesh;
  int procid, nprocs, maxNv;
  int k,n, sk=0;
  
  // Acoustic settings
  double sloc[3] = {0.5, 0.5, 0.5}; /* [EA] Source location*/
  double sxyzSQ = 0.1*0.1; /* [EA] Width of the initial pulse squared*/
  double c_Acoustic = 343.0; /* [EA] Speed of sound, if changed, 
                              change in SurfaceKernel too, (rho is also found in SurfaceKernel)*/
  double FinalTime = 0.01; /* [EA] Final time*/
  double CFL = 0.5; 

  /* initialize MPI */
  MPI_Init(&argc, &argv);

  /* assign gpu */
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  printf("procid=%d , nprocs=%d\n", procid, nprocs);


  /* (parallel) read part of fem mesh from file */
  mesh = ReadMesh3d(argv[1]);

  /* perform load balancing */
  //LoadBalance3d(mesh); 
  /* [EA] LoadBalance3d, not relevant for GPU (as nproc = 1). Can lead to problems for small meshes on many core CPU runs. 
  Will change ordering of elements on procs i.e. also output */

  /* find element-element connectivity */
  FacePair3d(mesh, &maxNv);

  /* perform start up */
  StartUp3d(mesh);

  /* field storage (double) */
  double *velX = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *velY = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *velZ = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *pres = (double*) calloc(mesh->K*p_Np, sizeof(double));
  

  /* initial conditions */
  /* [EA] Gaussian pulse with width sqrt(sxyzSQ) */
  for(k=0;k<mesh->K;++k){
    for(n=0;n<p_Np;++n) {
      velX[sk] = 0;
      velY[sk] = 0;
      velZ[sk] = 0;

      pres[sk] = (mesh->x[k][n]-sloc[0])*(mesh->x[k][n]-sloc[0]);
      pres[sk] += (mesh->y[k][n]-sloc[1])*(mesh->y[k][n]-sloc[1]);
      pres[sk] += (mesh->z[k][n]-sloc[2])*(mesh->z[k][n]-sloc[2]);
      pres[sk] /= sxyzSQ;
      pres[sk] = exp(-pres[sk]);
      ++sk;
    }
  }

  double dt, gdt;

  /* initialize OCCA info */
  double InitOCCA3d(Mesh *mesh, int Nfields);
  dt = InitOCCA3d(mesh, p_Nfields);

  /* load data onto GPU */
  gpu_set_data3d(mesh->K, velX, velY, velZ, pres);
  
  MPI_Allreduce(&dt, &gdt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  dt = CFL*gdt / (c_Acoustic*(p_N+1.0)*(p_N+1.0));

  if(!procid){
    printf("dt = %.15lf\n", dt);
    printf("FinalTime=%g\n", FinalTime);
  }
  
  double startTime, endTime;
  
  startTime = MPI_Wtime();
  /* solve */
  AcousticRun3d(mesh, FinalTime, dt); 

  MPI_Barrier(MPI_COMM_WORLD);
  endTime = MPI_Wtime();
  if(!procid){
    printf("Time to solve:%f\n",endTime-startTime);
  }

  /* unload data from GPU */
  void gpu_get_data3d(int K,
		      double *d_velX, double *d_velY, double *d_velZ,
		      double *d_pres);
  gpu_get_data3d(mesh->K, velX, velY, velZ, pres);


  /* [EA] Outputs the (x,y,z)-coordinates and pressure into data/ */
  FILE * xFP, * yFP, * zFP, * pFP;
  char xFN[50] = "data/x.txt";
  char yFN[50] = "data/y.txt";
  char zFN[50] = "data/z.txt";
  char pFN[50] = "data/p.txt";  
  for(int i = 0; i < nprocs; i++){
    if(procid == i){
      if(procid == 0){
          xFP = fopen(xFN, "w");
          yFP = fopen(yFN, "w");
          zFP = fopen(zFN, "w");
          pFP = fopen(pFN, "w");
      } else {
          xFP = fopen(xFN, "a");
          yFP = fopen(yFN, "a");
          zFP = fopen(zFN, "a");
          pFP = fopen(pFN, "a");
      }
      for(k=0;k<mesh->K;++k) {
        for(n=0;n<p_Np;++n){
          int id = n + p_Np*k;
          fprintf(xFP, "%.15lf ",mesh->x[k][n]);
          fprintf(yFP, "%.15lf ",mesh->y[k][n]);
          fprintf(zFP, "%.15lf ",mesh->z[k][n]);
          fprintf(pFP, "%.15lf ",pres[id]);
        }
        fprintf(xFP, "\n");
        fprintf(yFP, "\n");
        fprintf(zFP, "\n");
        fprintf(pFP, "\n");
      }

      fclose(xFP);
      fclose(yFP);
      fclose(zFP);
      fclose(pFP);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }


  /* nicely stop MPI */
  MPI_Finalize();

  /* end game */
  exit(0);
}
