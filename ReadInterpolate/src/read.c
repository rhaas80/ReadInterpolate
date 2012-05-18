#include <time.h> 
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// some macros to fix compatibility issues as long
// as 1.8.0 is in beta phase
#define H5_USE_16_API 1

#include <hdf5.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>

#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"
#include "util_Table.h"


/*****************************************************************************/
/*                           macro definitions                               */
/*****************************************************************************/
/* macro to do an HDF5 call, check its return code, and print a warning
   in case of an error */
#define CHECK_ERROR(hdf5_call)                                                \
          do                                                                  \
          {                                                                   \
            int _error_code = hdf5_call;                                      \
                                                                              \
                                                                              \
            if (_error_code < 0)                                              \
            {                                                                 \
              CCTK_VWarn (CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,\
                               "WARNING: line %d: HDF5 call '%s' returned "   \
                               "error code %d\n",                             \
                                __LINE__, #hdf5_call, _error_code);           \
            }                                                                 \
          } while (0)

#define SOLAR_MASS 1.98892e33
#define INV_R_GF 1.4768e5
#define RHO_GF 1.61620075314614e-18
#define INV_RHO_GF 6.18735016707159e17
#define INV_PRESS_GF 5.56082181777535e38

#define G_GRAV 6.673e-8 
#define MEV2KELVIN 1.1604505e10
#define ENTROPY_FACTOR 8.2476e7 // convert from erg/baryon to erg/g

#define INDEX3D(i,j,k) ((i) + lsh[0]*((j) + lsh[1]*(k)))
#define DIM(x) ((int)(sizeof(x)/sizeof(x[0])))
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

#define N_DIMS  3  
#define N_RAD   800 // number of radial zones
#define N_PHI   240 // Should be larger than 1. Recommended ~480.
#define N_THETA 60 // Should be larger than 1. Recommended ~120.
#define N_INTERP_POINTS   (N_RAD*N_THETA*N_PHI)
#define N_INPUT_ARRAYS    10  
#define N_OUTPUT_ARRAYS   N_INPUT_ARRAYS

// for loop index value at which the average shock radius is calculated. 
#define N_RAD_SHOCK   (N_RAD-2)

#define THETA_MIN 0.1e0
#define THETA_MAX (M_PI-THETA_MIN)

// declerations of functions
static double spherical_harmonics(double theta, double phi, int l, int m);
static double nlmfactor(int l, int m);
static double sh_coef(int l, int m, double *rshock, double *theta, double *phi, double dtheta, double dphi);


static void CCTK_FNAME(Turbulence_Analysis)(const cGH *cctkGH,
    const CCTK_INT *ptriteration, const CCTK_REAL *ptrtime,
    const CCTK_INT lsh[3], const CCTK_REAL origin[3], const CCTK_REAL delta[3],
    const CCTK_REAL *x, const CCTK_REAL *y, const CCTK_REAL *z, const CCTK_REAL *r,
    const CCTK_REAL *entropy, const CCTK_REAL *rho, const CCTK_REAL *velx,
    const CCTK_REAL *vely, const CCTK_REAL *velz, const CCTK_REAL *Y_e,
    CCTK_REAL *lum_local_0, CCTK_REAL *lum_local_1, 
    CCTK_REAL *lum_local_2)
{
  CCTK_REAL rad_min = 10.0e0;
  CCTK_REAL rad_shock_min = 30.0e0;
  CCTK_REAL c_light = 2.99792458e10;
  
  CCTK_REAL t_bounce = 36495.480; // bounce time for this model in code usints.
 
  /* (x,y) coordinates of interpolation points */  
  CCTK_REAL * interp_x = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * interp_y = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * interp_z = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  const void * interp_coords[N_DIMS];              /* see note above */

  /* input arrays */
  /* ... note Cactus uses Fortran storage ordering, i.e.\ X is contiguous */
  CCTK_INT input_array_type_codes[N_INPUT_ARRAYS]; /* one per input variable */
  const void *input_arrays[N_INPUT_ARRAYS];       /* see note above */
  for(int i = 0 ; i < N_INPUT_ARRAYS ; i++)
    input_array_type_codes[i] = CCTK_VARIABLE_REAL;

  /* output arrays */
  CCTK_REAL * velr_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * velt_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * velp_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));

  CCTK_REAL * rho_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * ye_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * temp_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * cs2_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * entropy_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));

  CCTK_REAL * lumloc1_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * lumloc2_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));
  CCTK_REAL * lumloc3_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));

  CCTK_REAL * tau_res = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));

  CCTK_REAL * press_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));

  CCTK_REAL e_bind = 0.0e0;
  CCTK_REAL q_heat = 0.0e0;
  CCTK_REAL tau_heat = 0.0e0;
  CCTK_REAL tau_res_av = 0.0e0;
  CCTK_REAL tau_res_mdot = 0.0e0; // residence time as define in Muller et al (2012)
  CCTK_REAL volume_gain = 0.0e0;
  CCTK_REAL mdot = 0.0e0;
  CCTK_REAL m_gain = 0.0e0;

  // vorticiy
  CCTK_REAL * vort1 = malloc(lsh[0] * lsh[1] * lsh[2] * sizeof(CCTK_REAL));
  CCTK_REAL * vort2 = malloc(lsh[0] * lsh[1] * lsh[2] * sizeof(CCTK_REAL));
  CCTK_REAL * vort3 = malloc(lsh[0] * lsh[1] * lsh[2] * sizeof(CCTK_REAL));
  CCTK_REAL * vort_cart = malloc(lsh[0] * lsh[1] * lsh[2] * sizeof(CCTK_REAL));

  CCTK_REAL * vort_spher = malloc(N_INTERP_POINTS * sizeof(CCTK_REAL));

  CCTK_REAL * f_e = malloc(N_RAD * sizeof(CCTK_REAL)); // specific entalpy flux
  CCTK_REAL * f_s = malloc(N_RAD * sizeof(CCTK_REAL)); // specific entropy flux
  CCTK_REAL * F_s = malloc(N_RAD * sizeof(CCTK_REAL)); // entropy flux
  CCTK_REAL * k_r = malloc(N_RAD * sizeof(CCTK_REAL)); // TKE r component
  CCTK_REAL * k_t = malloc(N_RAD * sizeof(CCTK_REAL)); // TKE theta component
  CCTK_REAL * k_p = malloc(N_RAD * sizeof(CCTK_REAL)); // TKE phi component
  CCTK_REAL * q_s = malloc(N_RAD * sizeof(CCTK_REAL)); // entropy variance
  CCTK_REAL * p_s = malloc(N_RAD * sizeof(CCTK_REAL)); // pressure variance

  // some auxiliary variables
  CCTK_REAL * vrsbar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * vr2bar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * vrbar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * vt2bar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * vtbar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * vp2bar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * vpbar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * s2bar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * sbar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * rhobar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * tempbar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * yebar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * pressbar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * press2bar = malloc(N_RAD * sizeof(CCTK_REAL));

  CCTK_REAL * rhosvrbar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * rhosbar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * rhovrbar = malloc(N_RAD * sizeof(CCTK_REAL));

  CCTK_REAL * cs2bar = malloc(N_RAD * sizeof(CCTK_REAL)); // sound speed squared
  CCTK_REAL * v1e2bar = malloc(N_RAD * sizeof(CCTK_REAL)); // escape velocity squared
  CCTK_REAL * massbar = malloc(N_RAD * sizeof(CCTK_REAL)); // mass inside sphere
  CCTK_REAL * ratio_sonic_1d = malloc(N_RAD * sizeof(CCTK_REAL)); // mass inside sphere

  CCTK_REAL * vortbar = malloc(N_RAD * sizeof(CCTK_REAL)); // average vorticity
  
  CCTK_REAL * sgainbar = malloc(N_RAD * sizeof(CCTK_REAL)); // average entropy in the gain region
  CCTK_REAL * rhogainbar = malloc(N_RAD * sizeof(CCTK_REAL));

  CCTK_REAL * netheatbar = malloc(N_RAD * sizeof(CCTK_REAL));

  CCTK_REAL * lumloc1bar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * lumloc2bar = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * lumloc3bar = malloc(N_RAD * sizeof(CCTK_REAL));

  // Calculate the center of the grid
  CCTK_REAL x0 = origin[0]+lsh[0]/2.0*delta[0];
  CCTK_REAL y0 = origin[1]+lsh[1]/2.0*delta[1];
  CCTK_REAL z0 = origin[2]+lsh[2]/2.0*delta[2];

  CCTK_REAL delta_v = delta[0]*delta[1]*delta[2]*pow(INV_R_GF,3)*pow(8,6);

  // Print out the coordinates of the origin.
  //fprintf(stdout,"xo, yo, zo =, %15.6E %15.6E %15.6E\n",origin[0],origin[1],origin[2]);

  // Print out the coordinates of the central grid.
  //fprintf(stdout,"x0, y0, z0 =, %15.6E %15.6E %15.6E\n",x0,y0,z0);

  // outer radius of the spherical grid
  CCTK_REAL rad_max = 0.9e0*fmin((x0-origin[0]),fmin((y0-origin[1]),(z0-origin[2])));

  //fprintf(stdout,"Radmax %15.6E\n",rad_max);

  CCTK_REAL rad_shock_av = 0.0e0;  // average shock radius
  CCTK_REAL rad_shock_dev = 0.0e0; // RMS of shock deviation
  CCTK_REAL rad_shock_max = 0.0e0; // max shock radius
  CCTK_INT  i_shock_max = 0;

  // shock radius
  CCTK_REAL * rad_shock = malloc((N_THETA*N_PHI) * sizeof(CCTK_REAL));

  // gain radius
  CCTK_REAL * rad_gain = malloc((N_THETA*N_PHI) * sizeof(CCTK_REAL));

  /* Define spherical coordinates*/ 
  CCTK_REAL drad,dtheta,dphi;
  CCTK_REAL * rad   = malloc(N_RAD * sizeof(CCTK_REAL));
  CCTK_REAL * theta = malloc(N_THETA * sizeof(CCTK_REAL));
  CCTK_REAL * phi   = malloc(N_PHI * sizeof(CCTK_REAL));
  dtheta = M_PI/(N_THETA-1);
  for(int i=0;i<N_THETA;i++) {
    theta[i] = i*dtheta;
    //fprintf(stdout,"theta =, %15.6E \n",theta[i]/M_PI);
  }
  dphi = 2*M_PI/(N_PHI);
  for(int i=0;i<N_PHI;i++) {
    phi[i] = i*dphi;
    //fprintf(stdout,"phi =, %15.6E \n",phi[i]/M_PI);
  }
  drad =  rad_max/(N_RAD-1);
  for(int i=0;i<N_RAD;i++) {
    rad[i] = i*drad;
  }

  // Calculate the Cartesian coordinates of the spherical grid points
  //#pragma omp parallel for
  for(int nr=0;nr<N_RAD;nr++)
  for(int nt=0;nt<N_THETA;nt++)
  for(int np=0;np<N_PHI;np++)
  {
    int const iind3d = nr+(N_RAD)*(nt+N_THETA*np);
    interp_x[iind3d] = x0 + rad[nr]*sin(theta[nt])*cos(phi[np]);
    interp_y[iind3d] = y0 + rad[nr]*sin(theta[nt])*sin(phi[np]);
    interp_z[iind3d] = z0 + rad[nr]*cos(theta[nt]);
  }

  // Spherical components of the velocity vector on Cartesian grid.
  CCTK_REAL * velr_cart = malloc((lsh[0]*lsh[1]*lsh[2]) * sizeof(CCTK_REAL));
  CCTK_REAL * velt_cart = malloc((lsh[0]*lsh[1]*lsh[2]) * sizeof(CCTK_REAL));
  CCTK_REAL * velp_cart = malloc((lsh[0]*lsh[1]*lsh[2]) * sizeof(CCTK_REAL));

  // Calculate the spherical components of the velocity field on the Cartesian grid
  // Also transform the values of lum_local
  //#pragma omp parallel for
  for(int i = 0; i < lsh[0]*lsh[1]*lsh[2]; i++)
    {
      CCTK_REAL cost;
      CCTK_REAL sint;
      CCTK_REAL sinp;
      CCTK_REAL cosp;
      CCTK_REAL rvalue;
      CCTK_REAL sxy;

      rvalue = sqrt(SQR(x[i]) + SQR(y[i]) + SQR(z[i]));
      sxy = sqrt(SQR(x[i]) + SQR(y[i]));

      //  be careful with r=0 and xy plane
      if (rvalue <= 1.0e-14) {
          cost = 1.0;
          sint = 0.0;
          sinp = 0.0;
          cosp = 1.0; }
      else if (sxy==0) {
          cost = 1.0;
          sint = 0.0;
          sinp = 0.0;
          cosp = 1.0; }
      else {
          cost = z[i]/rvalue;
          sint = sxy/rvalue;
          sinp = y[i]/sxy;
          cosp = x[i]/sxy; }

      velr_cart[i] =  velx[i]*cosp*sint+vely[i]*sinp*sint+velz[i]*cost;
      velt_cart[i] =  velx[i]*cosp*cost+vely[i]*sinp*cost-velz[i]*sint;
      velp_cart[i] = -velx[i]*sinp+vely[i]*cosp;

      // Transfor the values of lum_loc
      lum_local_0[i] = lum_local_0[i]/delta_v*1e51;
      lum_local_1[i] = lum_local_1[i]/delta_v*1e51;
      lum_local_2[i] = lum_local_2[i]/delta_v*1e51;
    }

  // calculate vortiricy
  for(int k = 0 ; k < lsh[2] ; k++)
    {
      for(int j = 0 ; j < lsh[1] ; j++)
	{
	  for(int i = 0 ; i < lsh[0] ; i++)
	    {
	      CCTK_INT ijk  = INDEX3D(i,j,k);
	      CCTK_INT imjk = INDEX3D(i-1,j,k);
	      CCTK_INT ipjk = INDEX3D(i+1,j,k);
	      CCTK_INT ijmk = INDEX3D(i,j-1,k);
	      CCTK_INT ijpk = INDEX3D(i,j+1,k);
	      CCTK_INT ijkm = INDEX3D(i,j,k-1);
	      CCTK_INT ijkp = INDEX3D(i,j,k+1);
	      
	      double dvelz_dy = 0.5e0*c_light*(velz[INDEX3D(i,j+1,k)]-velz[INDEX3D(i,j-1,k)])/(delta[1]*pow(INV_R_GF,1));
	      double dvely_dz = 0.5e0*c_light*(vely[INDEX3D(i,j,k+1)]-vely[INDEX3D(i,j,k+1)])/(delta[2]*pow(INV_R_GF,1));
	      double dvelx_dz = 0.5e0*c_light*(velx[INDEX3D(i,j,k+1)]-velx[INDEX3D(i,j,k-1)])/(delta[2]*pow(INV_R_GF,1));
	      double dvelz_dx = 0.5e0*c_light*(velz[INDEX3D(i+1,j,k)]-velx[INDEX3D(i-1,j,k)])/(delta[0]*pow(INV_R_GF,1));
	      double dvelx_dy = 0.5e0*c_light*(velz[INDEX3D(i,j+1,k)]-velx[INDEX3D(i,j-1,k)])/(delta[1]*pow(INV_R_GF,1));
	      double dvely_dx = 0.5e0*c_light*(vely[INDEX3D(i+1,j,k)]-vely[INDEX3D(i-1,j,k)])/(delta[0]*pow(INV_R_GF,1));

	      vort1[INDEX3D(i,j,k)] = dvelz_dy - dvely_dz;
	      vort2[INDEX3D(i,j,k)] = dvelx_dz - dvelz_dx;
	      vort3[INDEX3D(i,j,k)] = dvelx_dy - dvely_dx;

	      vort_cart[INDEX3D(i,j,k)] = pow(SQR(vort1[INDEX3D(i,j,k)])+SQR(vort2[INDEX3D(i,j,k)])+SQR(vort3[INDEX3D(i,j,k)]),0.5e0);
	    }
	}
    }
  

  // prepare for interpolation
  CCTK_INT output_array_type_codes[N_OUTPUT_ARRAYS]; /* one per output variable */
  CCTK_REAL * output_arrays[N_OUTPUT_ARRAYS];  /* see note above */
  for(int i = 0 ; i < N_OUTPUT_ARRAYS ; i++)
    output_array_type_codes[i] = CCTK_VARIABLE_REAL;

  int operator_handle, param_table_handle;
  operator_handle = CCTK_InterpHandle("uniform cartesian");
  if (operator_handle < 0)
          CCTK_WARN(CCTK_WARN_ABORT, "can’t get interpolation handle!");
  param_table_handle = Util_TableCreateFromString("order=3");
  if (param_table_handle < 0)
          CCTK_WARN(CCTK_WARN_ABORT, "can’t create parameter table!");

  /* initialize the rest of the parameter arrays */
  interp_coords[0] = (const void *) interp_x;
  interp_coords[1] = (const void *) interp_y;
  interp_coords[2] = (const void *) interp_z;

  input_arrays[0] = (const void *) entropy;    // interpolated variable 
  input_arrays[1] = (const void *) velr_cart;  // interpolated variable 
  input_arrays[2] = (const void *) velt_cart;  // interpolated variable 
  input_arrays[3] = (const void *) velp_cart;  // interpolated variable 
  input_arrays[4] = (const void *) rho;        // interpolated variable
  input_arrays[5] = (const void *) Y_e;        // interpolated variable 
  input_arrays[6] = (const void *) lum_local_0; // interpolated variable 
  input_arrays[7] = (const void *) lum_local_1; // interpolated variable
  input_arrays[8] = (const void *) lum_local_2; // interpolated variable 
  input_arrays[9] = (const void *) vort_cart;   // total vorticity in Cartesian coord

  output_arrays[0] = (CCTK_REAL *) entropy_spher;// output of first interp variable 
  output_arrays[1] = (CCTK_REAL *) velr_spher;   // output of second interpolated varbiale
  output_arrays[2] = (CCTK_REAL *) velt_spher;   // output of second interpolated varbiale
  output_arrays[3] = (CCTK_REAL *) velp_spher;   // output of second interpolated varbiale
  output_arrays[4] = (CCTK_REAL *) rho_spher;    // output of second interpolated varbiale
  output_arrays[5] = (CCTK_REAL *) ye_spher;     // output of second interpolated varbiale
  output_arrays[6] = (CCTK_REAL *) lumloc1_spher;// output of second interpolated varbiale
  output_arrays[7] = (CCTK_REAL *) lumloc2_spher;// output of second interpolated varbiale
  output_arrays[8] = (CCTK_REAL *) lumloc3_spher;// output of second interpolated varbiale
  output_arrays[9] = (CCTK_REAL *) vort_spher;   // total vorticity in sphericial coordinates

  // do the actual interpolation, and check for error returns 
  // Warning: this will *not* take symmetries into account! 
  int ierr = CCTK_InterpLocalUniform(N_DIMS,
                              operator_handle, param_table_handle,
                              origin, delta,
                              N_INTERP_POINTS,
                                 CCTK_VARIABLE_REAL,
                                 interp_coords,
                              N_INPUT_ARRAYS,
                                 lsh,
                                 input_array_type_codes,
                                 input_arrays,
                              N_OUTPUT_ARRAYS,
                                 output_array_type_codes,
                                 (void * const *)output_arrays);
  if (ierr < 0)
  {
          CCTK_WARN(CCTK_WARN_ABORT, "error return from interpolator!");
  }


  CCTK_INT n = 1;
  CCTK_INT eoskey = 4;
  CCTK_INT keytemp = 2;
  CCTK_INT keyerr,anyerr;

  CCTK_REAL xpress = 0.0e0;
  CCTK_REAL xeps = 0.0e0;
  CCTK_REAL xtemp = 0.0e0;
  CCTK_REAL xye,xrho,xent;
  CCTK_REAL xcs2 = 0.0e0;
  CCTK_REAL xdummy1 = 0.0e0;
  CCTK_REAL xdummy2 = 0.0e0;
  CCTK_REAL xdummy3 = 0.0e0;
  CCTK_REAL xdummy4 = 0.0e0;
  CCTK_REAL prec = 1.0e-8;
    
  // calculate temperature and cs2
  for (int nr=0;nr<N_RAD;nr++) 
    {
      for (int nt=0;nt<N_THETA;nt++) 
	{
	  if (theta[nt] > THETA_MIN && theta[nt] < THETA_MAX) 
	    {
	      for (int np=0;np<N_PHI;np++) 
		{
		  int const iind3d = nr+(N_RAD)*(nt+N_THETA*np);
		  xrho = rho_spher[iind3d];
		  xent = entropy_spher[iind3d];
		  xye  = ye_spher[iind3d];
		  anyerr = keyerr = 0;
		  EOS_Omni_short(eoskey,keytemp,prec,n,&xrho,&xeps,&xtemp,&xye,
				 &xpress,&xent,&xcs2,&xdummy1,&xdummy2,
				 &xdummy3,&xdummy4,&keyerr,&anyerr);
		  press_spher[iind3d] = xpress*INV_PRESS_GF;
		  temp_spher[iind3d] = xtemp;    
		  cs2_spher[iind3d] = xcs2;
		}
	    }
	}
    }  

  // calculate the shock radius
  rad_shock_max = 0.0e0;
  for(int nt=0;nt<N_THETA;nt++) {
    if (theta[nt] > THETA_MIN && theta[nt] < THETA_MAX) {
      for(int np=0;np<N_PHI;np++) {
	double rs = 0.0e0;
	double vmin = 10.0e0;
	for(int nr=0;nr<N_RAD;nr++) {
	  int const iind3d = nr+N_RAD*(nt+N_THETA*np);
	  if(velr_spher[iind3d] < vmin && rad[nr] >= rad_shock_min) {
	    vmin = velr_spher[iind3d];
	    rs = rad[nr];
	  }
	}
	int const iind2d = nt+N_THETA*np;
	rad_shock[iind2d] = rs;
	if (rad_shock_max < rs) rad_shock_max = rs;
      }
    }
  }

  // Doecompose the shock front into spherical harmonics
  double a00 = sh_coef(0,0, rad_shock, theta, phi, dtheta, dphi);
  double a10 = sh_coef(1,0, rad_shock, theta, phi, dtheta, dphi);
  double a20 = sh_coef(2,0, rad_shock, theta, phi, dtheta, dphi);
  double a30 = sh_coef(3,0, rad_shock, theta, phi, dtheta, dphi);
  double a40 = sh_coef(4,0, rad_shock, theta, phi, dtheta, dphi);

  rad_shock_av = a00; // average shock radius

  // calculate the standard deviation of shock radius
  // and max shock radius
  double delta_omega = 0.0e0;
  for(int nt=0;nt<N_THETA;nt++) 
    {
      if (theta[nt] > THETA_MIN && theta[nt] < THETA_MAX)
	{
	  for(int np=0;np<N_PHI;np++) 
	    {
	      int const iind2d = nt+N_THETA*np;
	      delta_omega = sin(theta[nt])*dtheta*dphi;
	      rad_shock_dev = rad_shock_dev+SQR(rad_shock_av-rad_shock[iind2d])*delta_omega;
	      if (rad_shock_max > rad_shock[iind2d])
		{
		  rad_shock_max = rad_shock[iind2d];
		}
	    }
	}
    }
  double Delta_omega = 4.0e0*M_PI*cos(THETA_MIN);
  rad_shock_dev = pow(rad_shock_dev/Delta_omega,0.5e0);
  
  for(int nr=0;nr<N_RAD;nr++) 
    {
      if (rad[nr] <= rad_shock_max)
	{
	  i_shock_max = nr;
	}
    }

  // calculate gain radius
  for (int nt=0;nt<N_THETA;nt++) 
    {
      if (theta[nt] > THETA_MIN && theta[nt] < THETA_MAX) 
	{
	  for (int np=0;np<N_PHI;np++) 
	    {
	      int const iind2d = nt+N_THETA*np;
	      rad_gain[iind2d] = rad_shock[iind2d];
	      for (int nr=N_RAD-1;nr>=0;nr--) 
		{
		  int const iind3d = nr+(N_RAD)*(nt+N_THETA*np);
		  double net_heat = -(lumloc1_spher[iind3d]+lumloc2_spher[iind3d]+lumloc3_spher[iind3d]);
		  if (rad[nr] < rad_shock[iind2d] && rad[nr] > rad_shock_min && net_heat > 0.0e0) 
		    {
		      rad_gain[iind2d] = rad[nr];
		    }
		}
	    }
	}
    }

  // calculate Mdot 
  for (int nt=0;nt<N_THETA;nt++) 
    {
      if (theta[nt] > THETA_MIN && theta[nt] < THETA_MAX) 
	{
	  for (int np=0;np<N_PHI;np++) 
	    {
	      int nr = i_shock_max + 1;
	      int const iind2d = nt+N_THETA*np;
	      int const iind3d = nr+(N_RAD)*(nt+N_THETA*np);
	      double delta_omega = sin(theta[nt])*dtheta*dphi;
	      mdot = mdot + (rho_spher[iind3d]*INV_RHO_GF)*(velr_spher[iind3d]*c_light)*
		SQR(rad[nr]*INV_R_GF)*delta_omega;
	    }
	}
    }
  mdot = fabs(mdot)*4.0e0*M_PI/Delta_omega; 

  // calculate the turbulence quantities 
  // and check the antesonic condition
  for (int nr=0;nr<N_RAD-1;nr++) 
    {
      double delta_omega = 0.0e0;
      sbar[nr]   = 0.0e0;
      s2bar[nr]  = 0.0e0;
      vrbar[nr]  = 0.0e0;
      vtbar[nr]  = 0.0e0;
      vpbar[nr]  = 0.0e0;
      vr2bar[nr] = 0.0e0;
      vt2bar[nr] = 0.0e0;
      vp2bar[nr] = 0.0e0;
      vrsbar[nr] = 0.0e0;
      cs2bar[nr] = 0.0e0;
      v1e2bar[nr] = 0.0e0;
      massbar[nr] = 0.0e0;
      rhobar[nr]  = 0.0e0;
      tempbar[nr] = 0.0e0;
      yebar[nr] = 0.0e0;
      pressbar[nr] = 0.0e0;
      press2bar[nr] = 0.0e0;
      rhosvrbar[nr] = 0.0e0;
      rhovrbar[nr] = 0.0e0;
      rhosbar[nr] = 0.0e0;
      sgainbar[nr] = 0.0e0;
      rhogainbar[nr] = 0.0e0;
      netheatbar[nr] = 0.0e0;
      lumloc1bar[nr] = 0.0e0;
      lumloc2bar[nr] = 0.0e0;
      lumloc3bar[nr] = 0.0e0;
      vortbar[nr] = 0.0e0;
      for(int nt=0;nt<N_THETA;nt++) 
	{
	  if (theta[nt] > THETA_MIN && theta[nt] < THETA_MAX) 
	    {
	      for(int np=0;np<N_PHI;np++) 
		{
		  int const iind2d = nt+N_THETA*np;
		  int const iind3d = nr+(N_RAD)*(nt+N_THETA*np);
		  delta_omega = sin(theta[nt])*dtheta*dphi;
		  sbar[nr]    = sbar[nr]   + entropy_spher[iind3d]*delta_omega;
		  s2bar[nr]   = s2bar[nr]  + SQR(entropy_spher[iind3d])*delta_omega;
		  vrbar[nr]   = vrbar[nr]  + velr_spher[iind3d]*delta_omega;
		  vtbar[nr]   = vtbar[nr]  + velt_spher[iind3d]*delta_omega;
		  vpbar[nr]   = vpbar[nr]  + velp_spher[iind3d]*delta_omega;
		  vr2bar[nr]  = vr2bar[nr] + SQR(velr_spher[iind3d])*delta_omega;
		  vt2bar[nr]  = vt2bar[nr] + SQR(velt_spher[iind3d])*delta_omega;
		  vp2bar[nr]  = vp2bar[nr] + SQR(velp_spher[iind3d])*delta_omega;
		  vrsbar[nr]  = vrsbar[nr] + velr_spher[iind3d]*entropy_spher[iind3d]*delta_omega;
		  cs2bar[nr]  = cs2bar[nr] + cs2_spher[iind3d]*delta_omega;
		  rhobar[nr]  = rhobar[nr] + rho_spher[iind3d]*delta_omega;
		  tempbar[nr] = tempbar[nr]+ temp_spher[iind3d]*delta_omega;
		  yebar[nr]   = yebar[nr]  + ye_spher[iind3d]*delta_omega;
		  pressbar[nr] = pressbar[nr]+ press_spher[iind3d]*delta_omega;
		  press2bar[nr] = press2bar[nr]+ SQR(press_spher[iind3d])*delta_omega;
		  rhosvrbar[nr] = rhosvrbar[nr] + rho_spher[iind3d]*entropy_spher[iind3d]*
		    velr_spher[iind3d]*delta_omega;
		  rhovrbar[nr] = rhovrbar[nr] + rho_spher[iind3d]*velr_spher[iind3d]*delta_omega;
		  rhosbar[nr] = rhosbar[nr] + rho_spher[iind3d]*entropy_spher[iind3d]*delta_omega;
		  lumloc1bar[nr] = lumloc1bar[nr] + lumloc1_spher[iind3d]*delta_omega;
		  lumloc2bar[nr] = lumloc2bar[nr] + lumloc2_spher[iind3d]*delta_omega;
		  lumloc3bar[nr] = lumloc3bar[nr] + lumloc3_spher[iind3d]*delta_omega;
		  netheatbar[nr] = netheatbar[nr] + (lumloc1_spher[iind3d]+lumloc2_spher[iind3d]+lumloc3_spher[iind3d])*delta_omega;
		  vortbar[nr]    = vortbar[nr] + vort_spher[iind3d]*delta_omega;
		}
	    }
	}
      double Delta_omega = 4.0e0*M_PI*cos(THETA_MIN);
      sbar[nr]   = sbar[nr]/Delta_omega;
      s2bar[nr]  = s2bar[nr]/Delta_omega;
      vrbar[nr]  = vrbar[nr]/Delta_omega;
      vtbar[nr]  = vtbar[nr]/Delta_omega;
      vpbar[nr]  = vpbar[nr]/Delta_omega;
      vr2bar[nr] = vr2bar[nr]/Delta_omega;
      vt2bar[nr] = vt2bar[nr]/Delta_omega;
      vp2bar[nr] = vp2bar[nr]/Delta_omega;
      vrsbar[nr] = vrsbar[nr]/Delta_omega;
      cs2bar[nr] = cs2bar[nr]/Delta_omega;
      rhobar[nr] = rhobar[nr]/Delta_omega;
      tempbar[nr]= tempbar[nr]/Delta_omega;
      yebar[nr]  = yebar[nr]/Delta_omega;
      pressbar[nr] = pressbar[nr]/Delta_omega;
      press2bar[nr] = press2bar[nr]/Delta_omega;
      rhosvrbar[nr] = rhosvrbar[nr]/Delta_omega;
      rhovrbar[nr] = rhovrbar[nr]/Delta_omega;
      rhosbar[nr] = rhosbar[nr]/Delta_omega;
      lumloc1bar[nr] = lumloc1bar[nr]/Delta_omega;
      lumloc2bar[nr] = lumloc2bar[nr]/Delta_omega;
      lumloc3bar[nr] = lumloc3bar[nr]/Delta_omega;
      netheatbar[nr] = netheatbar[nr]/Delta_omega;
      vortbar[nr] = vortbar[nr]/Delta_omega;

      if (nr == 0) // calculate enclosed mass
	{
	  massbar[nr] = rhobar[nr]*INV_RHO_GF // in gramms
	    *4.0e0*M_PI*CUBE(drad*INV_R_GF); 
	}
      else 
	{
	  massbar[nr] = massbar[nr-1]+rhobar[nr]*INV_RHO_GF // in gramms
	    *4.0e0*M_PI*SQR(rad[nr])*drad*CUBE(INV_R_GF);
	}
      
      v1e2bar[nr] = 2.0e0*G_GRAV*massbar[nr]/(rad[nr]*INV_R_GF);
      ratio_sonic_1d[nr] = cs2bar[nr]/v1e2bar[nr];

      F_s[nr] = (2.0e0*rhosvrbar[nr]-rhosbar[nr]*vrbar[nr]-rhovrbar[nr]*sbar[nr])*c_light;
      f_s[nr] = (vrsbar[nr]-vrbar[nr]*sbar[nr])*c_light;
      f_e[nr] = f_s[nr]*tempbar[nr]*rhobar[nr]*ENTROPY_FACTOR; 
      k_r[nr] = 0.5e0*(vr2bar[nr]-SQR(vrbar[nr]))*SQR(c_light);
      k_t[nr] = 0.5e0*(vt2bar[nr]-SQR(vtbar[nr]))*SQR(c_light);
      k_p[nr] = 0.5e0*(vp2bar[nr]-SQR(vpbar[nr]))*SQR(c_light);
      q_s[nr] = s2bar[nr]-SQR(sbar[nr]);
      p_s[nr] = press2bar[nr]-SQR(pressbar[nr]);
      
      // calculate binding energy and heating rate
      for(int nt=0;nt<N_THETA;nt++) 
	{
	  if (theta[nt] > THETA_MIN && theta[nt] < THETA_MAX) 
	    {
	      for(int np=0;np<N_PHI;np++) 
		{
		  int const iind2d = nt+N_THETA*np;
		  int const iind3d = nr+(N_RAD)*(nt+N_THETA*np);
		  if (rad[nr] > rad_gain[iind2d] && rad[nr] < rad_shock[iind2d])
		    {
		      //fprintf(stdout,"Here %d %15.6E %15.6E\n",nr,rad_gain[iind2d],rad_shock[iind2d]);
		      double delta_omega = sin(theta[nt])*dtheta*dphi;
		      double delta_v_spher = SQR(rad[nr])*drad*pow(INV_R_GF,3)*delta_omega;
		      
		      e_bind = e_bind + fabs(G_GRAV*massbar[nr]/(rad[nr]*INV_R_GF)*
					     (rho_spher[iind3d]*INV_RHO_GF))*delta_v_spher;

		      double net_heat = -(lumloc1_spher[iind3d]+lumloc2_spher[iind3d]+lumloc3_spher[iind3d]);
		      q_heat = q_heat + fabs(net_heat)*delta_v_spher;

		      sgainbar[nr] = sgainbar[nr] + entropy_spher[iind3d]*rho_spher[iind3d]*delta_omega;
		      rhogainbar[nr] = rhogainbar[nr] + rho_spher[iind3d]*delta_omega;

		      m_gain = m_gain + rho_spher[iind3d]*INV_RHO_GF*delta_v_spher;
		      
		      if (velr_spher[iind3d] < 0) // Based on eq. 6 from Takiwaki et al. (2012) 
			{
			  tau_res[iind3d] = (rad[nr]-rad_gain[iind2d])*INV_R_GF/(-velr_spher[iind3d]*c_light);
			}
		      else if (velr_spher[iind3d] > 0) 
			{
			  tau_res[iind3d] = (rad_shock[iind2d]-rad[nr])*INV_R_GF/(velr_spher[iind3d]*c_light);
			}
		      else
			{
			  tau_res[iind3d] = 0.0e0;
			}
		      tau_res_av = tau_res_av + tau_res[iind3d]*delta_v_spher;
		      volume_gain = volume_gain + delta_v_spher;
		    }
		}
	    }
	}

      if (rhogainbar[nr] != 0.0e0)
	{
	  sgainbar[nr] = sgainbar[nr]/rhogainbar[nr];
	}
      else
	{
	  sgainbar[nr] = 0.0e0;
	}
    }

  m_gain = m_gain*4.0e0*M_PI/Delta_omega;

  tau_res_mdot = m_gain / mdot; 

  if (q_heat != 0.0e0)
    {
      tau_heat = e_bind/q_heat;
    }
  else
    {
      tau_heat = 0.0e0;
    }

  if (volume_gain != 0.0e0)
    {
      tau_res_av = tau_res_av/volume_gain;
    }
  else
    {
      tau_res_av = 0.0e0;
    }

  double current_time = (*ptrtime - t_bounce)/203.0e0; // time after bounce in ms. 

  // output results to file
  FILE *f1;
  f1 = fopen ("shock_data_vs_t.dat", "a");
  fprintf(f1,"%d %g %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E\n",
	  *ptriteration,current_time,rad_shock_av*1.4768e0,rad_shock_dev*1.4768e0,
	  a00*1.4768e0,a10/a00,a20/a00,a30/a00,a40/a00);
  fclose(f1);

  f1 = fopen ("timescales_vs_t.dat", "a");
  fprintf(f1,"%d %g %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E\n",
	  *ptriteration,current_time,tau_heat,tau_res_av,tau_res_mdot,mdot,m_gain,
	  e_bind,q_heat);
  fclose(f1);

  f1 = fopen ("turbulence_data_vs_r.dat", "a");
  fprintf(f1,"\"Time= %g, Iteration= %d\"\n",current_time,*ptriteration);
  for(int nr=0; nr<N_RAD-1; nr++)
    {
      fprintf(f1,"%5d %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E %15.6E\n",
	      nr,(INV_R_GF*rad[nr]/1.0e5),f_s[nr],F_s[nr]*INV_RHO_GF*ENTROPY_FACTOR,k_r[nr],
	      k_t[nr],k_p[nr],q_s[nr],p_s[nr],sbar[nr],s2bar[nr],vortbar[nr]);
    }
  fprintf(f1," \n");
  fclose(f1);

  f1 = fopen ("matter_background_data_vs_r.dat", "a");
  fprintf(f1,"\"Time= %g, Iteration= %d\"\n",current_time,*ptriteration);
  for(int nr=0; nr<N_RAD-1; nr++)
    {
      fprintf(f1,"%5d %15.6E %15.6E %15.6E %15.6E %15.6E\n",
	      nr,(INV_R_GF*rad[nr]/1.0e5),(INV_RHO_GF*rhobar[nr]),(MEV2KELVIN*tempbar[nr]),
	      yebar[nr],(massbar[nr]/SOLAR_MASS));
    }
  fprintf(f1," \n");
  fclose(f1);
 
  f1 = fopen ("gain_region_data_vs_r.dat", "a");
  fprintf(f1,"\"Time= %g, Iteration= %d\"\n",current_time,*ptriteration);
  for(int nr=0; nr<N_RAD-1; nr++)
    {
      fprintf(f1,"%5d %15.6E %15.6E %15.6E %15.6E\n",
	      nr,(INV_R_GF*rad[nr]/1.0e5),sgainbar[nr],netheatbar[nr],rhogainbar[nr]);
    }
  fprintf(f1," \n");
  fclose(f1);

  f1 = fopen ("antesonic_data_vs_r.dat", "a");
  fprintf(f1,"\"Time= %g, Iteration= %d\"\n",current_time,*ptriteration);
  for(int nr=0; nr<N_RAD-1; nr++)
    {
      fprintf(f1,"%5d %15.6E %15.6E %15.6E %15.6E\n",
	      nr,(INV_R_GF*rad[nr]/1.0e5),ratio_sonic_1d[nr],cs2bar[nr],v1e2bar[nr]);
    }
  fprintf(f1," \n");
  fclose(f1);


  // always free() memory gotten from malloc() 
  free(interp_x);
  free(interp_y);
  free(interp_z);
  free(rad);
  free(theta);
  free(phi);
  free(velr_cart);
  free(velt_cart);
  free(velp_cart);
  free(velr_spher);
  free(velt_spher);
  free(velp_spher);
  free(press_spher);
  free(entropy_spher);
  free(lumloc1_spher);
  free(lumloc2_spher);
  free(lumloc3_spher);
  free(temp_spher);
  free(ye_spher);
  free(cs2_spher);
  free(rho_spher);
  free(f_s);
  free(F_s);
  free(f_e);
  free(k_r);
  free(k_t);
  free(k_p);
  free(q_s);
  free(p_s);
  free(sbar);
  free(sgainbar);
  free(rhogainbar);
  free(s2bar);
  free(vrbar);
  free(vr2bar);
  free(vrsbar);
  free(vtbar);
  free(vt2bar);
  free(vpbar);
  free(vp2bar);
  free(cs2bar);
  free(v1e2bar);
  free(massbar);
  free(ratio_sonic_1d);
  free(rhobar);
  free(tempbar);
  free(yebar);
  free(pressbar);
  free(press2bar);
  free(rad_shock);
  free(rad_gain);
  free(rhosvrbar);
  free(rhovrbar);
  free(rhosbar);
  free(netheatbar);
  free(tau_res);
  free(lumloc1bar);
  free(lumloc2bar);
  free(lumloc3bar);
  free(vort1);
  free(vort2);
  free(vort3);
  free(vort_cart);
  free(vort_spher);
  free(vortbar);
  return;
}

static double sh_coef(int l, int m, double *rshock, double *theta, double *phi, double dtheta, double dphi)
{
  // Calculate coefficients of spherical harmonics decomposition.

  double alm = 0.0e0;
  
  for (int nt=0;nt<N_THETA;nt++) 
    {
      if (theta[nt] > THETA_MIN && theta[nt] < THETA_MAX)
	{
	  for (int np=0;np<N_PHI;np++) 
	    {
	      int const iind2d = nt+N_THETA*np;
	      double delta_omega = sin(theta[nt])*dtheta*dphi;
	      alm = alm + pow(-1,abs(m))/pow(4.0*M_PI*(2.0*l+1.0),0.5e0)*
		rshock[iind2d]*spherical_harmonics(theta[nt],phi[np],l,m)*
		delta_omega;
	    }
	}
    }

  double Delta_omega = 4.0e0*M_PI*cos(THETA_MIN);
  alm = alm*4.0e0*M_PI/Delta_omega;

  return alm;
}
  

static double spherical_harmonics(double theta, double phi, int l, int m)
{
  // This function is defined here because GSL does not provide 
  // spherical harmonics for negative m.

  double ylm;
  double x = cos(theta);

  if (m > 0) {
    ylm = pow(2.0e0,0.5e0)*nlmfactor(l,m)*gsl_sf_legendre_Plm(l,m,x)*
      cos(m*phi);
  }
  else if (m == 0) {
    ylm = nlmfactor(l,m)*gsl_sf_legendre_Plm(l,m,x);
  }
  else {
    ylm = pow(2.0e0,0.5e0)*nlmfactor(l,abs(m))*gsl_sf_legendre_Plm(l,abs(m),x)*
      sin(abs(m)*phi);
  }
  return ylm;
}

static double nlmfactor(int l, int m)
{
  // calculates the factor in front of spherical harmonics
  
  double nlm = pow((2.0e0*l+1.0e0)/(4.0*M_PI)*gsl_sf_fact(l-m)/ 
		   gsl_sf_fact(l+m),0.5e0);
  return nlm;
}


// read in a CCTK_REAL[3] attribute
static void read_rvec_attr(hid_t from, const char *attrname, CCTK_REAL data[3])
{
  hid_t attr, datatype, dataspace;
  hsize_t attrsize;

  CHECK_ERROR (attr = H5Aopen_name (from, attrname));
  CHECK_ERROR (datatype = H5Aget_type (attr));
  CHECK_ERROR (dataspace = H5Aget_space (attr));

  assert(datatype = H5T_NATIVE_DOUBLE);

  CHECK_ERROR (attrsize = H5Sget_simple_extent_npoints (dataspace));
  assert(attrsize == 3);

  CHECK_ERROR (H5Aread (attr, datatype, data));

  CHECK_ERROR (H5Sclose (dataspace));
  CHECK_ERROR (H5Aclose (attr));
}

// read in a CCTK_REAL attribute
static void read_real_attr(hid_t from, const char *attrname, int nelems, CCTK_REAL *data)
{
  hid_t attr, datatype, dataspace;
  hsize_t attrsize;

  CHECK_ERROR (attr = H5Aopen_name (from, attrname));
  CHECK_ERROR (datatype = H5Aget_type (attr));
  CHECK_ERROR (dataspace = H5Aget_space (attr));

  assert(datatype = H5T_NATIVE_DOUBLE);

  CHECK_ERROR (attrsize = H5Sget_simple_extent_npoints (dataspace));
  assert((int)attrsize == nelems);

  CHECK_ERROR (H5Aread (attr, datatype, data));

  CHECK_ERROR (H5Sclose (dataspace));
  CHECK_ERROR (H5Aclose (attr));
}

// check if object is a dataset, then read in all related datasets and call analysis routine.
static herr_t ParseObject (hid_t from,
                          const char *objectname,
                          void *cctkGH)
{
  static const char * dsets[] = {
    "HYDROBASE::entropy it=%d tl=0 rl=0 c=0",
    "HYDROBASE::rho it=%d tl=0 rl=0 c=0",
    "HYDROBASE::vel[0] it=%d tl=0 rl=0 c=0",
    "HYDROBASE::vel[1] it=%d tl=0 rl=0 c=0",
    "HYDROBASE::vel[2] it=%d tl=0 rl=0 c=0",
    "HYDROBASE::Y_e it=%d tl=0 rl=0 c=0",
    "ZELMANILEAK::lum_local[0] it=%d tl=0 rl=0 c=0",
    "ZELMANILEAK::lum_local[1] it=%d tl=0 rl=0 c=0",
    "ZELMANILEAK::lum_local[2] it=%d tl=0 rl=0 c=0",
  };
  char buffer[1042];
  enum varidx { entropy = 0, rho, velx, vely, velz, Y_e, lum_local_0, lum_local_1, lum_local_2, num_vars };
  //enum varidx { entropy = 0, rho, velx, vely, velz, Y_e, num_vars };
  CCTK_REAL *vardata[DIM(dsets)];
  CCTK_REAL *coords[4];
  int iteration;
  CCTK_INT lsh[3];
  CCTK_REAL delta[DIM(lsh)], origin[DIM(lsh)], mytime;

  assert(DIM(dsets) == num_vars);

  // trigger reading of all input datasets when we see a density one
  if(1 == sscanf(objectname, dsets[0], &iteration))
  {
    // open all related datasets
    for(int i = 0 ; i < DIM(dsets) ; i++)
    {
      hsize_t dims[DIM(lsh)], ndims, objectsize;
      hid_t dataset, dataspace, datatype;
      int len_written;

      len_written = snprintf(buffer, sizeof(buffer), dsets[i], iteration);
      assert(len_written < (int)sizeof(buffer));

      CHECK_ERROR (dataset = H5Dopen (from, buffer));
      CHECK_ERROR (datatype = H5Dget_type (dataset));
      CHECK_ERROR (dataspace = H5Dget_space (dataset));

      assert(datatype = H5T_NATIVE_DOUBLE);

      objectsize = H5Sget_select_npoints (dataspace) * H5Tget_size (datatype);
      assert(objectsize > 0);
      vardata[i] = malloc (objectsize);
      if (vardata[i] == NULL)
      {
        fprintf (stderr, "failled to allocate %d bytes of memory for %s, giving up\n",
                 (int) objectsize, buffer);
        exit (-1);
      }
      CHECK_ERROR (H5Dread (dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                            vardata[i]));

      // store cctk_lssh and grid structure
      {
        CHECK_ERROR (ndims = H5Sget_simple_extent_ndims(dataspace));
        assert(ndims == DIM(lsh));
        CHECK_ERROR (H5Sget_simple_extent_dims(dataspace, dims, NULL));

        read_rvec_attr(dataset, "origin", origin);
        read_rvec_attr(dataset, "delta", delta);
        read_real_attr(dataset, "time", 1, &mytime);

        if(i == 0)
        {
          CCTK_INT size = 1;
          for(int d = 0 ; d < (int)ndims ; d++)
          {
            assert (ndims-1-d>=0 && ndims-1-d<ndims);
            lsh[d] = (CCTK_INT)dims[ndims-1-d]; // HDF5 has the slowest changing direction first, Cactus the fastest
            size *= lsh[d];
          }
          assert(size*sizeof(CCTK_REAL) == objectsize);
        }
        else
        {
          for(int d = 0 ; d < (int)ndims ; d++)
            assert(lsh[d] == (CCTK_INT)dims[ndims-1-d]); // translate between Fotran (Cactus) and C (HDF5 order)
        }
      }

      CHECK_ERROR (H5Dclose (dataset));
      CHECK_ERROR (H5Sclose (dataspace));
    } // for (dsets)

    // create coordinates arrays
    assert(DIM(lsh) == 3);
    for(int i = 0 ; i < DIM(coords) ; i++)
    {
      coords[i] = malloc(lsh[0] * lsh[1] * lsh[2] * sizeof(CCTK_REAL));
      assert(coords[i]);
    }
    for(int k = 0 ; k < lsh[2] ; k++)
    {
      for(int j = 0 ; j < lsh[1] ; j++)
      {
        for(int i = 0 ; i < lsh[0] ; i++)
        {
          CCTK_INT idx = i + j * lsh[1] + k * lsh[1]*lsh[2];
          CCTK_INT ijk[3] = {i,j,k};
          coords[DIM(coords)-1][idx] = 0.0; // radius
          for(int d = 0 ; d < DIM(coords)-1 ; d++)
          {
            coords[d][idx] = origin[d] + ijk[d] * delta[d];
            coords[DIM(coords)-1][idx] += SQR(coords[d][idx]);
          }
          coords[DIM(coords)-1][idx] = sqrt(coords[DIM(coords)-1][idx]);
        }
      }
    }
    
    // call analysis routine
    CCTK_FNAME(Turbulence_Analysis)(cctkGH, 
        &iteration, &mytime, lsh, origin, delta,
        coords[0], coords[1], coords[2], coords[3],
        vardata[entropy], vardata[rho], vardata[velx],
        vardata[vely], vardata[velz], vardata[Y_e],
        vardata[lum_local_0], vardata[lum_local_1], 
        vardata[lum_local_2]);
    
    // free data
    for(int i = 0 ; i < DIM(coords) ; i++)
    {
      assert(coords[i]);
      free(coords[i]);
      coords[i] = NULL;
    }
    for(int i = 0 ; i < DIM(dsets) ; i++)
    {
      assert(vardata[i]);
      free(vardata[i]);
      vardata[i] = NULL;
    }
  } // if(sscanf(dsets[0]

  return 0;
}

// scheduled routine, doesn't really do very much
void SOPP_Run(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  char *fns_buf;

  fns_buf = strdup(files);
  assert(fns_buf);

  for(const char *fn = strtok(fns_buf, " ") ; fn != NULL ; fn = strtok(NULL, " "))
  {
      hid_t fh;
      CHECK_ERROR (fh = H5Fopen (fn, H5F_ACC_RDONLY, H5P_DEFAULT));
      CHECK_ERROR (H5Giterate (fh, "/", NULL, ParseObject, cctkGH));
      CHECK_ERROR (H5Fclose (fh));
  }

  free(fns_buf);
  CCTK_TerminateNext(cctkGH);
}


