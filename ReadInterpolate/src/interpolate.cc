#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include "util_Table.h"

#include "carpet.hh"

#include "readinterpolate.h"


/********************************************************************
 ********************* Local Routine Prototypes *********************
 ********************************************************************/

#define DIM(x) ((int)(sizeof(x)/sizeof(x[0])))
#define MIN(x,y) ((x)<(y) ? (x) : (y))
#define MAX(x,y) ((x)>(y) ? (x) : (y))

static void DoInterpolate(size_t npoints, 
    const CCTK_REAL * x, const CCTK_REAL * y, const CCTK_REAL * z,
    const CCTK_INT lsh[3], const CCTK_REAL origin[3], const CCTK_REAL delta[3],
    const CCTK_REAL * in_array, CCTK_REAL * out_array);

/********************************************************************
 *********************     External Routines   **********************
 ********************************************************************/

// set all seen refinement level data to -1 so that the coarsest on triggers
void ReadInterpolate_ClearRefLevelSeen(const cGH * cctkGH)
{
  //BEGIN_REFLEVEL_LOOP(cctkGH) { // we run in level mode to look more like an ordinary id thorn
    BEGIN_LOCAL_MAP_LOOP (cctkGH, CCTK_GF) {
      BEGIN_LOCAL_COMPONENT_LOOP (cctkGH, CCTK_GF) {

        DECLARE_CCTK_ARGUMENTS;

        for(int idx = 0 ; idx < cctk_lsh[0]*cctk_lsh[1]*cctk_lsh[2] ; idx++)
          reflevelseen[idx] = -1;

      } END_LOCAL_COMPONENT_LOOP;
    } END_LOCAL_MAP_LOOP;
  //} END_REFLEVEL_LOOP;
}

// check that all target points have been set to something
void ReadInterpolate_CheckAllPointsSet(const cGH * cctkGH)
{
  DECLARE_CCTK_PARAMETERS;

  int nunset_points = 0;

  //BEGIN_REFLEVEL_LOOP(cctkGH) { // we run in level mode to look more like an ordinary id thorn
    int nunset_points_level = 0;
    BEGIN_LOCAL_MAP_LOOP (cctkGH, CCTK_GF) {
      BEGIN_LOCAL_COMPONENT_LOOP (cctkGH, CCTK_GF) {

        DECLARE_CCTK_ARGUMENTS;

        for(int idx = 0 ; idx < cctk_lsh[0]*cctk_lsh[1]*cctk_lsh[2] ; idx++)
        {
          if(reflevelseen[idx] == -1)
          {
            nunset_points_level += 1;
            if(verbosity >= 8)
            {
              CCTK_VInfo(CCTK_THORNSTRING, "Point (%g,%g,%g) on target level %d was not set",
                         x[idx],y[idx],z[idx], Carpet::reflevel);
            }
          }
        }

      } END_LOCAL_COMPONENT_LOOP;
    } END_LOCAL_MAP_LOOP;
    if(nunset_points_level > 0 && verbosity > 1)
    {
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "There were %d points that could not be set on target level %d.",
                 nunset_points_level, Carpet::reflevel);
    }
    nunset_points += nunset_points_level;
  //} END_REFLEVEL_LOOP;

  if(nunset_points > 0)
  {
    CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
               "There were %d points that could not be set.",
               nunset_points);
    return; // NOTREACHED
  }
}

// interpolate a HDF5 patch onto all Carpet patches that overlap
void ReadInterpolate_Interpolate(const cGH * cctkGH, int iteration, int component, int reflevel,
                                 int varindex, const CCTK_INT lsh[3], const CCTK_REAL origin[3],
                                 const CCTK_REAL delta[3], 
                                 CCTK_REAL const * const vardata, void *token)
{
  DECLARE_CCTK_PARAMETERS;

  //BEGIN_REFLEVEL_LOOP(cctkGH) { // we run in level mode to look more like an ordinary id thorn
    BEGIN_LOCAL_MAP_LOOP (cctkGH, CCTK_GF) {
      BEGIN_LOCAL_COMPONENT_LOOP (cctkGH, CCTK_GF) {

        DECLARE_CCTK_ARGUMENTS;

        // region for which we have enough inner and ghost points to interpolate,
        // assuming the interpolator needs cctk_nghostzones ghosts 
        CCTK_REAL xmin[3] = {origin[0]+(cctk_nghostzones[0]-1)*delta[0],
                             origin[1]+(cctk_nghostzones[1]-1)*delta[1],
                             origin[2]+(cctk_nghostzones[2]-1)*delta[2]};
        CCTK_REAL xmax[3] = {origin[0]+(lsh[0]-cctk_nghostzones[0])*delta[0],
                             origin[1]+(lsh[1]-cctk_nghostzones[1])*delta[1],
                             origin[2]+(lsh[2]-cctk_nghostzones[2])*delta[2]};
        CCTK_REAL *xyz[3] = {x,y,z};

        CCTK_REAL * outvardata;  // pointer to output variable data
       
        outvardata = static_cast<CCTK_REAL*>(CCTK_VarDataPtrI(cctkGH, 0, varindex));
        if(outvardata == NULL)
        {
          CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
                     "Requested variable '%s' does not have storage.",
                     CCTK_VarName(varindex));
          return; // NOTREACHED
        }

        if(verbosity >= 6)
        {
          CCTK_VInfo(CCTK_THORNSTRING, "checking for overlap against (%g,%g,%g)-(%g,%g,%g)",
                     xmin[0],xmin[1],xmin[2], xmax[0],xmax[1],xmax[2]);
        }

        // check for overlap and interpolate
        size_t npoints = 0;

        for(int idx = 0 ; idx < cctk_lsh[0]*cctk_lsh[1]*cctk_lsh[2] ; idx++)
        {
          CCTK_REAL xL = x[idx], yL = y[idx], zL = z[idx];
          if(undo_rot90)
          {
            while(xL < 0 || yL < 0) // 90 degree rotation
            {
              CCTK_REAL tmp = yL;
              yL = -xL;
              xL = tmp;
            }
          }

          if(reflevelseen[idx] <= reflevel && // need <= since we re-use the same level information for all output grid functions
             xmin[0]-epsilon <= xL && xL-epsilon <= xmax[0] &&
             xmin[1]-epsilon <= yL && yL-epsilon <= xmax[1] &&
             xmin[2]-epsilon <= zL && zL-epsilon <= xmax[2])
          {
            interp_x[npoints] = xL;
            interp_y[npoints] = yL;
            interp_z[npoints] = zL;
            interpthispoint[idx] = 1; // record that we need this point
            npoints += 1;
            if(verbosity >= 10 || (verbosity >= 9 && npoints % (1 + npoints / 10) == 0))
            {
              CCTK_VInfo(CCTK_THORNSTRING, "setting up interpolation for point %d (%g,%g,%g) source level %d",
                         (int)npoints, x[idx],y[idx],z[idx], reflevel);
            }
          }
          else
            interpthispoint[idx] = 0;
        }

        if(verbosity >= 5-(npoints>0))
        {
          CCTK_VInfo(CCTK_THORNSTRING, "found overlap and %d not-yet-seen points on source level %d on destination level %d for variable %s",
                     (int)npoints, reflevel, Carpet::reflevel, CCTK_VarName(varindex));
        }

        if(npoints > 0)
        {
          // ask for delayed read to be performed now that we know we need data
          ReadInterpolate_PullData(token);
          DoInterpolate(npoints, interp_x, interp_y, interp_z, 
                        lsh, origin, delta, vardata, interp_data);

          for(int idx = cctk_lsh[0]*cctk_lsh[1]*cctk_lsh[2]-1 ; idx >= 0 ; --idx)
          {
            if(interpthispoint[idx])
            {
              npoints -= 1; // push/pop logic, must be before access
              outvardata[idx] = interp_data[npoints];
              reflevelseen[idx] = reflevel;
              if(verbosity >= 10 || (verbosity >= 9 && npoints % (1 + npoints / 10) == 0))
              {
                CCTK_VInfo(CCTK_THORNSTRING, "received value %g for point (%g,%g,%g) source level %d",
                           outvardata[idx], x[idx],y[idx],z[idx], reflevel);
              }
            }
          }
        }
        assert(npoints == 0);

      } END_LOCAL_COMPONENT_LOOP;
    } END_LOCAL_MAP_LOOP;
  //} END_REFLEVEL_LOOP;
}

static void DoInterpolate(size_t npoints, 
    const CCTK_REAL * x, const CCTK_REAL * y, const CCTK_REAL * z,
    const CCTK_INT lsh[3], const CCTK_REAL origin[3], const CCTK_REAL delta[3],
    const CCTK_REAL * in_array, CCTK_REAL * out_array)
{
  DECLARE_CCTK_PARAMETERS;

  /* (x,y) coordinates of interpolation points */  
  void const * const interp_coords[3] = {x,y,z};

  /* input arrays */
  CCTK_INT const input_array_type_codes[1] = {CCTK_VARIABLE_REAL};
  void const * const input_arrays[1] = {in_array};

  /* output arrays */
  const CCTK_INT output_array_type_codes[1] = {CCTK_VARIABLE_REAL};
  void * output_arrays[1] = {out_array};

  int operator_handle, param_table_handle;
  operator_handle = CCTK_InterpHandle(interpolator_name);
  if (operator_handle < 0)
          CCTK_WARN(CCTK_WARN_ABORT, "can’t get interpolation handle!");
  param_table_handle = Util_TableCreateFromString(interpolator_pars);
  if (param_table_handle < 0)
          CCTK_WARN(CCTK_WARN_ABORT, "can’t create parameter table!");


  // do the actual interpolation, and check for error returns 
  int ierr = CCTK_InterpLocalUniform(3,
                              operator_handle, param_table_handle,
                              origin, delta,
                              (CCTK_INT)npoints,
                                 CCTK_VARIABLE_REAL,
                                 interp_coords,
                              DIM(input_arrays),
                                 lsh,
                                 input_array_type_codes,
                                 input_arrays,
                              DIM(output_arrays),
                                 output_array_type_codes,
                                 (void * const *)output_arrays);
  if (ierr < 0)
  {
          CCTK_WARN(CCTK_WARN_ABORT, "error return from interpolator!");
  }

  Util_TableDestroy(param_table_handle);
}
