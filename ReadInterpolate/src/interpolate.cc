#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <map>
#include <iostream>
#include <sstream>

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
 *********************     Local Data         ***********************
 ********************************************************************/

typedef std::map<int, int> varseenmap;
static varseenmap varseen;

/********************************************************************
 ********************* Internal Routines  ***************************
 ********************************************************************/

static void ClearRefLevelSeen(const cGH * cctkGH, const int var);

/********************************************************************
 *********************     External Routines   **********************
 ********************************************************************/

// check that all target points have been set to something
void ReadInterpolate_CheckAllPointsSet(const cGH * cctkGH)
{
  DECLARE_CCTK_PARAMETERS;

  int nunset_points = 0;
  int nunset_vars = 0;

  for(varseenmap::const_iterator it = varseen.begin(), end = varseen.end() ;
      it != end ;
      it++) {
    const int varindex = it->first;
    const int var = it->second;

    // when more variables are requested than I have buffer space, I ignore the
    // extra ones but abort and inform the user after parsing the files what
    // the correct number would have been
    if(var >= max_number_of_read_variables)
      continue;

  //BEGIN_REFLEVEL_LOOP(cctkGH) { // we run in level mode to look more like an ordinary id thorn
    int nunset_points_var = 0;
    BEGIN_LOCAL_MAP_LOOP (cctkGH, CCTK_GF) {
      BEGIN_LOCAL_COMPONENT_LOOP (cctkGH, CCTK_GF) {

        DECLARE_CCTK_ARGUMENTS;
        CCTK_INT *myreflevelseen = static_cast<CCTK_INT*>(
          CCTK_VarDataPtr(cctkGH, 0,
                          CCTK_THORNSTRING "::reflevelseen[0]"));
        assert(myreflevelseen);
        myreflevelseen += var * cctk_ash[0] * cctk_ash[1] * cctk_ash[2];
        assert(cctk_lsh[0] == cctk_ash[0]);
        assert(cctk_lsh[1] == cctk_ash[1]);
        assert(cctk_lsh[2] == cctk_ash[2]);

        for(int idx = 0 ; idx < cctk_lsh[0]*cctk_lsh[1]*cctk_lsh[2] ; idx++)
        {
          if(myreflevelseen[idx] == -1)
          {
            if(nunset_points_var == 0)
              nunset_vars += 1;
            nunset_points_var += 1;
            if(verbosity >= 8)
            {
              char *varname = CCTK_FullName(varindex);
              CCTK_VInfo(CCTK_THORNSTRING, "Point (%g,%g,%g) on target level %d was not set for variable '%s'",
                         x[idx],y[idx],z[idx], Carpet::reflevel, varname);
              free(varname);
            }
          }
        }

      } END_LOCAL_COMPONENT_LOOP;
    } END_LOCAL_MAP_LOOP;
    if(nunset_points_var > 0 && verbosity > 1)
    {
      char *varname = CCTK_FullName(varindex);
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "There were %d points that could not be set on target level %d for variable '%s'.",
                 nunset_points_var, Carpet::reflevel, varname);
      free(varname);
    }
    nunset_points += nunset_points_var;
  //} END_REFLEVEL_LOOP;
  }

  if(int(varseen.size()) > max_number_of_read_variables)
  {
    std::ostringstream buf;
    for(varseenmap::const_iterator it = varseen.begin(), end = varseen.end() ;
        it != end ;
        it++) {
      const int varindex = it->first;
      const int var = it->second;
      if(var >= max_number_of_read_variables)
        buf << " " << CCTK_FullVarName(varindex);
    }
    CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
               "Not enough scratch space was allocated to record what was read. max_number_of_read_variables was set to %d but at least %d variables were read. The variables that could not be read properly are:%s.",
               max_number_of_read_variables, int(varseen.size()), buf.str().c_str());
    return; // NOTREACHED
  }

  if(nunset_points > 0)
  {
    CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
               "There were %d points in %d variables that could not be set.",
               nunset_points, nunset_vars);
    return; // NOTREACHED
  }

  // free storage for temp workspace
  const int timelevels = 0; // number of timelevels for out temp. variables
  const int group = CCTK_GroupIndex(CCTK_THORNSTRING "::reflevelseen");
  int ierr = CCTK_GroupStorageDecrease(cctkGH, 1, &group, &timelevels, NULL);
  if(ierr < 0)
  {
    CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
               "Could not deallocate storage for '%s', error = %d",
               CCTK_THORNSTRING "::reflevelseen", ierr);
  }
  varseen.clear();
}

// interpolate a HDF5 patch onto all Carpet patches that overlap
void ReadInterpolate_Interpolate(const cGH * cctkGH, int iteration,
                                 int timelevel, int component, int reflevel,
                                 int varindex, const CCTK_INT lsh[3], const CCTK_REAL origin[3],
                                 const CCTK_REAL delta[3], 
                                 CCTK_REAL const * const vardata, void *token)
{
  DECLARE_CCTK_PARAMETERS;

  // make sure there is storage for temp workspace
  const int timelevels = 1; // number of timelevels for out temp. variables
  const int group = CCTK_GroupIndex(CCTK_THORNSTRING "::reflevelseen");
  int ierr = CCTK_GroupStorageIncrease(cctkGH, 1, &group, &timelevels, NULL);
  if(ierr < 0)
  {
    CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
               "Could not allocate storage for '%s', error = %d",
               CCTK_THORNSTRING "::reflevelseen", ierr);
  }

  // keep track of temporary storage associated with each variable
  if(!varseen.count(varindex)) {
    // allocate storage for temp workspace
    const int var = int(varseen.size());
    varseen[varindex] = var;
    // when more variables are requested than I have buffer space, I ignore the
    // extra ones but abort and inform the user after parsing the files what
    // the correct number would have been
    if(varseen[varindex] < max_number_of_read_variables)
      ClearRefLevelSeen(cctkGH, varseen[varindex]);
  }
  if(varseen[varindex] >= max_number_of_read_variables)
    return;

  const int current_reflevel = GetRefinementLevel(cctkGH);
  const int current_timelevel = GetTimeLevel(cctkGH);
  // for tl != 0 the different rl are not aligned in time so we only want to
  // use the same rl as we are currently on
  // RH: this is not generally possible
  if(current_reflevel != reflevel && current_timelevel != 0)
    return;

  //BEGIN_REFLEVEL_LOOP(cctkGH) { // we run in level mode to look more like an ordinary id thorn
    BEGIN_LOCAL_MAP_LOOP (cctkGH, CCTK_GF) {
      BEGIN_LOCAL_COMPONENT_LOOP (cctkGH, CCTK_GF) {

        DECLARE_CCTK_ARGUMENTS;
        CCTK_INT *myreflevelseen = static_cast<CCTK_INT*>(
          CCTK_VarDataPtr(cctkGH, 0,
                          CCTK_THORNSTRING "::reflevelseen[0]"));
        assert(myreflevelseen);
        myreflevelseen += varseen[varindex] * cctk_ash[0] * cctk_ash[1] *
                          cctk_ash[2];
        assert(cctk_lsh[0] == cctk_ash[0]);
        assert(cctk_lsh[1] == cctk_ash[1]);
        assert(cctk_lsh[2] == cctk_ash[2]);

        CCTK_REAL *xyz[3] = {x,y,z};
        // region for which we have enough inner and ghost points to interpolate,
        // assuming the interpolator needs nghostzones ghosts
        // we allow usage of one sided interpolation at outer boundaries and
        // symmetry boundaries, where we have no other choice
        // NOTE: this will give unexpected results should the read in data
        // happen to have a component end at the outer boundary of the new
        // domain and then have further components further out. In this case
        // the value the is interpolated depends on the order in which the
        // datasets appear in the file. The HDF5 files have cctk_bbox attribute
        // however I suspect it to not be trustworthy once we start merging
        // files or use the slicer to get data out of files.
        int nghostzones[6];
        for(int i = 0 ; i < 6 ; i++) {
          if(interpolator_half_width == -1)
            nghostzones[i] = cctk_nghostzones[i];
          else
            nghostzones[i] = interpolator_half_width;
        }
        int require_ghosts[6];
        for(int d = 0 ; d < 3 ; d++) { // allow one side interpolation if read in patch covers outer boundary
          int ijk[3] = {0,0,0};
          ijk[d] = cctk_lsh[d]-1;
          CCTK_REAL xyzmin = xyz[d][CCTK_GFINDEX3D(cctkGH, 0,0,0)];
          CCTK_REAL xyzmax = xyz[d][CCTK_GFINDEX3D(cctkGH, ijk[0],ijk[1],ijk[2])];
          require_ghosts[2*d+0] = !(origin[d] <= xyzmin && cctk_bbox[2*d+0]);
          require_ghosts[2*d+1] = !(origin[d]+(lsh[d]-1)*delta[d] >= xyzmax && cctk_bbox[2*d+1]);
        }
        CCTK_REAL xmin[3] = {origin[0]+require_ghosts[0]*(nghostzones[0]-1)*delta[0],
                             origin[1]+require_ghosts[2]*(nghostzones[1]-1)*delta[1],
                             origin[2]+require_ghosts[4]*(nghostzones[2]-1)*delta[2]};
        CCTK_REAL xmax[3] = {origin[0]+(lsh[0]-1-require_ghosts[1]*(nghostzones[0]-1))*delta[0],
                             origin[1]+(lsh[1]-1-require_ghosts[3]*(nghostzones[1]-1))*delta[1],
                             origin[2]+(lsh[2]-1-require_ghosts[5]*(nghostzones[2]-1))*delta[2]};

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

          if(myreflevelseen[idx] <= reflevel && // need <= since we re-use the same level information for all output grid functions
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
              myreflevelseen[idx] = reflevel;
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
          CCTK_WARN(CCTK_WARN_ABORT, "cannot get interpolation handle!");
  param_table_handle = Util_TableCreateFromString(interpolator_pars);
  if (param_table_handle < 0)
          CCTK_WARN(CCTK_WARN_ABORT, "cannot create parameter table!");


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

/********************************************************************
 *********************     Internal Routines   **********************
 ********************************************************************/

// set all seen refinement level data to -1 so that the coarsest on triggers
static void ClearRefLevelSeen(const cGH * cctkGH, const int var)
{
  DECLARE_CCTK_PARAMETERS;

  assert(var < max_number_of_read_variables);

  //BEGIN_REFLEVEL_LOOP(cctkGH) { // we run in level mode to look more like an ordinary id thorn
    BEGIN_LOCAL_MAP_LOOP (cctkGH, CCTK_GF) {
      BEGIN_LOCAL_COMPONENT_LOOP (cctkGH, CCTK_GF) {

        DECLARE_CCTK_ARGUMENTS;
        CCTK_INT *myreflevelseen = static_cast<CCTK_INT*>(
          CCTK_VarDataPtr(cctkGH, 0,
                          CCTK_THORNSTRING "::reflevelseen[0]"));
        assert(myreflevelseen);
        myreflevelseen += var * cctk_ash[0] * cctk_ash[1] * cctk_ash[2];
        assert(cctk_lsh[0] == cctk_ash[0]);
        assert(cctk_lsh[1] == cctk_ash[1]);
        assert(cctk_lsh[2] == cctk_ash[2]);

        for(int idx = 0 ; idx < cctk_lsh[0]*cctk_lsh[1]*cctk_lsh[2] ; idx++)
          myreflevelseen[idx] = -1;

      } END_LOCAL_COMPONENT_LOOP;
    } END_LOCAL_MAP_LOOP;
  //} END_REFLEVEL_LOOP;
}

