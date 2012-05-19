#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

// force interface to be 1.6
#define H5_USE_16_API 1
#include <hdf5.h>

#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"
#include "cctk_GNU.h" // for regex.h

#include "util_String.h"

#include "CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h"

#include "readinterpolate.h"

/********************************************************************
 *********************     Local Data Types   ***********************
 ********************************************************************/
struct pulldata
{
  CCTK_INT hasbeenread; // only read data once
  CCTK_REAL * vardata;  // a buffer suffiently large to hold all data
  hid_t dataset;        // dataset that hold the data
  hid_t datatype;       // data type of variable (must be REAL)
};

#define METADATA_GROUP "Parameters and Global Attributes"
#define ALL_PARAMETERS "All Parameters"
#define MAX_N_REGEX 200 // maximum number of comma separated reggular expression supported

/********************************************************************
 *********************     Local Data         ***********************
 ********************************************************************/
static int regexmatchedsomething[MAX_N_REGEX];

/********************************************************************
 ********************* Other Routine Prototypes *********************
 ********************************************************************/

int CCTK_RegexMatch (const char *string,
                     const char *pattern,
                     const int nmatch,
                     regmatch_t *pmatch);

/********************************************************************
 ********************    Internal Routines   ************************
 ********************************************************************/
static int get_nioprocs(const cGH * cctkGH, const char *basename);
static void read_int_attr(hid_t from, const char *attrname, int nelems,
                          CCTK_INT *data);
static void read_real_attr(hid_t from, const char *attrname, int nelems,
                           CCTK_REAL *data);

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

#define DIM(x) ((int)(sizeof(x)/sizeof(x[0])))

/*****************************************************************************/
/*             some helper function to make handling hdf5 simpler            */
/*****************************************************************************/

// find out if there are multiple file a.file_*.h5 and return their number if so
static int get_nioprocs(const cGH * cctkGH, const char *basename)
{
  DECLARE_CCTK_PARAMETERS;

  CCTK_INT retval, ioproc;
  char * filename;
  char * filenames[4];
  int nfilenames = 0;

  hid_t metadata, file;

  // first try to open a chunked file written on this processor
  ioproc = CCTK_MyProc (cctkGH);
  filename = IOUtil_AssembleFilename (cctkGH, basename, "", ".h5",
                                      FILEREADER_DATA, ioproc, 0);

  // try to open the file (prevent HDF5 error messages if it fails)
  H5E_BEGIN_TRY {
    assert(nfilenames < DIM(filenames)-1);
    filenames[nfilenames++] = filename;
    file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  } H5E_END_TRY;

  // if that failed, try a chunked file written on processor 0
  // (which always is an I/O proc)
  if (file < 0) {
    ioproc = 0;
    filename = IOUtil_AssembleFilename (cctkGH, basename, "", ".h5",
                                        FILEREADER_DATA, ioproc, 0);
    H5E_BEGIN_TRY {
      filenames[nfilenames++] = filename;
      file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    } H5E_END_TRY;
  }

  // if that still failed, try an unchunked file
  // (which is always written on processor 0)
  if (file < 0) {
    ioproc = 0;
    filename = IOUtil_AssembleFilename (cctkGH, basename, "", ".h5",
                                        FILEREADER_DATA, ioproc, 1);
    H5E_BEGIN_TRY {
      filenames[nfilenames++] = filename;
      file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    } H5E_END_TRY;
  }

  // return if no valid checkpoint could be found otherwise just free the
  // memory for the filenames
  if (file < 0) {
    CCTK_VWarn (CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                "No valid HDF5 file with basename \"%s\" found", basename);
  }
  for (int i = 0 ; i < nfilenames ; i++)
  {
    if (file < 0) {
      CCTK_VWarn (CCTK_WARN_PICKY, __LINE__, __FILE__, CCTK_THORNSTRING,
                  "Tried filename \"%s\"", filenames[i]);
    }
    free((void*)filenames[i]);
    filenames[i]=NULL;
  }
  if(file < 0)
    return -1;

  CHECK_ERROR( metadata = H5Gopen (file, METADATA_GROUP));

  read_int_attr(metadata, "nioprocs", 1, &retval);

  CHECK_ERROR (H5Gclose (metadata));
  CHECK_ERROR (H5Fclose (file));

  return retval;
}

// read in a CCTK_INT attribute
static void read_int_attr(hid_t from, const char *attrname, int nelems, CCTK_INT *data)
{
  hid_t attr, datatype, dataspace;
  hsize_t attrsize;

  CHECK_ERROR (attr = H5Aopen_name (from, attrname));
  CHECK_ERROR (datatype = H5Aget_type (attr));
  CHECK_ERROR (dataspace = H5Aget_space (attr));

  assert(datatype = H5T_NATIVE_INT);

  CHECK_ERROR (attrsize = H5Sget_simple_extent_npoints (dataspace));
  assert((int)attrsize == nelems);

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

// check if object is a dataset, then read it in and interpolate onto all
// overlapping grids
static herr_t ParseObject (hid_t from,
                          const char *objectname,
                          void *cctkGH)
{
  DECLARE_CCTK_PARAMETERS;

  int use_dataset;

  char varname[1042];
  int iteration, reflevel, component, timelevel, map, varindex;
  CCTK_INT lsh[3], map_is_cartesian;
  CCTK_REAL delta[DIM(lsh)], origin[DIM(lsh)];
  CCTK_REAL * vardata = NULL;


  if(verbosity >= 3)
    CCTK_VInfo(CCTK_THORNSTRING, "Checking out dataset '%s'", objectname);

  // check the object name against
  // * any of the known dataset patterns
  // * the user specified given regex
  // * the list of existing Cactus variables
  // TODO: move into function of its own
  {
    regmatch_t pmatch[8];
    char * dataset_regex = strdup(only_these_datasets), *scratchptr;
    int matches_regex, is_known_variable, is_desired_patch;

    assert(dataset_regex);

    matches_regex = 0;
    int iregex = 0; // count which regex we re looking at
    for(char *regex = strtok_r(dataset_regex, ",", &scratchptr) ;
        regex != NULL ;
        regex = strtok_r(NULL, ",", &scratchptr), iregex += 1)
    {
      // strip white space from beginning and end of pattern
      for(int i = strlen(regex) - 1 ; i >= 0 && isspace(regex[i]) ; --i) regex[i] = '\0';
      if(CCTK_RegexMatch(objectname, regex+strspn(regex, " \n\t\v"), DIM(pmatch), pmatch))
      {
        matches_regex = 1;
        regexmatchedsomething[iregex] = 1; // record that this regular expression matched at least once
        break;
      }
    }
    if(verbosity >= 4)
    {
      CCTK_VInfo(CCTK_THORNSTRING, "Tested dataset '%s' against regex(s) '%s': %smatch",
                 objectname, only_these_datasets, matches_regex ? "" : "no ");
    }

    is_known_variable = 0;
    is_desired_patch = 0;
    // match against dataset name pattern "thorn::name it=X tl=Y m=Z rl=U c=V"
    {
      int nread, offset = 0;
      struct {
        const char * tag;
        int * val;
      } tagvals[] = {
        {" tl=%d%n", &timelevel},
        {" m=%d%n", &map},
        {" rl=%d%n", &reflevel},
        {" c=%d%n", &component}
      };

      if(matches_regex && sscanf(objectname, "%s it=%d%n", varname, &iteration, &nread) == 2)
      {
        offset += nread;
        for(int i = 0 ; i < DIM(tagvals) ; i++)
        {
          int didmatch;
          if((didmatch = sscanf(objectname+offset, tagvals[i].tag, tagvals[i].val, &nread)) == 1)
            offset += nread;
          else
            *tagvals[i].val = 0;
          if(verbosity >= 5)
          {
            CCTK_VInfo(CCTK_THORNSTRING, "Testing dataset name coda '%s' against tag '%s': found %smatch and will use value %d",
                        objectname+offset-didmatch*nread, tagvals[i].tag, didmatch?"":"no ", *tagvals[i].val);
          }
        }
        if(objectname[offset] == '\0') // there was leftover stuff we could not identify
        {
          // skip some reflevels if we already know we won't need them
          is_desired_patch = (timelevel == 0)               && 
                             (map == 0)                     &&
                             (minimum_reflevel <= reflevel) &&
                             (reflevel <= maximum_reflevel);

          if (is_desired_patch && (varindex = CCTK_VarIndex(varname)) >= 0)
          {
            is_known_variable = 1;
            if(verbosity >= 4)
            {
              CCTK_VInfo(CCTK_THORNSTRING, "Tested dataset '%s': match", objectname);
            }
          }
        }
        else
        {
          if(verbosity >= 1)
          {
            CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                       "Objectname '%s' could not be fully parsed.", objectname);
          }
        }
      }
    }
    use_dataset = is_known_variable && matches_regex && is_desired_patch;

    free(dataset_regex);
  }

  if(use_dataset)
  {
    hsize_t dims[DIM(lsh)], ndims, objectsize;
    hid_t dataset, dataspace, datatype;
    struct pulldata pd;

    CHECK_ERROR (dataset = H5Dopen (from, objectname));
    CHECK_ERROR (datatype = H5Dget_type (dataset));
    CHECK_ERROR (dataspace = H5Dget_space (dataset));

    assert(datatype = H5T_NATIVE_DOUBLE);

    if(verbosity >= 2)
      CCTK_VInfo(CCTK_THORNSTRING, "Reading dataset '%s'", objectname);
    objectsize = H5Sget_select_npoints (dataspace) * H5Tget_size (datatype);
    assert(objectsize > 0);
    assert(vardata == NULL);
    vardata = malloc (objectsize);
    if (vardata == NULL)
    {
      CCTK_VWarn (CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
                  "failled to allocate %d bytes of memory for %s, giving up",
                  (int) objectsize, objectname);
      return -1; // NOTREACHED
    }
    // we hold of the actual read until really asked for in PullData
    pd.hasbeenread = 0;
    pd.vardata = vardata;
    pd.dataset = dataset;
    pd.datatype = datatype;

    // store cctk_lssh and grid structure
    {
      CHECK_ERROR (ndims = H5Sget_simple_extent_ndims(dataspace));
      assert(ndims == DIM(lsh));
      CHECK_ERROR (H5Sget_simple_extent_dims(dataspace, dims, NULL));

      read_real_attr(dataset, "origin", 3, origin);
      read_real_attr(dataset, "delta", 3, delta);

      for(int i = 0 ; i < 3 ; i++)
        origin[i] += shift_read_datasets_by[i];

      map_is_cartesian = 1;
#if 0 // TODO: Make something like this work
      HDF5_BEGIN_TRY {
        read_int_attr(dataset, "MapIsCartesian", &map_is_cartesian);
      } HDF5_END_TRY;
#endif
      assert(map_is_cartesian); // TODO: support multipatch

      CCTK_INT size = 1;
      for(int d = 0 ; d < (int)ndims ; d++)
      {
        assert (ndims-1-d>=0 && ndims-1-d<ndims);
        lsh[d] = (CCTK_INT)dims[ndims-1-d]; // HDF5 has the slowest changing direction first, Cactus the fastest
        size *= lsh[d];
      }
      assert(size*sizeof(CCTK_REAL) == objectsize);

      if(verbosity >= 3)
      {
        CCTK_VInfo(CCTK_THORNSTRING, "Dataset '%s' sizes: origin=(%g,%g,%g), "
                   "delta=(%g,%g,%g), lsh=(%d,%d,%d), map_is_cartesian=%d, iteration=%d,"
                   " component=%d, reflevel=%d",
                   objectname, origin[0],origin[1],origin[2],
                   delta[0],delta[1],delta[2], lsh[0],lsh[1],lsh[2], map_is_cartesian,
                   iteration, component, reflevel);
      }
    }

    // interpolate onto all overlapping grid patches
    ReadInterpolate_Interpolate(cctkGH, iteration, component, reflevel,
                                varindex, lsh, origin, delta, vardata, &pd);
    
    // needs to be after a possible call to PullData!
    CHECK_ERROR (H5Dclose (dataset));
    CHECK_ERROR (H5Sclose (dataspace));
    
    // free data
    assert(vardata);
    free(vardata);
    vardata = NULL;
  } // if(is_known_variable

  return 0;
}

// actually read data into buffer allocated ine ParseObject. This is a separate
// function to avoid reading in datasets we don't need at all since they do not
// overlap the local components.
void ReadInterpolate_PullData(void * token)
{
  struct pulldata * pd = (struct pulldata *)token;

  if(!pd->hasbeenread)
  {
    CHECK_ERROR (H5Dread (pd->dataset, pd->datatype, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, pd->vardata));
    pd->hasbeenread = 1;
  }
}

// scheduled routine, doesn't really do very much
void ReadInterpolate_Read(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  const char * groups[] = {
    CCTK_THORNSTRING "::reflevelseen",
    CCTK_THORNSTRING "::interpthispoint",
    CCTK_THORNSTRING "::interp_coords",
  };

  // allocate storage for temp workspace
  {
    for(int i = 0 ; i < DIM(groups) ; i++)
    {
      const int timelevels = 1; // number of timelevels for out temp. variables
      const int group = CCTK_GroupIndex(groups[i]);
      int ierr = CCTK_GroupStorageIncrease(cctkGH, 1, &group, &timelevels, NULL);
      if(ierr < 0)
      {
        CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "Could not allocate storage for '%s', error = %d",
                   groups[i], ierr);
      }
    }
    ReadInterpolate_ClearRefLevelSeen(cctkGH); // needs to be a C++ function
  }

  // loop over all input files
  {
    char *fns_buf = strdup(files), *scratchptr;
    assert(fns_buf);
    for(const char *fn = strtok_r(fns_buf, " ", &scratchptr) ;
        fn != NULL ;
        fn = strtok_r(NULL, " ", &scratchptr))
    {
        const CCTK_INT nioprocs = get_nioprocs(cctkGH, fn);
        const CCTK_INT myproc = CCTK_MyProc(cctkGH);

        if(nioprocs < 0)
        {
          CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
                     "Could not open file '%s' to read initial data", fn);
          return; // NOTREACHED
        }

        // try to interleave access to the files between all current processors
        for(int i = myproc ; i < myproc + nioprocs ; i++)
        {
          hid_t fh;
          char * full_fn;
          int filenum = i % nioprocs;

          full_fn = IOUtil_AssembleFilename (cctkGH, fn, "", ".h5",
                                             FILEREADER_DATA, filenum, nioprocs == 1);

          if(verbosity >= 1)
            CCTK_VInfo(CCTK_THORNSTRING, "Reading datasets from file '%s'", full_fn);
          
          CHECK_ERROR (fh = H5Fopen (full_fn, H5F_ACC_RDONLY, H5P_DEFAULT));
          CHECK_ERROR (H5Giterate (fh, "/", NULL, ParseObject, cctkGH));
          CHECK_ERROR (H5Fclose (fh));

          free((void*)full_fn);

          if(nioprocs == 1)
            break;
        }
    }
    free(fns_buf);
  }

  // check each regular expression matched at least once
  {
    int allmatched = 1;
    char * dataset_regex = strdup(only_these_datasets), *scratchptr;

    assert(dataset_regex);

    int iregex = 0;
    for(const char *regex = strtok_r(dataset_regex, ",", &scratchptr) ;
        regex != NULL ;
        regex = strtok_r(NULL, ",", &scratchptr), iregex += 1)
    {
      if(!regexmatchedsomething[iregex]) // never matched
      {
        allmatched = 0;
        CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "Regular expresion '%s' did not match anything.",
                   regex);
      }
    }

    free(dataset_regex);

    if(!allmatched)
    {
      CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Some regular expresion(s) did not match anything.");
      return; // NOTREACHED
    }
  }

  // free storage for temp workspace
  {
    ReadInterpolate_CheckAllPointsSet(cctkGH);
    for(int i = 0 ; i < DIM(groups) ; i++)
    {
      const int timelevels = 0; // number of timelevels for out temp. variables
      const int group = CCTK_GroupIndex(groups[i]);
      int ierr = CCTK_GroupStorageDecrease(cctkGH, 1, &group, &timelevels, NULL);
      if(ierr < 0)
      {
        CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "Could not deallocate storage for '%s', error = %d",
                   groups[i], ierr);
      }
    }
  }
}


