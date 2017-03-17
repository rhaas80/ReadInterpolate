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
#include "cctk_Functions.h"
#include "cctk_GNU.h" // for regex.h

#include "util_String.h"

#include "CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h"

#include "readinterpolate.h"

/********************************************************************
 *********************     Local Data Types   ***********************
 ********************************************************************/

/* we cache the datasets in the files after having iterated over them once so
 * that we don't iterate over and over again */
typedef int ivect[3];

struct patch {
  struct patch * next;
  char * patchname;
  int vindex;
  int map;
  int reflevel;
  int timestep;
  int timelevel;
  int component;
  hsize_t rank;
  int lsh[3];
  double time;
  double origin[3];
  double delta[3];
  hid_t datatype;
  hsize_t objectsize;
  int cached; // all member variables are valid
};
typedef struct patch patch_t;

// structure describing the contents of an HDF5 file to read from
struct file {
  struct file * next;
  char * filename;
  patch_t * patches;
};
typedef struct file file_t;

struct pulldata
{
  CCTK_INT hasbeenread;    // only read data once
  CCTK_REAL * vardata;     // a buffer suffiently large to hold all data
  hid_t from;              // HDF5 group (file) containing patch
  const patch_t * patch;   // dataset that hold the data
};

#define METADATA_GROUP "Parameters and Global Attributes"
#define ALL_PARAMETERS "All Parameters"
#define MAX_N_REGEX 200 // maximum number of comma separated reggular expression supported

/********************************************************************
 *********************     Local Data         ***********************
 ********************************************************************/
static int regexmatchedsomething[MAX_N_REGEX];
static int * varsread = NULL;
static file_t * filecache = NULL;

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
static char *trim(char *s);

static herr_t ParseObject (hid_t from, const char *objectname, void *calldata);
static patch_t *RecordGFPatch(hid_t from, const char *objectname);
static void HandleDataset(hid_t from, const patch_t *patch, cGH *cctkGH);
static int UseThisDataset(const patch_t *patch, const int current_timelevel);
static int ParseDatasetNameTags(const char *objectsname, char *varname, 
                                int *iteration, int *timelevel, int *map,
                                int *reflevel, int *component);
static int MatchDatasetAgainstRegex(const char *objectname);

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
              CCTK_VError (__LINE__, __FILE__, CCTK_THORNSTRING,\
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
                "No valid HDF5 file with basename \"%s/%s\" found", 
                filereader_ID_dir, basename);
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
  ssize_t open_objects;
  CHECK_ERROR (open_objects = H5Fget_obj_count (file,
                                                H5F_OBJ_ALL | H5F_OBJ_LOCAL));
  if(open_objects != 1) {
    char buf[1024];
    CHECK_ERROR(H5Fget_name (file, buf, sizeof(buf)));
    CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
               "There are %d open objects in file %s", (int)open_objects-1,
               buf);
  }
  CHECK_ERROR (H5Fclose (file));

  return retval;
}

// read in a CCTK_INT attribute
static void read_int_attr(hid_t from, const char *attrname, int nelems, CCTK_INT *data)
{
  hid_t attr, dataspace;
  hsize_t attrsize;

  CHECK_ERROR (attr = H5Aopen_name (from, attrname));
  CHECK_ERROR (dataspace = H5Aget_space (attr));

  CHECK_ERROR (attrsize = H5Sget_simple_extent_npoints (dataspace));
  assert((int)attrsize == nelems);

  CHECK_ERROR (H5Aread (attr, H5T_NATIVE_INT, data));

  CHECK_ERROR (H5Sclose (dataspace));
  CHECK_ERROR (H5Aclose (attr));
}

// read in a CCTK_REAL attribute
static void read_real_attr(hid_t from, const char *attrname, int nelems, CCTK_REAL *data)
{
  hid_t attr, dataspace;
  hsize_t attrsize;

  CHECK_ERROR (attr = H5Aopen_name (from, attrname));
  CHECK_ERROR (dataspace = H5Aget_space (attr));

  CHECK_ERROR (attrsize = H5Sget_simple_extent_npoints (dataspace));
  assert((int)attrsize == nelems);

  CHECK_ERROR (H5Aread (attr, H5T_NATIVE_DOUBLE, data));

  CHECK_ERROR (H5Sclose (dataspace));
  CHECK_ERROR (H5Aclose (attr));
}

/*****************************************************************************/
/*             Routines to decide if to read in a dataset                    */
/*****************************************************************************/

// match against dataset name pattern "thorn::name it=X tl=Y m=Z rl=U c=V"
static int ParseDatasetNameTags(const char *objectname, char *varname, 
                                int *iteration, int *timelevel, int *map,
                                int *reflevel, int *component)
{
  DECLARE_CCTK_PARAMETERS;

  int nread, offset = 0;
  struct {
    const char * tag;
    int * val;
  } tagvals[] = {
    {" tl=%d%n", timelevel},
    {" m=%d%n", map},
    {" rl=%d%n", reflevel},
    {" c=%d%n", component}
  };
  int retval;

  if(sscanf(objectname, "%s it=%d%n", varname, iteration, &nread) == 2)
  {
    offset += nread;
    for(int i = 0 ; i < DIM(tagvals) ; i++)
    {
      const char *coda = objectname+offset;
      const int didmatch = sscanf(coda, tagvals[i].tag, tagvals[i].val, &nread) == 1;
      if(didmatch)
        offset += nread;
      else
        *tagvals[i].val = 0;
      if(verbosity >= 5)
      {
        CCTK_VInfo(CCTK_THORNSTRING, "Testing dataset name coda '%s' against tag '%s': found %smatch and will use value %d",
                    coda, tagvals[i].tag, didmatch?"":"no ", *tagvals[i].val);
      }
    }
    retval = (objectname[offset] == '\0'); // was there any leftover stuff we could not identify?
  }
  else
    retval = 0;

  return retval;
}

// trim whitespace from beginning and end of string, changes input array, save
// to pass NULL
static char *trim(char *s)
{
  if(s != NULL)
  {
    for(int i = strlen(s) - 1 ; i >= 0 && isspace(s[i]) ; --i)
      s[i] = '\0';
    while(*s != '\0' && isspace(*s))
      s++;
  }

  return s;
}

// match a string (object name) against a number of regular expressions
static int MatchDatasetAgainstRegex(const char *objectname)
{
  DECLARE_CCTK_PARAMETERS;

  int retval = 0;

  regmatch_t pmatch[8];
  char * dataset_regex = strdup(only_these_datasets), *scratchptr;
  assert(dataset_regex);

  int iregex = 0; // count which regex we re looking at
  for(char *regex = trim(strtok_r(dataset_regex, ",", &scratchptr)) ;
      regex != NULL ;
      regex = trim(strtok_r(NULL, ",", &scratchptr)), iregex += 1)
  {
    // strip white space from beginning and end of pattern
    const int matched = CCTK_RegexMatch(objectname, regex, DIM(pmatch), pmatch);
    if(matched > 0)
    {
      retval = 1;
      regexmatchedsomething[iregex] = 1; // record that this regular expression matched at least once
      break;
    }
    else if(matched < 0)
    {
      CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                 "Invalid regular expression '%s': does not compile", regex);
      // NOTREACHED
    }
  }
  if(verbosity >= 4)
  {
    CCTK_VInfo(CCTK_THORNSTRING, "Tested dataset '%s' against regex(s) '%s': %smatch",
               objectname, only_these_datasets, retval ? "" : "no ");
  }

  free(dataset_regex);

  return retval;
}

// inspect dataset and create patch structure for it if it is a grid function
// dataset for a grid function Cactus knows about
static patch_t *RecordGFPatch(hid_t from, const char *objectname)
{
  DECLARE_CCTK_PARAMETERS;

  patch_t *patch = NULL;

  H5G_stat_t object_info;
  CHECK_ERROR (H5Gget_objinfo (from, objectname, 0, &object_info));
  if (object_info.type == H5G_DATASET)
  {
    char varname[1042];
    int iteration, timelevel, map, reflevel, component, varindex;
    // we are interested in datasets only - skip anything else
    if(ParseDatasetNameTags(objectname, varname, &iteration, &timelevel, &map, &reflevel, &component))
    {
      varindex = CCTK_VarIndex(varname);
      const int is_gf = (varindex >= 0 && CCTK_GroupTypeFromVarI(varindex) == CCTK_GF);
      if (is_gf)
      {
        if(verbosity >= 4)
        {
          CCTK_VInfo(CCTK_THORNSTRING,
                     "Tested that '%s' is a known grid function: yes",
                       objectname);
        }

        // add new dataset to cache
        // TODO: make this work with objects in subdirs rather than just /
        patch = calloc(1, sizeof(*patch));
        patch->patchname = strdup(objectname);

        patch->vindex = varindex;
        patch->map = map;
        patch->reflevel = reflevel;
        patch->timestep = iteration;
        patch->timelevel = timelevel;
        patch->component = component;

        hsize_t dims[DIM(patch->lsh)], ndims;
        hid_t dataset, dataspace;

        CHECK_ERROR (dataset = H5Dopen (from, patch->patchname));
        CHECK_ERROR (dataspace = H5Dget_space (dataset));

        int typesize, vartype;

        vartype = CCTK_VarTypeI(patch->vindex);
        assert(vartype >= 0);

        // get storage size for data
        if(vartype == CCTK_VARIABLE_REAL) {
          patch->datatype = H5T_NATIVE_DOUBLE;
          typesize = sizeof(CCTK_REAL);
        } else if(vartype == CCTK_VARIABLE_INT) {
          patch->datatype = H5T_NATIVE_INT;
          typesize = sizeof(CCTK_INT);
        } else {
          CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                      "Do not know how to handle CCTK type class %d", vartype);
        }

        patch->objectsize = H5Sget_select_npoints (dataspace) * typesize;
        assert(patch->objectsize > 0);

        // store cctk_lssh and grid structure
        {
          CHECK_ERROR (ndims = H5Sget_simple_extent_ndims(dataspace));
          assert(ndims == DIM(patch->lsh));
          CHECK_ERROR (H5Sget_simple_extent_dims(dataspace, dims, NULL));

          read_real_attr(dataset, "origin", 3, patch->origin);
          read_real_attr(dataset, "delta", 3, patch->delta);
          // Carpet has a bug where origin is incorrect for cell centered data,
          // fix this here
          // unfortunatly there is no foolproof way to detect is a givne HDF5 file
          // was written using the old incorrect code or just has strange setting
          // for the origin of the grid. Eg. for "normal" grids that are symmetric
          // around the origin and have an even number of cells, the origin
          // coordinate is always given by origin = delta * (i + 0.5) where i is
          // some integer. However nothing stops a user from offsetting their grid
          // by half a step thus adding just such an offset.
          if(fix_cell_centered_origins)
          {
            int ioffset[3], ioffsetdenom[3];
            read_int_attr(dataset, "ioffset", 3, ioffset);
            read_int_attr(dataset, "ioffsetdenom", 3, ioffsetdenom);
            for(int i = 0; i < 3 ; i++)
            {
              patch->origin[i] += patch->delta[i] *
                                    ((double)ioffset[i]/ioffsetdenom[i]);
            }
          }

          for(int i = 0 ; i < 3 ; i++)
            patch->origin[i] += shift_read_datasets_by[i];

          const int map_is_cartesian = 1;
#if 0 // TODO: Make something like this work
          HDF5_BEGIN_TRY {
            read_int_attr(dataset, "MapIsCartesian", &map_is_cartesian);
          } HDF5_END_TRY;
#endif
          assert(map_is_cartesian); // TODO: support multipatch

          CCTK_INT size = 1;
          for(int d = 0 ; d < (int)ndims ; d++)
          {
            assert ((int)ndims-1-d>=0 && ndims-1-d<ndims);
            patch->lsh[d] = (CCTK_INT)dims[ndims-1-d]; // HDF5 has the slowest changing direction first, Cactus the fastest
            size *= patch->lsh[d];
          }
          if(size*typesize != (int)patch->objectsize) {
            CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                        "Unexpected size %d bytes of dataset '%s' does not agree "
                        "with size of CCTK_REAL or CCTK_INT dataset (%d).",
                        (int)patch->objectsize, patch->patchname, size*typesize);
          }

          if(verbosity >= 3)
          {
            CCTK_VInfo(CCTK_THORNSTRING, "Dataset '%s' sizes: origin=(%g,%g,%g), "
                       "delta=(%g,%g,%g), lsh=(%d,%d,%d), map_is_cartesian=%d, iteration=%d,"
                       " component=%d, reflevel=%d",
                       patch->patchname,
                       patch->origin[0],patch->origin[1],patch->origin[2],
                       patch->delta[0],patch->delta[1],patch->delta[2],
                       patch->lsh[0],patch->lsh[1],patch->lsh[2], map_is_cartesian,
                       patch->timestep, patch->component, patch->reflevel);
          }
        }

        read_real_attr(dataset, "time", 1, &patch->time);
        patch->cached = 1;

        CHECK_ERROR (H5Sclose (dataspace));
        CHECK_ERROR (H5Dclose (dataset));
      } // is_gf
    } // ParseDataSets
    else
    {
      if(verbosity >= 1)
      {
        CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                   "Objectname '%s' could not be fully parsed.", objectname);
      }
    }
  }

  return patch;
}

// check the object name against
// * any of the known dataset patterns
// * the user specified given regex
// * the list of existing Cactus variables
// TODO: move into function of its own
static int UseThisDataset(const patch_t *patch, const int current_timelevel)
{
  DECLARE_CCTK_PARAMETERS;

  int retval = 0;

  const int matches_regex = MatchDatasetAgainstRegex(patch->patchname);
  const int use_this_timelevel = read_only_timelevel_0 ?
                                 patch->timelevel == 0 :
                                 current_timelevel == patch->timelevel;
  if(verbosity >= 4)
  {
    CCTK_VInfo(CCTK_THORNSTRING, "Tested that '%s' should be read for timelevel %d: %s",
               patch->patchname, current_timelevel,
               use_this_timelevel ? "yes" : "no");
  }

  // skip some reflevels if we already know we won't need them
  const int is_desired_patch = use_this_timelevel             &&
                               (patch->map == 0)              &&
                               (minimum_reflevel <= patch->reflevel) &&
                               (patch->reflevel <= maximum_reflevel);

  const int is_real = CCTK_VarTypeI(patch->vindex) == CCTK_VARIABLE_REAL;

  retval = matches_regex && is_desired_patch && is_real;
  assert(!retval || patch->cached);

  // TODO: handle integer variables
  if(!retval && !is_real)
  {
    static char *have_warned = NULL;
    if(have_warned == NULL)
    {
      // calloc initializes to zero
      have_warned = calloc(CCTK_NumVars(), sizeof(char));
      assert(have_warned != NULL);
    }
    if((verbosity == 1 && !have_warned[patch->vindex]) || verbosity >= 2)
    {
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Skipping integer variable '%s'. Do not know how to interpolate integers.",
                 patch->patchname);
      have_warned[patch->vindex] = 1;
    }
  }

  return retval;
}

// check if object is a dataset, then read it in and interpolate onto all
// overlapping grids
static void HandleDataset(hid_t from, const patch_t *patch, cGH *cctkGH)
{
  DECLARE_CCTK_PARAMETERS;

  const int current_timelevel = GetTimeLevel(cctkGH);
  CCTK_REAL * vardata;

  if(verbosity >= 3)
    CCTK_VInfo(CCTK_THORNSTRING, "Checking out dataset '%s'", patch->patchname);

  if(UseThisDataset(patch, current_timelevel))
  {
    struct pulldata pd;

    if(verbosity >= 2)
      CCTK_VInfo(CCTK_THORNSTRING, "Using dataset '%s'", patch->patchname);

    vardata = malloc (patch->objectsize);
    if (vardata == NULL)
    {
      CCTK_VError (__LINE__, __FILE__, CCTK_THORNSTRING,
                  "failled to allocate %d bytes of memory for %s, giving up",
                  (int) patch->objectsize, patch->patchname);
      return; // NOTREACHED
    }

    // we hold of the actual read until really asked for in PullData
    pd.hasbeenread = 0;
    pd.vardata = vardata;
    pd.from = from;
    pd.patch = patch;

    // interpolate onto all overlapping grid patches
    varsread[patch->vindex] = 1;
    ReadInterpolate_Interpolate(cctkGH, patch->timestep, patch->timelevel,
                                patch->component, patch->reflevel, patch->time,
                                patch->vindex, patch->lsh, patch->origin,
                                patch->delta, vardata, &pd);
    
    // free data
    assert(vardata);
    free(vardata);
    vardata = NULL;
  } // if(UseThisDataset
  else
  {
    if(verbosity >= 2)
      CCTK_VInfo(CCTK_THORNSTRING, "Not using dataset '%s' after all.", patch->patchname);
  }
}

// check if object is a dataset, then read it in and interpolate onto all
// overlapping grids
static herr_t ParseObject (hid_t from, const char *objectname, void *calldata)
{
  DECLARE_CCTK_PARAMETERS;
  file_t * file = ((void**)calldata)[0];
  void * cctkGH = ((void**)calldata)[1];

  if(verbosity >= 3)
    CCTK_VInfo(CCTK_THORNSTRING, "Parsing dataset '%s'", objectname);

  patch_t * patch = RecordGFPatch(from, objectname);
  if(patch != NULL)
  {
    if(verbosity >= 3)
      CCTK_VInfo(CCTK_THORNSTRING, "Recording dataset '%s'", objectname);

    patch->next = file->patches;
    file->patches = patch;
    HandleDataset (from, patch, cctkGH);
  }

  return 0;
}

// actually read data into buffer allocated ine ParseObject. This is a separate
// function to avoid reading in datasets we don't need at all since they do not
// overlap the local components.
void ReadInterpolate_PullData(void * token)
{
  DECLARE_CCTK_PARAMETERS;

  struct pulldata * pd = (struct pulldata *)token;

  if(!pd->hasbeenread)
  {
    hid_t dataset;

    if(verbosity >= 2)
      CCTK_VInfo(CCTK_THORNSTRING, "Reading data from dataset '%s'", pd->patch->patchname);

    CHECK_ERROR (dataset = H5Dopen (pd->from, pd->patch->patchname));
    CHECK_ERROR (H5Dread (dataset, pd->patch->datatype, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, pd->vardata));
    CHECK_ERROR (H5Dclose (dataset));
    pd->hasbeenread = 1;
  }
}

// scheduled routine, doesn't really do very much
void ReadInterpolate_Read(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;


  varsread = calloc(CCTK_NumVars(), sizeof(*varsread));
  assert(varsread);

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
          CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                     "Could not open file with basename '%s/%s' to read initial data",
                     filereader_ID_dir, fn);
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

          // find cached file content if it exists or set up new one otherwise
          file_t * file = NULL;
          for(file = filecache ; file != NULL ; file = file->next)
          {
            if(strcmp(full_fn, file->filename) == 0)
              break;
          }

          if(file == NULL)
          {
            file = calloc(1, sizeof(*file));
            file->filename = strdup(full_fn);
            file->next = filecache;
            filecache = file;

            void *calldata[2] = {file, cctkGH};
            CHECK_ERROR (H5Giterate (fh, "/", NULL, ParseObject, calldata));
            if(verbosity >= 1 && H5Fget_obj_count(fh, H5F_OBJ_ALL) > 1)
            {
              CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                         "Leaked %d HDF5 objects when parsing file '%s'.", (int)H5Fget_obj_count(fh, H5F_OBJ_ALL) - 1, full_fn);
            }
          }
          else
          {
            for(patch_t *patch = file->patches ;
                patch != NULL ;
                patch = patch->next)
            {
              HandleDataset(fh, patch, cctkGH);
            }
          }
          ssize_t open_objects;
          CHECK_ERROR (open_objects =
            H5Fget_obj_count (fh, H5F_OBJ_ALL | H5F_OBJ_LOCAL));
          if(open_objects != 1) {
            char buf[1024];
            CHECK_ERROR(H5Fget_name (fh, buf, sizeof(buf)));
            CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                       "There are %d open objects in file %s",
                       (int)open_objects-1, buf);
          }
          CHECK_ERROR (H5Fclose (fh));

          free((void*)full_fn);
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
    for(const char *regex = strtok_r(trim(dataset_regex), ",", &scratchptr) ;
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
      CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                 "Some regular expresion(s) did not match anything.");
      return; // NOTREACHED
    }
  }

  // output names of variables for which we read data so that users knows what we did
  if(verbosity >= 1)
  {
    const int numvars = CCTK_NumVars();
    for(int varindex = 0 ; varindex < numvars ; varindex++)
    {
      if(varsread[varindex])
      {
        char * fullname = CCTK_FullName(varindex);
        assert(fullname);
        CCTK_VInfo(CCTK_THORNSTRING, "Read data for '%s'", fullname);
        free(fullname), fullname = NULL;
      }
    }
  }

  // warn if any variable was not completely set
  ReadInterpolate_CheckAllPointsSet(cctkGH);

  free(varsread), varsread = NULL;
}


// scheduled routine, free cache data once done (can be ~GB I think)
void ReadInterpolate_FreeCache(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;

  if(verbosity >= 3)
    CCTK_VInfo(CCTK_THORNSTRING, "Freeing dataset cache");

  CCTK_DisableGroupStorage(cctkGH, "ReadInterpolate::timeread");

  for(file_t *file = filecache, *next_file = NULL ;
      file != NULL ;
      next_file = file->next, free(file), file = next_file)
  {
    free(file->filename);
    file->filename = NULL;
    for(patch_t *patch = file->patches, *next_patch = NULL;
        patch != NULL ;
        next_patch = patch->next, free(patch), patch = next_patch)
    {
      free(patch->patchname);
      patch->patchname = NULL;
    }
    file->patches = NULL;
  }
  filecache = NULL;

  ssize_t open_objects;
  const unsigned int types = H5F_OBJ_ALL;
  CHECK_ERROR (open_objects = H5Fget_obj_count (H5F_OBJ_ALL, types));
  if(open_objects != 0) {
    hid_t objs[100]; // if there's more than 100 leaked object we are lost
    ssize_t got_objs;
    CHECK_ERROR(got_objs = H5Fget_obj_ids (H5F_OBJ_ALL, types,
                                sizeof(objs)/sizeof(objs[0]), objs));

    CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
               "There are %d (%d) open objects in application",
               (int)open_objects, (int)got_objs);
    for(int i = 0 ; i < open_objects ; i++) {
      char buf[1024] = "<anonymous>";
      H5I_type_t type;
      static const struct typename_t {
        const char *name;
        H5I_type_t type;
      } typenames[] = {
        {"FILE", H5I_FILE},
        {"GROUP", H5I_GROUP},
        {"DATATYPE", H5I_DATATYPE},
        {"DATASPACE", H5I_DATASPACE},
        {"DATASET", H5I_DATASET},
        {"ATTR", H5I_ATTR},
        {"BADID", H5I_BADID},
      };
      H5E_BEGIN_TRY {
      H5Iget_name(objs[i], buf, sizeof(buf));
      } H5E_END_TRY; // memory only types fail if one queries for their name
      CHECK_ERROR(type = H5Iget_type(objs[i]));
      for(size_t t = 0 ; t < sizeof(typenames)/sizeof(typenames[0]) ; t++) {
        if(type == typenames[t].type) {
          CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                     "object name: %s type: %s", buf,
                     typenames[t].name);
        }
      }
    }
  }

  // force close HDF5 library (and all open files)
  CHECK_ERROR (H5close());
}


// scheduled routine, doesn't really do very much
void ReadInterpolate_EnforceSymmetry(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  int failedtoapplysymmetries = 0;
  const int numgroups = CCTK_NumGroups();
  for(int groupindex = 0 ; groupindex < numgroups ; groupindex++)
  {
    const int firstvaringroup = CCTK_FirstVarIndexI(groupindex);
    const int numvars = CCTK_NumVarsInGroupI(groupindex);
    for(int varindex = 0 ; varindex < numvars ; varindex++)
    {
      if(varsread[firstvaringroup+varindex])
      {
        // TODO: use boundary width rather than ghost width
        const int ierr = Boundary_SelectGroupForBCI(cctkGH, CCTK_ALL_FACES, cctk_nghostzones[0], -1, groupindex, "none");
        if(!ierr)
        {
          CCTK_SyncGroupI(cctkGH, groupindex);
        }
        else
        {
          failedtoapplysymmetries = 1;
          char * fullname = CCTK_FullName(firstvaringroup+varindex);
          assert(fullname);
          CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                     "Failed to select symmetry condition for group containing '%s'",
                     fullname);
          free(fullname), fullname = NULL;
        }
        break; // only need to select bc once per group
      }
    }
  }
  if(failedtoapplysymmetries)
  {
    CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
               "Some symmetry conditions could not be enforced.");
    return; // NOTREACHED
  }
}
