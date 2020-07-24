#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"

#include <string.h>
#include <ctype.h>
#include <stdlib.h>

/********************************************************************
 ********************    Internal Routines   ************************
 ********************************************************************/
static char *trim(char *s);

/********************************************************************
 ********************* Other Routine Prototypes *********************
 ********************************************************************/

int CCTK_RegexMatch (const char *string,
                     const char *pattern,
                     const int nmatch,
                     regmatch_t *pmatch);

/*****************************************************************************/
/*                           macro definitions                               */
/*****************************************************************************/
#define DIM(x) ((int)(sizeof(x)/sizeof(x[0])))

/********************************************************************
 ********************    Internal Routines   ************************
 ********************************************************************/

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

/********************************************************************
 ********************* Scheduled Routines ***************************
 ********************************************************************/
void ReadInterpolate_ParamCheck(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  // check that each of the given regular expressions matches at least one
  // Cactus grid function
  char * dataset_regex = strdup(only_these_datasets), *scratchptr;
  assert(dataset_regex);

  int allmatched = 1;
  for(const char *regex = trim(strtok_r(dataset_regex, ",", &scratchptr)) ;
      regex != NULL ;
      regex = trim(strtok_r(NULL, ",", &scratchptr)))
  {
    int any_matched = 0;
    const int num_vars = CCTK_NumVars();
    for(int i = 0 ; i < num_vars ; i++)
    {
      char *varname = CCTK_FullName(i);
      assert(varname);

      regmatch_t pmatch[8];
      const int matched = CCTK_RegexMatch(varname, regex, DIM(pmatch), pmatch);

      free(varname);

      if(matched > 0)
      {
        any_matched = 1;
        break;
      }
      else if(matched < 0)
      {
        CCTK_VError(__LINE__, __FILE__, CCTK_THORNSTRING,
                   "Invalid regular expression '%s': does not compile", regex);
        // NOTREACHED
      }
    }

    if(!any_matched) // never matched
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
               "Some regular expresion(s) did not match any Cactus variable.");
    return; // NOTREACHED
  }
}
