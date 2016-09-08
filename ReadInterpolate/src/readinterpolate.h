#ifndef _READINTERPOLATE_H_
#define _READINTERPOLATE_H_

#ifdef __cplusplus 
extern "C"
{
#endif

void ReadInterpolate_CheckAllPointsSet(const cGH * cctkGH);
void ReadInterpolate_Interpolate(const cGH * cctkGH, int iteration,
                                 int timelevel, int component, int reflevel,
                                 CCTK_REAL time,
                                 int varindex, const CCTK_INT lsh[3], const CCTK_REAL origin[3],
                                 const CCTK_REAL delta[3], 
                                 CCTK_REAL const * const vardata, void *token);

void ReadInterpolate_PullData(void * token);

#ifdef __cplusplus 
}
#endif

#endif
