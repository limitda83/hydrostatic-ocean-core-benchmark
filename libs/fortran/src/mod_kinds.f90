!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_kinds                                                    !
!  Description: Precision selectors. dp is the fp64 gate precision      !
!               (RULES.md R9); wp is the working precision that the    !
!               mixed-precision axis (G) will vary in Phase 7.          !
!  Pipeline: base module for every other Fortran module                 !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_kinds
   use, intrinsic :: iso_fortran_env, only: real32, real64, int64
   implicit none
   public
   integer, parameter :: sp = real32
   integer, parameter :: dp = real64
   integer, parameter :: i8 = int64
   ! Working precision (axis G). Compile with -DWP_SP (gfortran -cpp /
   ! nvfortran -Mpreprocess) for the fp32 variant; the numerics never change,
   ! only this constant. Files on disk stay fp64 in BOTH builds so that an
   ! fp32 run can be compared against the fp64 reference (R9).
#ifdef WP_SP
   integer, parameter :: wp = sp
#else
   integer, parameter :: wp = dp
#endif
end module mod_kinds
