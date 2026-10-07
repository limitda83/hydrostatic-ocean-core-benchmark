!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: eos_bench                                                    !
!  Description: Cost of the equation of state at three levels of        !
!               nonlinearity (spec S10.6). Every other kernel in a      !
!               hydrostatic core is memory bound; the EOS is the one    !
!               that is not. polyTEOS10 reads three fields and writes   !
!               one for ~70 flop, an arithmetic intensity of ~2.2       !
!               flop/byte against an fp64 ridge of 1.25 on the RTX      !
!               5090 - so raising the EOS level should IMPROVE the      !
!               GPU-to-CPU ratio. RQ7 is that prediction.               !
!  Pipeline: eos_bench -> docs/25                                       !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_eos
   use mod_kinds, only: wp
   implicit none
   public

   ! Simplified EOS, Roquet et al. (2015) eq. (25).
   real(wp), parameter :: SA0 = 1.6550e-1_wp, SB0 = 7.6554e-1_wp,   &
                          SL1 = 5.9520e-2_wp, SL2 = 7.4914e-4_wp,   &
                          SM1 = 1.4970e-4_wp, SM2 = 1.1090e-5_wp,   &
                          SNU = 2.4341e-3_wp

   ! polyTEOS10-bsq reduced variables and the 55 coefficients (Table 4).
   real(wp), parameter :: RDELTA_S = 32.0_wp, R1_S0 = 0.875_wp / 35.16504_wp, &
                          R1_T0 = 1.0_wp / 40.0_wp, R1_Z0 = 1.0e-4_wp
   real(wp), parameter :: &
      R00 =  4.6494977072e+01_wp, R01 = -5.2099962525e+00_wp, &
      R02 =  2.2601900708e-01_wp, R03 =  6.4326772569e-02_wp, &
      R04 =  1.5616995503e-02_wp, R05 = -1.7243708991e-03_wp
   real(wp), parameter :: &
      E000 = 8.0189615746e+02_wp, E100 = 8.6672408165e+02_wp, &
      E200 =-1.7864682637e+03_wp, E300 = 2.0375295546e+03_wp, &
      E400 =-1.2849161071e+03_wp, E500 = 4.3227585684e+02_wp, &
      E600 =-6.0579916612e+01_wp, E010 = 2.6010145068e+01_wp, &
      E110 =-6.5281885265e+01_wp, E210 = 8.1770425108e+01_wp, &
      E310 =-5.6888046321e+01_wp, E410 = 1.7681814114e+01_wp, &
      E510 =-1.9193502195e+00_wp, E020 =-3.7074170417e+01_wp, &
      E120 = 6.1548258127e+01_wp, E220 =-6.0362551501e+01_wp, &
      E320 = 2.9130021253e+01_wp, E420 =-5.4723692739e+00_wp, &
      E030 = 2.1661789529e+01_wp, E130 =-3.3449108469e+01_wp, &
      E230 = 1.9717078466e+01_wp, E330 =-3.1742946532e+00_wp, &
      E040 =-8.3627885467e+00_wp, E140 = 1.1311538584e+01_wp, &
      E240 =-5.3563304045e+00_wp, E050 = 5.4048723791e-01_wp, &
      E150 = 4.5111434961e-01_wp, E060 =-1.9098268277e-01_wp, &
      E001 = 1.9681925209e+01_wp, E101 =-4.2549998214e+01_wp, &
      E201 = 5.0774768218e+01_wp, E301 =-3.0938076334e+01_wp, &
      E401 = 6.6051753097e+00_wp, E011 =-1.3336301113e+01_wp, &
      E111 =-4.4870114575e+00_wp, E211 = 5.0042598061e+00_wp, &
      E311 =-6.5399043664e-01_wp, E021 = 6.7080479603e+00_wp, &
      E121 = 3.5063081279e+00_wp, E221 =-1.8795372996e+00_wp, &
      E031 =-2.4649669534e+00_wp, E131 =-5.5077101279e-01_wp, &
      E041 = 5.5927935970e-01_wp, E002 = 2.0660924175e+00_wp, &
      E102 =-4.9527603989e+00_wp, E202 = 2.5019633803e+00_wp, &
      E012 = 2.0564311499e+00_wp, E112 =-2.1311365518e-01_wp, &
      E022 =-1.2419983026e+00_wp, E003 =-2.3342758797e-02_wp, &
      E103 =-1.8507636718e-02_wp, E013 = 3.7969820455e-01_wp

contains

   subroutine eos_linear(n, t, s, z, rho, alpha, beta, rho0)
      integer,  intent(in)  :: n
      real(wp), intent(in)  :: t(n), s(n), z(n), alpha, beta, rho0
      real(wp), intent(out) :: rho(n)
      integer :: i
      !$omp parallel do schedule(static)
      !$acc parallel loop gang vector
      do i = 1, n
         rho(i) = rho0 * (-alpha * (t(i) - 10.0_wp) + beta * (s(i) - 35.0_wp))
      end do
      !$acc end parallel loop
      !$omp end parallel do
   end subroutine eos_linear

   subroutine eos_seos(n, t, s, z, rho)
      integer,  intent(in)  :: n
      real(wp), intent(in)  :: t(n), s(n), z(n)
      real(wp), intent(out) :: rho(n)
      integer  :: i
      real(wp) :: ta, sa
      !$omp parallel do schedule(static) private(ta, sa)
      !$acc parallel loop gang vector private(ta, sa)
      do i = 1, n
         ta = t(i) - 10.0_wp
         sa = s(i) - 35.0_wp
         rho(i) = -SA0 * (1.0_wp + 0.5_wp * SL1 * ta + SM1 * z(i)) * ta   &
                +  SB0 * (1.0_wp - 0.5_wp * SL2 * sa - SM2 * z(i)) * sa   &
                -  SNU * ta * sa
      end do
      !$acc end parallel loop
      !$omp end parallel do
   end subroutine eos_seos

   subroutine eos_teos10(n, t, s, z, rho)
      integer,  intent(in)  :: n
      real(wp), intent(in)  :: t(n), s(n), z(n)
      real(wp), intent(out) :: rho(n)
      integer  :: i
      real(wp) :: zh, ss, tt, r0, rz0, rz1, rz2, rz3
      !$omp parallel do schedule(static) private(zh, ss, tt, r0, rz0, rz1, rz2, rz3)
      !$acc parallel loop gang vector private(zh, ss, tt, r0, rz0, rz1, rz2, rz3)
      do i = 1, n
         zh = z(i) * R1_Z0
         ss = sqrt((s(i) + RDELTA_S) * R1_S0)
         tt = t(i) * R1_T0
         r0 = (((((R05 * zh + R04) * zh + R03) * zh + R02) * zh + R01) * zh + R00) * zh
         rz3 = E013 * tt + E103 * ss + E003
         rz2 = (E022 * tt + E112 * ss + E012) * tt + (E202 * ss + E102) * ss + E002
         rz1 = (((E041 * tt + E131 * ss + E031) * tt                          &
                + (E221 * ss + E121) * ss + E021) * tt                        &
                + ((E311 * ss + E211) * ss + E111) * ss + E011) * tt          &
             + (((E401 * ss + E301) * ss + E201) * ss + E101) * ss + E001
         rz0 = ((((( E060 * tt + E150 * ss + E050 ) * tt                     &
                    + (E240 * ss + E140) * ss + E040 ) * tt                    &
                   + ((E330 * ss + E230) * ss + E130) * ss + E030 ) * tt       &
                  + (((E420 * ss + E320) * ss + E220) * ss + E120) * ss + E020 &
                 ) * tt                                                        &
                 + ((((E510 * ss + E410) * ss + E310) * ss + E210) * ss + E110)&
                   * ss + E010 ) * tt                                          &
               + (((((E600 * ss + E500) * ss + E400) * ss + E300) * ss + E200) &
                   * ss + E100) * ss + E000
         rho(i) = ((rz3 * zh + rz2) * zh + rz1) * zh + rz0 + r0
      end do
      !$acc end parallel loop
      !$omp end parallel do
   end subroutine eos_teos10

end module mod_eos


program eos_bench
   use mod_kinds, only: wp
   use mod_eos
   implicit none
   integer  :: n = 3000000, n_repeat = 20, n_warmup = 3, i, rep, argc
   character(len=32) :: arg, kind = 'teos10'
   real(wp), allocatable :: t(:), s(:), z(:), rho(:)
   real(wp) :: t0, t1, wall, chk, wall_mad
   real(wp), allocatable :: samples(:)
   integer(kind=8) :: c, r

   argc = command_argument_count()
   if (argc >= 1) then
      call get_command_argument(1, arg); read(arg, *) n
   end if
   if (argc >= 2) call get_command_argument(2, kind)
   if (argc >= 3) then
      call get_command_argument(3, arg); read(arg, *) n_repeat
   end if

   allocate(t(n), s(n), z(n), rho(n))
   do i = 1, n
      t(i) = 5.0_wp + 15.0_wp * real(modulo(i, 977), wp) / 977.0_wp
      s(i) = 33.0_wp + 3.0_wp * real(modulo(i, 733), wp) / 733.0_wp
      z(i) = 5000.0_wp * real(modulo(i, 521), wp) / 521.0_wp
   end do

   !$acc data copyin(t, s, z) create(rho)
   do rep = 1, n_warmup
      call dispatch()
   end do
   allocate(samples(n_repeat))
   do rep = 1, n_repeat
      call system_clock(count=c, count_rate=r); t0 = real(c, wp) / real(r, wp)
      call dispatch()
      call system_clock(count=c, count_rate=r); t1 = real(c, wp) / real(r, wp)
      samples(rep) = t1 - t0
   end do
   ! R7-2: the median of the repeats, not the best of them.
   call sort_inplace(samples)
   if (modulo(n_repeat, 2) == 1) then
      wall = samples((n_repeat + 1) / 2)
   else
      wall = 0.5_wp * (samples(n_repeat / 2) + samples(n_repeat / 2 + 1))
   end if
   wall_mad = mad_of(samples, wall)
   !$acc update self(rho)
   !$acc end data

   chk = 0.0_wp
   do i = 1, n, max(1, n / 1000)
      chk = chk + rho(i)
   end do
   write(*, '(a,a10,a,i10,a,es13.6,a,es13.6,a,es13.6)') 'eos=', trim(kind), &
      ' n=', n, ' wall=', wall, ' mad=', wall_mad, ' checksum=', chk

contains
   subroutine sort_inplace(a)
      real(wp), intent(inout) :: a(:)
      real(wp) :: tmp
      integer  :: i, j
      do i = 2, size(a)
         tmp = a(i);  j = i - 1
         do while (j >= 1)
            if (a(j) <= tmp) exit
            a(j + 1) = a(j);  j = j - 1
         end do
         a(j + 1) = tmp
      end do
   end subroutine sort_inplace
   function mad_of(a, m) result(d)
      real(wp), intent(in) :: a(:), m
      real(wp) :: d, b(size(a))
      b = abs(a - m)
      call sort_inplace(b)
      if (modulo(size(b), 2) == 1) then
         d = b((size(b) + 1) / 2)
      else
         d = 0.5_wp * (b(size(b) / 2) + b(size(b) / 2 + 1))
      end if
   end function mad_of
   subroutine dispatch()
      select case (trim(kind))
      case ('linear'); call eos_linear(n, t, s, z, rho, 2.0e-4_wp, 7.4e-4_wp, 1025.0_wp)
      case ('seos');   call eos_seos(n, t, s, z, rho)
      case ('teos10'); call eos_teos10(n, t, s, z, rho)
      case default
         write(*, '(a)') 'FATAL: unknown eos '//trim(kind); stop 1
      end select
   end subroutine dispatch
end program eos_bench
