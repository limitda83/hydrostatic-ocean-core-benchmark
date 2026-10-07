!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_cases                                                    !
!  Description: Verification cases of docs/03_discretization_spec.md S5 !
!               - igw (exact continuum solution) and geo_balance (exact !
!               steady state of the DISCRETE operators, built from the  !
!               Fourier symbols). Mirrors libs/core/cases.py.           !
!  Pipeline: mod_grid -> mod_cases -> program (init + exact comparison) !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_cases
   use mod_kinds,  only: wp
   use mod_grid,   only: grid_t, omp_min_points
   use mod_config, only: run_config_t
   implicit none
   private
   public :: case_t, case_init, case_exact

   real(wp), parameter :: pi = 3.14159265358979323846_wp

   type :: case_t
      character(len=32) :: name = 'igw'
      real(wp) :: k = 0.0_wp, l = 0.0_wp, kappa2 = 0.0_wp
      real(wp) :: omega = 0.0_wp, period = 0.0_wp, eta0 = 1.0_wp
      complex(wp) :: u_hat = (0.0_wp, 0.0_wp), v_hat = (0.0_wp, 0.0_wp)
      real(wp) :: tx = 0.0_wp, ty = 0.0_wp
      ! igw_broadband: one entry per mode, built identically to
      ! libs/core/cases.py::InertiaGravityWaveBroadband.
      integer :: n_mode = 0
      real(wp), allocatable :: mk(:), ml(:), mkappa2(:), momega(:), mamp(:), mphase(:)
   end type case_t

contains

   subroutine case_init(cfg, g, cs, u, v, eta)
      type(run_config_t), intent(in)  :: cfg
      type(grid_t),       intent(in)  :: g
      type(case_t),       intent(out) :: cs
      real(wp),           intent(out) :: u(:, :), v(:, :), eta(:, :)
      complex(wp) :: ex, ey, phase
      real(wp)    :: fac
      integer     :: i, j

      cs%name = cfg%case_name
      cs%eta0 = cfg%eta0

      select case (trim(cfg%case_name))
      case ('igw')
         cs%k = 2.0_wp * pi * real(cfg%mode_x, wp) / g%lx
         cs%l = 2.0_wp * pi * real(cfg%mode_y, wp) / g%ly
         cs%kappa2 = cs%k**2 + cs%l**2
         cs%omega = sqrt(cfg%f0**2 + cfg%g * cfg%h0 * cs%kappa2)
         cs%period = 2.0_wp * pi / cs%omega
         call case_exact(cfg, g, cs, 0.0_wp, u, v, eta)

      case ('igw_broadband')
         call build_broadband(cfg, g, cs)
         call case_exact(cfg, g, cs, 0.0_wp, u, v, eta)

      case ('geo_balance')
         if (cfg%f0 == 0.0_wp) then
            write(*, '(a)') 'FATAL: geo_balance requires a non-zero Coriolis parameter'
            error stop 1
         end if
         cs%period = 2.0_wp * pi / abs(cfg%f0)
         cs%tx = 2.0_wp * pi * real(cfg%mode_x, wp) / real(g%nx, wp)
         cs%ty = 2.0_wp * pi * real(cfg%mode_y, wp) / real(g%ny, wp)
         ex = exp(cmplx(0.0_wp, cs%tx, wp))
         ey = exp(cmplx(0.0_wp, cs%ty, wp))
         if (abs(1.0_wp + ex) < 1.0e-12_wp .or. abs(1.0_wp + ey) < 1.0e-12_wp) then
            write(*, '(a)') 'FATAL: geo_balance mode hits the grid Nyquist limit'
            error stop 1
         end if
         fac = 4.0_wp * cfg%g / cfg%f0
         cs%v_hat =  fac * (ex - 1.0_wp) / (g%dx * (1.0_wp + conjg(ey)) * (1.0_wp + ex)) * cfg%eta0
         cs%u_hat = -fac * (ey - 1.0_wp) / (g%dy * (1.0_wp + conjg(ex)) * (1.0_wp + ey)) * cfg%eta0
         do j = 1, g%ny
            do i = 1, g%nx
               phase = exp(cmplx(0.0_wp, cs%tx * real(i - 1, wp) &
                                       + cs%ty * real(j - 1, wp), wp))
               eta(i, j) = cfg%eta0 * real(phase, wp)
               u(i, j) = real(cs%u_hat * phase, wp)
               v(i, j) = real(cs%v_hat * phase, wp)
            end do
         end do

      case default
         write(*, '(a)') 'FATAL: unknown case '//trim(cfg%case_name)// &
                         ' (implemented: igw|igw_broadband|geo_balance)'
         error stop 1
      end select
   end subroutine case_init

   subroutine build_broadband(cfg, g, cs)
      ! Superposition of n_modes^2 exact inertia-gravity modes. Phases come
      ! from integer arithmetic so every backend builds bit-identical fields.
      type(run_config_t), intent(in)    :: cfg
      type(grid_t),       intent(in)    :: g
      type(case_t),       intent(inout) :: cs
      integer  :: m, n, idx, nm
      real(wp) :: raw, raw_sq_sum, norm

      nm = cfg%n_modes
      cs%n_mode = nm * nm
      allocate(cs%mk(cs%n_mode), cs%ml(cs%n_mode), cs%mkappa2(cs%n_mode), &
               cs%momega(cs%n_mode), cs%mamp(cs%n_mode), cs%mphase(cs%n_mode))

      raw_sq_sum = 0.0_wp
      idx = 0
      do m = 1, nm
         do n = 1, nm
            idx = idx + 1
            cs%mk(idx) = 2.0_wp * pi * real(m, wp) / g%lx
            cs%ml(idx) = 2.0_wp * pi * real(n, wp) / g%ly
            cs%mkappa2(idx) = cs%mk(idx)**2 + cs%ml(idx)**2
            cs%momega(idx) = sqrt(cfg%f0**2 + cfg%g * cfg%h0 * cs%mkappa2(idx))
            raw = real(m * m + n * n, wp) ** (-0.5_wp * cfg%slope)
            cs%mamp(idx) = raw
            raw_sq_sum = raw_sq_sum + raw * raw
            cs%mphase(idx) = 2.0_wp * pi * real(mod(m * 37 + n * 17, 101), wp) / 101.0_wp
         end do
      end do
      norm = cfg%eta0 / sqrt(raw_sq_sum / 2.0_wp)
      cs%mamp = cs%mamp * norm
      cs%eta0 = cfg%eta0
      cs%period = 2.0_wp * pi / minval(cs%momega)      ! slowest mode
   end subroutine build_broadband

   subroutine case_exact(cfg, g, cs, t, u, v, eta)
      type(run_config_t), intent(in)  :: cfg
      type(grid_t),       intent(in)  :: g
      type(case_t),       intent(in)  :: cs
      real(wp),           intent(in)  :: t
      real(wp),           intent(out) :: u(:, :), v(:, :), eta(:, :)
      real(wp)    :: amp, xe, ye, xu, yu, xv, yv, pe, pu, pv
      complex(wp) :: phase
      integer     :: i, j, idx

      select case (trim(cs%name))
      case ('igw')
         amp = cs%eta0 / (cfg%h0 * cs%kappa2)
         !$omp parallel do collapse(2) private(i, j, xe, ye, xu, yu, xv, yv, pe, pu, pv) if(g%nx*g%ny >= omp_min_points)
         do j = 1, g%ny
            do i = 1, g%nx
               ! Physical coordinates of each variable's own grid point (S2).
               xe = (real(i - 1, wp) + 0.5_wp) * g%dx
               ye = (real(j - 1, wp) + 0.5_wp) * g%dy
               xu = (real(i - 1, wp) + 1.0_wp) * g%dx
               yu = ye
               xv = xe
               yv = (real(j - 1, wp) + 1.0_wp) * g%dy
               pe = cs%k * xe + cs%l * ye - cs%omega * t
               pu = cs%k * xu + cs%l * yu - cs%omega * t
               pv = cs%k * xv + cs%l * yv - cs%omega * t
               eta(i, j) = cs%eta0 * cos(pe)
               u(i, j) = amp * (cs%omega * cs%k * cos(pu) - cfg%f0 * cs%l * sin(pu))
               v(i, j) = amp * (cs%omega * cs%l * cos(pv) + cfg%f0 * cs%k * sin(pv))
            end do
         end do
         !$omp end parallel do

      case ('igw_broadband')
         !$omp parallel do collapse(2) private(i, j, xe, ye, xu, yu, xv, yv) if(g%nx*g%ny >= omp_min_points)
         do j = 1, g%ny
            do i = 1, g%nx
               eta(i, j) = 0.0_wp
               u(i, j) = 0.0_wp
               v(i, j) = 0.0_wp
            end do
         end do
         !$omp end parallel do
         do idx = 1, cs%n_mode
            amp = cs%mamp(idx) / (cfg%h0 * cs%mkappa2(idx))
            !$omp parallel do collapse(2) private(i, j, xe, ye, xu, yu, xv, yv, pe, pu, pv) if(g%nx*g%ny >= omp_min_points)
            do j = 1, g%ny
               do i = 1, g%nx
                  xe = (real(i - 1, wp) + 0.5_wp) * g%dx
                  ye = (real(j - 1, wp) + 0.5_wp) * g%dy
                  xu = (real(i - 1, wp) + 1.0_wp) * g%dx
                  yu = ye
                  xv = xe
                  yv = (real(j - 1, wp) + 1.0_wp) * g%dy
                  pe = cs%mk(idx) * xe + cs%ml(idx) * ye - cs%momega(idx) * t + cs%mphase(idx)
                  pu = cs%mk(idx) * xu + cs%ml(idx) * yu - cs%momega(idx) * t + cs%mphase(idx)
                  pv = cs%mk(idx) * xv + cs%ml(idx) * yv - cs%momega(idx) * t + cs%mphase(idx)
                  eta(i, j) = eta(i, j) + cs%mamp(idx) * cos(pe)
                  u(i, j) = u(i, j) + amp * (cs%momega(idx) * cs%mk(idx) * cos(pu) &
                                             - cfg%f0 * cs%ml(idx) * sin(pu))
                  v(i, j) = v(i, j) + amp * (cs%momega(idx) * cs%ml(idx) * cos(pv) &
                                             + cfg%f0 * cs%mk(idx) * sin(pv))
               end do
            end do
            !$omp end parallel do
         end do

      case ('geo_balance')
         ! Exact discrete steady state: the solution is the initial state.
         do j = 1, g%ny
            do i = 1, g%nx
               phase = exp(cmplx(0.0_wp, cs%tx * real(i - 1, wp) &
                                       + cs%ty * real(j - 1, wp), wp))
               eta(i, j) = cs%eta0 * real(phase, wp)
               u(i, j) = real(cs%u_hat * phase, wp)
               v(i, j) = real(cs%v_hat * phase, wp)
            end do
         end do
      end select
   end subroutine case_exact

end module mod_cases
