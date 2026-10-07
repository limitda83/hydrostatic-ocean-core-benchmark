!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_cases3d                                                  !
!  Description: 3D verification cases of docs/03_discretization_spec.md !
!               S7.6, mirroring libs/core/cases3d.py:                   !
!                 barotropic3d   - exact reduction to the 2D case       !
!                 vdiffusion     - discrete vertical eigenmode decay    !
!                 baroclinic_igw - exact internal gravity wave mode     !
!  Pipeline: mod_grid -> mod_cases3d -> cfd_exp3d                       !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_cases3d
   use mod_kinds,  only: wp
   use mod_grid,   only: grid_t
   use mod_config, only: run_config_t
   implicit none
   private
   public :: case3d_t, case3d_init, case3d_exact

   real(wp), parameter :: pi = 3.14159265358979323846_wp

   type :: case3d_t
      character(len=32) :: name = 'barotropic3d'
      real(wp) :: k = 0.0_wp, l = 0.0_wp, kappa2 = 0.0_wp
      real(wp) :: omega = 0.0_wp, period = 0.0_wp
      real(wp) :: m = 0.0_wp, c2 = 0.0_wp, rate = 0.0_wp
   end type case3d_t

contains

   pure function zt_centre(cfg, k) result(z)
      ! Height above the bottom at layer centre k (spec S7.3).
      type(run_config_t), intent(in) :: cfg
      integer,            intent(in) :: k
      real(wp) :: z, dz
      dz = cfg%h0 / real(cfg%nz, wp)
      z = (real(cfg%nz - k, wp) + 0.5_wp) * dz
   end function zt_centre

   subroutine case3d_init(cfg, g, cs)
      type(run_config_t), intent(in)  :: cfg
      type(grid_t),       intent(in)  :: g
      type(case3d_t),     intent(out) :: cs
      real(wp) :: dz, lam

      cs%name = cfg%case_name
      dz = cfg%h0 / real(cfg%nz, wp)

      select case (trim(cfg%case_name))
      case ('barotropic3d')
         cs%k = 2.0_wp * pi * real(cfg%mode_x, wp) / g%lx
         cs%l = 2.0_wp * pi * real(cfg%mode_y, wp) / g%ly
         cs%kappa2 = cs%k**2 + cs%l**2
         cs%omega = sqrt(cfg%f0**2 + cfg%g * cfg%h0 * cs%kappa2)
         cs%period = 2.0_wp * pi / cs%omega

      case ('vdiffusion')
         if (cfg%nu <= 0.0_wp) then
            write(*, '(a)') 'FATAL: vdiffusion requires nu > 0'
            error stop 1
         end if
         cs%m = pi * real(cfg%mode_z, wp) / cfg%h0
         lam = (2.0_wp * cos(pi * real(cfg%mode_z, wp) / real(cfg%nz, wp)) - 2.0_wp) / dz**2
         cs%rate = cfg%nu * lam
         cs%period = -1.0_wp / cs%rate

      case ('baroclinic_igw')
         if (cfg%n2 <= 0.0_wp) then
            write(*, '(a)') 'FATAL: baroclinic_igw requires n2 > 0'
            error stop 1
         end if
         cs%k = 2.0_wp * pi * real(cfg%mode_x, wp) / g%lx
         cs%l = 2.0_wp * pi * real(cfg%mode_y, wp) / g%ly
         cs%kappa2 = cs%k**2 + cs%l**2
         cs%m = pi * real(cfg%mode_z, wp) / cfg%h0
         cs%c2 = cfg%n2 / cs%m**2
         cs%omega = sqrt(cfg%f0**2 + cs%c2 * cs%kappa2)
         cs%period = 2.0_wp * pi / cs%omega

      case default
         write(*, '(a)') 'FATAL: unknown 3D case '//trim(cfg%case_name)// &
            ' (implemented: barotropic3d|vdiffusion|baroclinic_igw)'
         error stop 1
      end select
   end subroutine case3d_init

   subroutine case3d_exact(cfg, g, cs, t, u, v, b, eta)
      type(run_config_t), intent(in)  :: cfg
      type(grid_t),       intent(in)  :: g
      type(case3d_t),     intent(in)  :: cs
      real(wp),           intent(in)  :: t
      real(wp),           intent(out) :: u(:,:,:), v(:,:,:), b(:,:,:), eta(:,:)
      real(wp) :: amp, xe, ye, xu, yu, xv, yv, pe, pu, pv, cz, sz, decay
      integer  :: i, j, k

      u = 0.0_wp;  v = 0.0_wp;  b = 0.0_wp;  eta = 0.0_wp

      select case (trim(cs%name))
      case ('barotropic3d')
         amp = cfg%eta0 / (cfg%h0 * cs%kappa2)
         do k = 1, cfg%nz
            do j = 1, g%ny
               do i = 1, g%nx
                  xe = (real(i - 1, wp) + 0.5_wp) * g%dx
                  ye = (real(j - 1, wp) + 0.5_wp) * g%dy
                  xu = (real(i - 1, wp) + 1.0_wp) * g%dx
                  xv = xe
                  yv = (real(j - 1, wp) + 1.0_wp) * g%dy
                  pu = cs%k * xu + cs%l * ye - cs%omega * t
                  pv = cs%k * xv + cs%l * yv - cs%omega * t
                  u(i, j, k) = amp * (cs%omega * cs%k * cos(pu) - cfg%f0 * cs%l * sin(pu))
                  v(i, j, k) = amp * (cs%omega * cs%l * cos(pv) + cfg%f0 * cs%k * sin(pv))
               end do
            end do
         end do
         do j = 1, g%ny
            do i = 1, g%nx
               xe = (real(i - 1, wp) + 0.5_wp) * g%dx
               ye = (real(j - 1, wp) + 0.5_wp) * g%dy
               pe = cs%k * xe + cs%l * ye - cs%omega * t
               eta(i, j) = cfg%eta0 * cos(pe)
            end do
         end do

      case ('vdiffusion')
         decay = exp(cs%rate * t)
         do k = 1, cfg%nz
            cz = cos(cs%m * zt_centre(cfg, k))
            do j = 1, g%ny
               do i = 1, g%nx
                  u(i, j, k) = cfg%u0 * cz * decay
               end do
            end do
         end do

      case ('baroclinic_igw')
         amp = cfg%g * cfg%eta0 / (cs%c2 * cs%kappa2)
         do k = 1, cfg%nz
            cz = cos(cs%m * zt_centre(cfg, k))
            sz = sin(cs%m * zt_centre(cfg, k))
            do j = 1, g%ny
               do i = 1, g%nx
                  xe = (real(i - 1, wp) + 0.5_wp) * g%dx
                  ye = (real(j - 1, wp) + 0.5_wp) * g%dy
                  xu = (real(i - 1, wp) + 1.0_wp) * g%dx
                  xv = xe
                  yv = (real(j - 1, wp) + 1.0_wp) * g%dy
                  pu = cs%k * xu + cs%l * ye - cs%omega * t
                  pv = cs%k * xv + cs%l * yv - cs%omega * t
                  pe = cs%k * xe + cs%l * ye - cs%omega * t
                  u(i, j, k) = amp * (cs%omega * cs%k * cos(pu) &
                                      - cfg%f0 * cs%l * sin(pu)) * cz
                  v(i, j, k) = amp * (cs%omega * cs%l * cos(pv) &
                                      + cfg%f0 * cs%k * sin(pv)) * cz
                  b(i, j, k) = -cfg%g * cfg%eta0 * cs%m * cos(pe) * sz
               end do
            end do
         end do
      end select
   end subroutine case3d_exact

end module mod_cases3d
