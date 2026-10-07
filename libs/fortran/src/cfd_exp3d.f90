!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Program: cfd_exp3d (3D backend, spec v0.2)                           !
!  Description: Entry point for the compiled 3D hydrostatic model.      !
!               Reads the namelist that tools/toml2nml.py generates and !
!               writes a raw state dump plus metrics for the R2 gate    !
!               against the NumPy reference.                            !
!  Pipeline: toml2nml.py -> cfd_exp3d -> compare_backends3d.py          !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program cfd_exp3d
   use mod_kinds,   only: wp, i8
   use mod_config,  only: run_config_t, read_config
   use mod_grid,    only: grid_t, build_grid, set_omp_min_points
   use mod_cases3d, only: case3d_t, case3d_init, case3d_exact
   use mod_model3d, only: stepper3d_t, stepper3d_init, stepper3d_step
   use mod_diag,    only: jnum
   use mod_vertical, only: set_tridiag_kernel
   implicit none

   type(run_config_t) :: cfg
   type(grid_t)       :: grid
   type(case3d_t)     :: cs
   type(stepper3d_t)  :: st

   real(wp), allocatable :: u0(:,:,:), v0(:,:,:), b0(:,:,:), e0(:,:)
   real(wp), allocatable :: u(:,:,:), v(:,:,:), b(:,:,:), eta(:,:)
   real(wp), allocatable :: ue(:,:,:), ve(:,:,:), be(:,:,:), ee(:,:)
   real(wp), allocatable :: samples(:)

   character(len=512) :: nml_path, host
   real(wp) :: t_med, t_mad, l2u, l2b, l2e
   integer  :: rep, step, nthreads, ios, unit, nx, ny, nz
   integer(i8) :: c0, c1, crate
!$ integer :: omp_get_max_threads
!$ external :: omp_get_max_threads

   if (command_argument_count() < 1) then
      write(*, '(a)') 'usage: cfd_exp3d <namelist>'
      error stop 1
   end if
   call get_command_argument(1, nml_path)
   call read_config(trim(nml_path), cfg)
   if (cfg%nz < 1) then
      write(*, '(a)') 'FATAL: cfd_exp3d needs nz >= 1 in grid_nml'
      error stop 1
   end if

   call get_environment_variable('HOSTNAME', host, status=ios)
   if (ios /= 0 .or. len_trim(host) == 0) host = 'unknown'
   nthreads = 1
!$ nthreads = omp_get_max_threads()

   call set_omp_min_points(cfg%omp_min_points)
   call set_tridiag_kernel(cfg%tridiag_kernel)
   call build_grid(cfg%nx, cfg%ny, cfg%lx, cfg%ly, grid)
   nx = cfg%nx;  ny = cfg%ny;  nz = cfg%nz

   allocate(u0(nx,ny,nz), v0(nx,ny,nz), b0(nx,ny,nz), e0(nx,ny))
   allocate(u(nx,ny,nz),  v(nx,ny,nz),  b(nx,ny,nz),  eta(nx,ny))
   allocate(ue(nx,ny,nz), ve(nx,ny,nz), be(nx,ny,nz), ee(nx,ny))
   allocate(samples(cfg%n_repeat))

   call case3d_init(cfg, grid, cs)
   call case3d_exact(cfg, grid, cs, 0.0_wp, u0, v0, b0, e0)
   call stepper3d_init(st, cfg, grid, cfg%dt)

   write(*, '(a,a,a,i0,a,i0,a,a,a,f6.3,a,es12.5,a,i0,a,i0)') &
      'fortran3d: case=', trim(cfg%case_name), ' nx=', nx, ' nz=', nz, &
      ' scheme=', trim(cfg%scheme_name), ' theta=', cfg%theta, &
      ' dt=', cfg%dt, ' steps=', cfg%n_steps, ' threads=', nthreads

   do rep = 1, cfg%n_warmup
      u = u0;  v = v0;  b = b0;  eta = e0
      do step = 1, cfg%n_steps
         call stepper3d_step(st, u, v, b, eta)
      end do
   end do

   st%solver%total_iterations = 0
   st%barotropic_substeps = 0
   do rep = 1, cfg%n_repeat
      u = u0;  v = v0;  b = b0;  eta = e0
      call system_clock(c0, crate)
      do step = 1, cfg%n_steps
         call stepper3d_step(st, u, v, b, eta)
      end do
      call system_clock(c1)
      samples(rep) = real(c1 - c0, wp) / real(crate, wp)
   end do
   st%solver%total_iterations = st%solver%total_iterations / max(1, cfg%n_repeat)
   st%barotropic_substeps = st%barotropic_substeps / max(1, cfg%n_repeat)

   call sort_ascending(samples)
   t_med = median(samples)
   t_mad = median_abs_dev(samples, t_med)

   call case3d_exact(cfg, grid, cs, cfg%t_final, ue, ve, be, ee)
   l2u = rel_l2_3d(u, ue)
   l2b = rel_l2_3d(b, be)
   l2e = rel_l2_2d(eta, ee)

   write(*, '(a,es12.5,a,es12.5,a,es12.5)') &
      '  L2(u)=', l2u, '  L2(b)=', l2b, '  L2(eta)=', l2e
   write(*, '(a,es12.5,a,es12.5,a,i0)') &
      '  wall median=', t_med, ' s  mad=', t_mad, ' s  pcg_iters/run=', &
      st%solver%total_iterations

   open(newunit=unit, file=trim(cfg%out_prefix)//'_state3d.bin', access='stream', &
        form='unformatted', status='replace', action='write')
   write(unit) eta
   write(unit) u
   write(unit) v
   write(unit) b
   close(unit)

   open(newunit=unit, file=trim(cfg%out_prefix)//'_metrics.json', status='replace', &
        action='write')
   write(unit, '(a)')           '{'
   write(unit, '(a)')           '  "backend": "fortran3d",'
   write(unit, '(a,a,a)')       '  "host": "', trim(host), '",'
   write(unit, '(a,i0,a)')      '  "nx": ', nx, ','
   write(unit, '(a,i0,a)')      '  "ny": ', ny, ','
   write(unit, '(a,i0,a)')      '  "nz": ', nz, ','
   write(unit, '(a,i0,a)')      '  "cells": ', nx * ny * nz, ','
   write(unit, '(a,i0,a)')      '  "n_steps": ', cfg%n_steps, ','
   write(unit, '(a,a,a)')       '  "scheme": "', trim(cfg%scheme_name), '",'
   write(unit, '(a,a,a)')       '  "theta": ', trim(adjustl(jnum(cfg%theta))), ','
   write(unit, '(a,a,a)')       '  "theta_v": ', trim(adjustl(jnum(cfg%theta_v))), ','
   write(unit, '(a,a,a)')       '  "dt": ', trim(adjustl(jnum(cfg%dt))), ','
   write(unit, '(a,a,a)')       '  "h_eff": ', trim(adjustl(jnum(st%h_eff))), ','
   write(unit, '(a,i0,a)')      '  "solver_iterations": ', st%solver%total_iterations, ','
   write(unit, '(a,i0,a)')      '  "tridiagonal_solves": ', st%tridiagonal_solves, ','
   write(unit, '(a,i0,a)')      '  "n_split": ', st%n_split, ','
   write(unit, '(a,i0,a)')      '  "barotropic_substeps": ', st%barotropic_substeps, ','
   write(unit, '(a,a,a)')       '  "tridiag_kernel": "', trim(cfg%tridiag_kernel), '",'
   write(unit, '(a,a,a)')       '  "l2_rel_u": ', trim(adjustl(jnum(l2u))), ','
   write(unit, '(a,a,a)')       '  "l2_rel_b": ', trim(adjustl(jnum(l2b))), ','
   write(unit, '(a,a,a)')       '  "l2_rel_eta": ', trim(adjustl(jnum(l2e))), ','
   write(unit, '(a,a,a)')       '  "wall_s": ', trim(adjustl(jnum(t_med))), ','
   write(unit, '(a,a,a)')       '  "wall_mad_s": ', trim(adjustl(jnum(t_mad))), ','
   write(unit, '(a,i0,a)')      '  "n_repeat": ', cfg%n_repeat, ','
   write(unit, '(a,i0)')        '  "omp_num_threads": ', nthreads
   write(unit, '(a)')           '}'
   close(unit)

contains

   function rel_l2_3d(a, r) result(e)
      real(wp), intent(in) :: a(:,:,:), r(:,:,:)
      real(wp) :: e, num, den
      num = sqrt(sum((a - r)**2))
      den = sqrt(sum(r**2))
      e = merge(num / den, num, den > 0.0_wp)
   end function rel_l2_3d

   function rel_l2_2d(a, r) result(e)
      real(wp), intent(in) :: a(:,:), r(:,:)
      real(wp) :: e, num, den
      num = sqrt(sum((a - r)**2))
      den = sqrt(sum(r**2))
      e = merge(num / den, num, den > 0.0_wp)
   end function rel_l2_2d

   subroutine sort_ascending(a)
      real(wp), intent(inout) :: a(:)
      integer  :: i, j
      real(wp) :: key
      do i = 2, size(a)
         key = a(i)
         j = i - 1
         do while (j >= 1)
            if (a(j) <= key) exit
            a(j + 1) = a(j)
            j = j - 1
         end do
         a(j + 1) = key
      end do
   end subroutine sort_ascending

   pure function median(a) result(m)
      real(wp), intent(in) :: a(:)
      real(wp) :: m
      integer  :: n
      n = size(a)
      if (mod(n, 2) == 1) then
         m = a((n + 1) / 2)
      else
         m = 0.5_wp * (a(n / 2) + a(n / 2 + 1))
      end if
   end function median

   function median_abs_dev(a, med) result(mad)
      real(wp), intent(in) :: a(:), med
      real(wp) :: mad
      real(wp), allocatable :: d(:)
      allocate(d(size(a)))
      d = abs(a - med)
      call sort_ascending(d)
      mad = median(d)
   end function median_abs_dev

end program cfd_exp3d
