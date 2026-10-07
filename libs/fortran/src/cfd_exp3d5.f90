!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: cfd_exp3d5 (program)                                         !
!  Description: Driver for the spec v0.5 full-core backend. It is a     !
!               pure stepper-plus-timer: the domain and the initial     !
!               state come from binaries that tools/toml2nml.py --v05   !
!               writes from the NumPy reference, the final state goes   !
!               back out as a binary, and accuracy is judged in Python  !
!               against the reference (tools/compare_backends3d5.py).   !
!               That keeps every case definition in one language, so    !
!               the R2 gate compares backends on a byte-identical        !
!               problem (docs/25 S1).                                   !
!  Pipeline: toml2nml --v05 -> cfd_exp3d5 -> *_state3d5.bin + JSON      !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program cfd_exp3d5
   use mod_kinds,       only: wp, dp
   use mod_config,      only: run_config_t, read_config
   use mod_grid,        only: grid_t, build_grid, set_omp_min_points
   use mod_vertical,    only: set_tridiag_kernel
   use mod_diag,        only: jnum
   use mod_domain5,     only: domain5_t, domain5_build, domain5_read
   use mod_model3d_v05, only: v05_config_t, read_v05_config, stepper5_t, stepper5_init, stepper5_step
   !$ use omp_lib
   implicit none

   type(run_config_t) :: cfg
   type(v05_config_t) :: v5
   type(grid_t)       :: grid
   type(domain5_t)    :: dom
   type(stepper5_t)   :: st
   character(len=512) :: nml_path
   character(len=256) :: host
   real(wp), allocatable :: h(:,:), mask(:,:)
   real(wp), allocatable :: u0(:,:,:), v0(:,:,:), b0(:,:,:), t0(:,:,:), s0(:,:,:), e0(:,:)
   real(wp), allocatable :: u(:,:,:), v(:,:,:), b(:,:,:), t(:,:,:), s(:,:,:), eta(:,:)
   real(wp), allocatable :: samples(:)
   integer  :: nx, ny, nz, rep, step, unit, ios, nthreads, n_done
   logical  :: diverged
   integer(kind=8) :: c0, c1, crate
   real(wp) :: t_med, t_mad
   logical  :: is_ts
   real(dp), allocatable :: buf2(:,:), buf3(:,:,:)

   if (command_argument_count() < 1) then
      write(*, '(a)') 'usage: cfd_exp3d5 <namelist>'
      stop 1
   end if
   call get_command_argument(1, nml_path)
   call read_config(trim(nml_path), cfg)
   call read_v05_config(trim(nml_path), v5)
   call get_environment_variable('HOSTNAME', host, status=ios)
   if (ios /= 0) host = 'unknown'
   nthreads = 1
   !$ nthreads = omp_get_max_threads()

   call set_omp_min_points(cfg%omp_min_points)
   call set_tridiag_kernel(cfg%tridiag_kernel)
   call build_grid(cfg%nx, cfg%ny, cfg%lx, cfg%ly, grid)
   nx = cfg%nx;  ny = cfg%ny;  nz = cfg%nz
   is_ts = trim(v5%tracers) == 'TS'

   allocate(h(nx,ny), mask(nx,ny))
   call domain5_read(trim(v5%domain_file), nx, ny, h, mask)
   call domain5_build(dom, nx, ny, nz, h, mask, trim(v5%face_rule), v5%min_partial, &
                      trim(v5%bc_x), trim(v5%bc_y))

   allocate(u0(nx,ny,nz), v0(nx,ny,nz), b0(nx,ny,nz), t0(nx,ny,nz), s0(nx,ny,nz), e0(nx,ny))
   allocate(u(nx,ny,nz),  v(nx,ny,nz),  b(nx,ny,nz),  t(nx,ny,nz),  s(nx,ny,nz),  eta(nx,ny))
   allocate(samples(cfg%n_repeat))
   call read_init(trim(v5%init_file))

   call stepper5_init(st, cfg, v5, grid, dom, cfg%dt)

   write(*, '(a,a,a,i0,a,i0,a,a,a,a,a,a,a,es12.5,a,i0,a,i0)') &
      'fortran3d5: case=', trim(cfg%case_name), ' nx=', nx, ' nz=', nz, &
      ' scheme=', trim(cfg%scheme_name), ' solver=', trim(cfg%solver_kind), &
      ' eos=', trim(v5%eos), ' dt=', cfg%dt, ' steps=', cfg%n_steps, ' threads=', nthreads

   ! Divergence guard (N18): a blown-up run must not grind through max_iter
   ! Helmholtz sweeps for hours; every 10 steps |eta| is checked and the run
   ! is stopped and flagged.
   diverged = .false.
   do rep = 1, cfg%n_warmup
      if (diverged) exit
      call reset_state()
      do step = 1, cfg%n_steps
         call stepper5_step(st, u, v, b, eta, t, s)
         if (mod(step, 10) == 0) then
            if (.not. (maxval(abs(eta)) < 1.0e6_wp)) then
               diverged = .true.;  exit
            end if
         end if
      end do
   end do

   st%solver%total_iterations = 0
   st%barotropic_substeps = 0
   st%tridiagonal_solves = 0
   n_done = 0
   do rep = 1, cfg%n_repeat
      call reset_state()
      call system_clock(c0, crate)
      do step = 1, cfg%n_steps
         call stepper5_step(st, u, v, b, eta, t, s)
         if (mod(step, 10) == 0) then
            if (.not. (maxval(abs(eta)) < 1.0e6_wp)) then
               diverged = .true.;  exit
            end if
         end if
      end do
      call system_clock(c1)
      samples(rep) = real(c1 - c0, wp) / real(crate, wp)
      n_done = rep
      if (diverged) then
         write(*, '(a)') '  DIVERGED (|eta| > 1e6 or NaN) - run stopped early'
         exit
      end if
   end do
   ! Only n_done repeats actually ran. Padding the array with a copy of the last
   ! sample made the MAD come out ZERO, which reads as "five repeats, very
   ! stable" when the truth is one repeat; and dividing the counters by
   ! n_repeat instead of n_done under-reported iterations per run by the same
   ! factor, which could hide a diverged run from the collector (docs/90 N33).
   st%solver%total_iterations = st%solver%total_iterations / max(1, n_done)
   st%barotropic_substeps = st%barotropic_substeps / max(1, n_done)

   call sort_ascending(samples(1:n_done))
   t_med = median(samples(1:n_done))
   t_mad = median_abs_dev(samples(1:n_done), t_med)

   write(*, '(a,es12.5,a,es12.5,a,i0,a,i0)') &
      '  wall median=', t_med, ' s  mad=', t_mad, ' s  solver_iters/run=', &
      st%solver%total_iterations, '  substeps/run=', st%barotropic_substeps

   open(newunit=unit, file=trim(cfg%out_prefix)//'_state3d5.bin', access='stream', &
        form='unformatted', status='replace', action='write')
   ! Always fp64 on disk so tools/state_error3d5.py can compare any pair (R9).
   write(unit) real(eta, dp)
   write(unit) real(u, dp)
   write(unit) real(v, dp)
   write(unit) real(b, dp)
   if (is_ts) then
      write(unit) real(t, dp)
      write(unit) real(s, dp)
   end if
   close(unit)

   open(newunit=unit, file=trim(cfg%out_prefix)//'_metrics.json', status='replace', action='write')
   write(unit, '(a)')           '{'
   write(unit, '(a)')           '  "backend": "fortran3d5",'
   write(unit, '(a,a,a)')       '  "host": "', trim(host), '",'
   write(unit, '(a,i0,a)')      '  "nx": ', nx, ','
   write(unit, '(a,i0,a)')      '  "ny": ', ny, ','
   write(unit, '(a,i0,a)')      '  "nz": ', nz, ','
   write(unit, '(a,i0,a)')      '  "cells": ', nx * ny * nz, ','
   write(unit, '(a,i0,a)')      '  "n_steps": ', cfg%n_steps, ','
   write(unit, '(a,a,a)')       '  "scheme": "', trim(cfg%scheme_name), '",'
   write(unit, '(a,a,a)')       '  "solver": "', trim(cfg%solver_kind), '",'
   write(unit, '(a,a,a)')       '  "eos": "', trim(v5%eos), '",'
   write(unit, '(a,a,a)')       '  "advection": "', trim(v5%advection), '",'
   write(unit, '(a,a,a)')       '  "tracers": "', trim(v5%tracers), '",'
   write(unit, '(a,a,a)')       '  "theta": ', trim(adjustl(jnum(cfg%theta))), ','
   write(unit, '(a,a,a)')       '  "dt": ', trim(adjustl(jnum(cfg%dt))), ','
   write(unit, '(a,i0,a)')      '  "solver_iterations": ', st%solver%total_iterations, ','
   write(unit, '(a,i0,a)')      '  "solver_failures": ', st%solver%failures, ','
   write(unit, '(a,i0,a)')      '  "tridiagonal_solves": ', st%tridiagonal_solves, ','
   write(unit, '(a,i0,a)')      '  "n_split": ', st%n_split, ','
   write(unit, '(a,i0,a)')      '  "barotropic_substeps": ', st%barotropic_substeps, ','
   if (diverged) then
      write(unit, '(a)')        '  "diverged": true,'
   else
      write(unit, '(a)')        '  "diverged": false,'
   end if
   write(unit, '(a,a,a)')       '  "wall_s": ', trim(adjustl(jnum(t_med))), ','
   write(unit, '(a,a,a)')       '  "wall_mad_s": ', trim(adjustl(jnum(t_mad))), ','
   write(unit, '(a,i0,a)')      '  "n_repeat": ', cfg%n_repeat, ','
   write(unit, '(a,i0)')        '  "omp_num_threads": ', nthreads
   write(unit, '(a)')           '}'
   close(unit)

contains

   subroutine read_init(path)
      character(len=*), intent(in) :: path
      integer :: u_
      open(newunit=u_, file=path, form='unformatted', access='stream', &
           status='old', action='read', iostat=ios)
      if (ios /= 0) then
         write(*, '(a)') 'FATAL: cannot open init file '//trim(path)
         error stop 1
      end if
      ! The bundle is fp64 regardless of the working precision (R9).
      allocate(buf2(nx, ny), buf3(nx, ny, nz))
      read(u_) buf2;  e0 = real(buf2, wp)
      read(u_) buf3;  u0 = real(buf3, wp)
      read(u_) buf3;  v0 = real(buf3, wp)
      read(u_) buf3;  b0 = real(buf3, wp)
      if (is_ts) then
         read(u_) buf3;  t0 = real(buf3, wp)
         read(u_) buf3;  s0 = real(buf3, wp)
      else
         t0 = 0.0_wp;  s0 = 0.0_wp
      end if
      deallocate(buf2, buf3)
      close(u_)
   end subroutine read_init

   subroutine reset_state()
      u = u0;  v = v0;  b = b0;  eta = e0;  t = t0;  s = s0
   end subroutine reset_state

   subroutine sort_ascending(a)
      real(wp), intent(inout) :: a(:)
      integer  :: i, j
      real(wp) :: key
      do i = 2, size(a)
         key = a(i);  j = i - 1
         do while (j >= 1)
            if (a(j) <= key) exit
            a(j + 1) = a(j);  j = j - 1
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

end program cfd_exp3d5
