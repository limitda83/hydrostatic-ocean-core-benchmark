!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Program: cfd_exp (fortran_cpu backend)                               !
!  Description: Entry point for the Fortran CPU backend. Reads the      !
!               namelist generated from the config TOML tree, integrates the   !
!               2D linear rotating shallow water equations, and writes  !
!               a raw state dump plus metrics for the R2 gate check     !
!               against the NumPy reference.                            !
!  Pipeline: toml2nml.py -> cfd_exp -> compare_backends.py              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program cfd_exp
   use mod_kinds,   only: wp, i8
   use mod_config,  only: run_config_t, read_config
   use mod_grid,    only: grid_t, build_grid, set_omp_min_points
   use mod_cases,   only: case_t, case_init, case_exact
   use mod_schemes, only: stepper_t, stepper_init, stepper_step
   use mod_diag,    only: l2_rel, linf_rel, total_mass, total_energy, rms, &
                          write_state_bin, write_metrics_json
   implicit none

   type(run_config_t) :: cfg
   type(grid_t)       :: grid
   type(case_t)       :: cs
   type(stepper_t)    :: stepper

   real(wp), allocatable :: u0(:, :), v0(:, :), e0(:, :)
   real(wp), allocatable :: u(:, :), v(:, :), eta(:, :)
   real(wp), allocatable :: ue(:, :), ve(:, :), ee(:, :)
   real(wp), allocatable :: samples(:)

   character(len=512) :: nml_path, host
   real(wp) :: t_med, t_mad, t_min, t_max
   real(wp) :: mass0, energy0, mass1, energy1, mass_scale, domain_area
   integer  :: rep, step, nthreads, ios
   integer(i8) :: c0, c1, crate

   integer :: n_args
!$ integer :: omp_get_max_threads
!$ external :: omp_get_max_threads

   n_args = command_argument_count()
   if (n_args < 1) then
      write(*, '(a)') 'usage: cfd_exp <namelist>'
      error stop 1
   end if
   call get_command_argument(1, nml_path)
   call read_config(trim(nml_path), cfg)

   call get_environment_variable('HOSTNAME', host, status=ios)
   if (ios /= 0 .or. len_trim(host) == 0) host = 'unknown'

   nthreads = 1
!$ nthreads = omp_get_max_threads()

   call set_omp_min_points(cfg%omp_min_points)
   call build_grid(cfg%nx, cfg%ny, cfg%lx, cfg%ly, grid)
   allocate(u0(cfg%nx, cfg%ny), v0(cfg%nx, cfg%ny), e0(cfg%nx, cfg%ny))
   allocate(u(cfg%nx, cfg%ny),  v(cfg%nx, cfg%ny),  eta(cfg%nx, cfg%ny))
   allocate(ue(cfg%nx, cfg%ny), ve(cfg%nx, cfg%ny), ee(cfg%nx, cfg%ny))
   allocate(samples(cfg%n_repeat))

   call case_init(cfg, grid, cs, u0, v0, e0)
   call stepper_init(stepper, cfg, grid, cfg%dt)

   mass0 = total_mass(e0, grid)
   energy0 = total_energy(u0, v0, e0, grid, cfg%h0, cfg%g)

   write(*, '(a,a,a,i0,a,a,a,f6.3,a,es12.5,a,i0,a,i0)') &
      'fortran_cpu: case=', trim(cfg%case_name), ' nx=', cfg%nx, &
      ' scheme=', trim(cfg%scheme_name), ' theta=', cfg%theta, &
      ' dt=', cfg%dt, ' steps=', cfg%n_steps, ' threads=', nthreads

   ! ---- timing protocol: discard warm-ups, then n_repeat measured runs (R7)
   do rep = 1, cfg%n_warmup
      u = u0;  v = v0;  eta = e0
      do step = 1, cfg%n_steps
         call stepper_step(stepper, u, v, eta)
      end do
   end do

   stepper%solver%total_iterations = 0
   stepper%solver%failures = 0
   do rep = 1, cfg%n_repeat
      u = u0;  v = v0;  eta = e0
      call system_clock(c0, crate)
      do step = 1, cfg%n_steps
         call stepper_step(stepper, u, v, eta)
      end do
      call system_clock(c1)
      samples(rep) = real(c1 - c0, wp) / real(crate, wp)
   end do
   ! Solver iteration count of a single measured pass.
   stepper%solver%total_iterations = stepper%solver%total_iterations / cfg%n_repeat

   call sort_ascending(samples)
   t_med = median(samples)
   t_min = samples(1)
   t_max = samples(size(samples))
   t_mad = median_abs_dev(samples, t_med)

   call case_exact(cfg, grid, cs, cfg%t_final, ue, ve, ee)
   mass1 = total_mass(eta, grid)
   energy1 = total_energy(u, v, eta, grid, cfg%h0, cfg%g)
   domain_area = grid%cell_area * real(grid%nx, wp) * real(grid%ny, wp)
   mass_scale = rms(ee) * domain_area
   if (mass_scale == 0.0_wp) mass_scale = 1.0_wp

   write(*, '(a,es12.5,a,es12.5,a,es12.5)') &
      '  L2(eta)=', l2_rel(eta, ee), '  mass drift=', (mass1 - mass0) / mass_scale, &
      '  energy drift=', (energy1 - energy0) / abs(energy0)
   write(*, '(a,es12.5,a,es12.5,a,i0)') &
      '  wall median=', t_med, ' s  mad=', t_mad, ' s  pcg_iters/run=', &
      stepper%solver%total_iterations

   call write_state_bin(trim(cfg%out_prefix)//'_state.bin', u, v, eta)
   call write_metrics_json(trim(cfg%out_prefix)//'_metrics.json', &
        grid%nx, grid%ny, grid%dx, cfg%dt, cfg%n_steps, cfg%scheme_name, &
        cfg%theta, cfg%n_picard, cfg%solver_kind, &
        stepper%solver%total_iterations, stepper%solver%failures, &
        l2_rel(eta, ee), l2_rel(u, ue), l2_rel(v, ve), linf_rel(eta, ee), &
        mass1, energy1, (mass1 - mass0) / mass_scale, &
        (energy1 - energy0) / abs(energy0), t_med, t_mad, t_min, t_max, &
        cfg%n_repeat, cfg%n_warmup, nthreads, trim(host))

contains

   subroutine sort_ascending(a)
      real(wp), intent(inout) :: a(:)
      integer  :: i, j
      real(wp) :: key
      do i = 2, size(a)          ! insertion sort; n_repeat is small
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
      real(wp), intent(in) :: a(:)     ! must already be sorted
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

end program cfd_exp
