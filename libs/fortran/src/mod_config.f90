!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_config                                                   !
!  Description: Read the run configuration from a Fortran namelist that !
!               tools/toml2nml.py generates from the config TOML tree, so !
!               every backend is driven by one TOML tree (RULES.md R3).!
!  Pipeline: toml2nml.py -> namelist -> mod_config -> all modules       !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_config
   use mod_kinds, only: wp
   implicit none
   private
   public :: run_config_t, read_config

   type :: run_config_t
      ! grid
      integer  :: nx = 100, ny = 100
      real(wp) :: lx = 1.0e7_wp, ly = 1.0e7_wp
      ! physics
      real(wp) :: g = 9.80616_wp, f0 = 1.0e-4_wp, h0 = 1000.0_wp
      ! scheme
      character(len=32) :: scheme_name = 'theta'
      real(wp) :: theta = 0.5_wp, theta_cor = 0.5_wp
      integer  :: n_picard = 2
      character(len=32) :: solver_kind = 'pcg_jacobi'
      real(wp) :: rtol = 1.0e-12_wp
      integer  :: max_iter = 2000
      ! case
      character(len=32) :: case_name = 'igw'
      integer  :: mode_x = 2, mode_y = 2
      integer  :: n_modes = 8              ! igw_broadband: modes per direction
      real(wp) :: slope = 1.0_wp           ! igw_broadband: amplitude spectrum
      real(wp) :: eta0 = 1.0_wp
      ! time: dt and n_steps come from the Python driver so both backends
      ! integrate exactly the same discrete problem (R2 comparison).
      real(wp) :: dt = 0.0_wp, t_final = 0.0_wp
      integer  :: n_steps = 0
      ! timing protocol (R7)
      integer  :: n_repeat = 5, n_warmup = 1
      integer  :: omp_min_points = 65536   ! serial below this size
      ! --- 3D (spec v0.2) ---
      integer  :: nz = 1                   ! 1 selects the 2D barotropic model
      real(wp) :: theta_v = 0.5_wp         ! implicitness of vertical diffusion
      real(wp) :: n2 = 0.0_wp              ! buoyancy frequency squared
      real(wp) :: nu = 0.0_wp              ! vertical viscosity
      real(wp) :: kappa = 0.0_wp           ! vertical diffusivity
      real(wp) :: rho0 = 1025.0_wp
      real(wp) :: tau_x = 0.0_wp, tau_y = 0.0_wp
      real(wp) :: bottom_drag = 0.0_wp
      character(len=8) :: tridiag_kernel = 'plane'   ! plane | column
      ! Accepted so the shared namelist parses, but unused here: the
      ! Fortran/OpenACC PCG always computes its scalars on the host,
      ! which is precisely the behaviour the CUDA backend compares against.
      character(len=8) :: pcg_sync = 'host'
      integer  :: pcg_check_every = 1
      integer  :: n_split = 0              ! barotropic substeps; 0 = pick from CFL
      integer  :: mode_z = 1               ! vertical mode for the 3D cases
      real(wp) :: u0 = 1.0_wp              ! amplitude for vdiffusion
      character(len=256) :: out_prefix = 'output/fortran_run'
   end type run_config_t

contains

   subroutine read_config(path, cfg)
      character(len=*), intent(in)  :: path
      type(run_config_t), intent(out) :: cfg
      integer :: unit, ios
      character(len=256) :: msg

      integer  :: nx, ny, n_picard, max_iter, mode_x, mode_y, n_steps
      integer  :: n_modes, nz, mode_z, n_split
      integer  :: n_repeat, n_warmup, omp_min_points
      real(wp) :: lx, ly, g, f0, h0, theta, theta_cor, rtol, eta0, slope
      real(wp) :: theta_v, n2, nu, kappa, rho0, tau_x, tau_y, bottom_drag, u0
      real(wp) :: dt, t_final
      character(len=32)  :: scheme_name, solver_kind, case_name
      character(len=8)   :: tridiag_kernel, pcg_sync
      integer            :: pcg_check_every
      character(len=256) :: out_prefix

      namelist /grid_nml/    nx, ny, nz, lx, ly
      namelist /physics_nml/ g, f0, h0, n2, nu, kappa, rho0, &
                             tau_x, tau_y, bottom_drag
      namelist /scheme_nml/  scheme_name, theta, theta_cor, theta_v, n_picard, n_split, &
                             solver_kind, rtol, max_iter
      namelist /case_nml/    case_name, mode_x, mode_y, mode_z, n_modes, &
                             slope, eta0, u0
      namelist /run_nml/     dt, t_final, n_steps, n_repeat, n_warmup, &
                             omp_min_points, tridiag_kernel, &
                             pcg_sync, pcg_check_every, out_prefix

      ! Seed the locals from the type defaults so a partial namelist is valid.
      nx = cfg%nx;  ny = cfg%ny;  nz = cfg%nz;  lx = cfg%lx;  ly = cfg%ly
      g = cfg%g;  f0 = cfg%f0;  h0 = cfg%h0
      n2 = cfg%n2;  nu = cfg%nu;  kappa = cfg%kappa;  rho0 = cfg%rho0
      tau_x = cfg%tau_x;  tau_y = cfg%tau_y;  bottom_drag = cfg%bottom_drag
      theta_v = cfg%theta_v;  mode_z = cfg%mode_z;  u0 = cfg%u0;  n_split = cfg%n_split
      scheme_name = cfg%scheme_name;  theta = cfg%theta
      theta_cor = cfg%theta_cor;  n_picard = cfg%n_picard
      solver_kind = cfg%solver_kind;  rtol = cfg%rtol;  max_iter = cfg%max_iter
      case_name = cfg%case_name;  mode_x = cfg%mode_x;  mode_y = cfg%mode_y
      n_modes = cfg%n_modes;  slope = cfg%slope;  eta0 = cfg%eta0
      dt = cfg%dt;  t_final = cfg%t_final;  n_steps = cfg%n_steps
      n_repeat = cfg%n_repeat;  n_warmup = cfg%n_warmup
      omp_min_points = cfg%omp_min_points
      tridiag_kernel = cfg%tridiag_kernel
      pcg_sync = cfg%pcg_sync;  pcg_check_every = cfg%pcg_check_every
      out_prefix = cfg%out_prefix

      open(newunit=unit, file=trim(path), status='old', action='read', &
           iostat=ios, iomsg=msg)
      if (ios /= 0) then
         write(*, '(a)') 'FATAL: cannot open namelist '//trim(path)//': '//trim(msg)
         error stop 1
      end if
      read(unit, nml=grid_nml,    iostat=ios); call check(ios, 'grid_nml')
      rewind(unit); read(unit, nml=physics_nml, iostat=ios); call check(ios, 'physics_nml')
      rewind(unit); read(unit, nml=scheme_nml,  iostat=ios); call check(ios, 'scheme_nml')
      rewind(unit); read(unit, nml=case_nml,    iostat=ios); call check(ios, 'case_nml')
      rewind(unit); read(unit, nml=run_nml,     iostat=ios); call check(ios, 'run_nml')
      close(unit)

      cfg%nx = nx;  cfg%ny = ny;  cfg%nz = nz;  cfg%lx = lx;  cfg%ly = ly
      cfg%g = g;  cfg%f0 = f0;  cfg%h0 = h0
      cfg%n2 = n2;  cfg%nu = nu;  cfg%kappa = kappa;  cfg%rho0 = rho0
      cfg%tau_x = tau_x;  cfg%tau_y = tau_y;  cfg%bottom_drag = bottom_drag
      cfg%theta_v = theta_v;  cfg%mode_z = mode_z;  cfg%u0 = u0;  cfg%n_split = n_split
      cfg%scheme_name = scheme_name;  cfg%theta = theta
      cfg%theta_cor = theta_cor;  cfg%n_picard = n_picard
      cfg%solver_kind = solver_kind;  cfg%rtol = rtol;  cfg%max_iter = max_iter
      cfg%case_name = case_name;  cfg%mode_x = mode_x;  cfg%mode_y = mode_y
      cfg%n_modes = n_modes;  cfg%slope = slope;  cfg%eta0 = eta0
      cfg%dt = dt;  cfg%t_final = t_final;  cfg%n_steps = n_steps
      cfg%n_repeat = n_repeat;  cfg%n_warmup = n_warmup
      cfg%omp_min_points = omp_min_points
      cfg%tridiag_kernel = tridiag_kernel
      cfg%pcg_sync = pcg_sync;  cfg%pcg_check_every = pcg_check_every
      cfg%out_prefix = out_prefix

      call validate(cfg)
   end subroutine read_config

   subroutine check(ios, name)
      integer, intent(in) :: ios
      character(len=*), intent(in) :: name
      if (ios /= 0) then
         write(*, '(a,i0)') 'FATAL: failed to read namelist group '//name//', iostat=', ios
         error stop 1
      end if
   end subroutine check

   subroutine validate(cfg)
      type(run_config_t), intent(in) :: cfg
      ! Boundary validation mirrors libs/core/schemes.py::SchemeParams.validate.
      if (trim(cfg%scheme_name) /= 'fb' .and. trim(cfg%scheme_name) /= 'theta' &
          .and. trim(cfg%scheme_name) /= 'split_explicit') then
         write(*, '(a)') 'FATAL: unknown scheme '//trim(cfg%scheme_name)// &
                         ' (expected fb|theta|split_explicit)'
         error stop 1
      end if
      if (trim(cfg%scheme_name) == 'theta') then
         if (cfg%theta <= 0.0_wp .or. cfg%theta > 1.0_wp) then
            write(*, '(a,es12.5)') 'FATAL: theta must lie in (0,1]; got ', cfg%theta
            error stop 1
         end if
      end if
      if (cfg%n_picard < 1) then
         write(*, '(a)') 'FATAL: n_picard must be >= 1'
         error stop 1
      end if
      if (cfg%n_steps < 1 .or. cfg%dt <= 0.0_wp) then
         write(*, '(a)') 'FATAL: dt and n_steps must be supplied by the driver'
         error stop 1
      end if
   end subroutine validate

end module mod_config
