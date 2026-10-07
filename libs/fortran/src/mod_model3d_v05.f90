!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_model3d_v05                                              !
!  Description: The spec v0.5 full core (docs/03 S10) for the compiled  !
!               backend: real bathymetry (z-level partial cells), land  !
!               and face masks, closed or periodic walls, flux-form     !
!               advection, prognostic T/S with linear / S-EOS / TEOS-10 !
!               equations of state, horizontal viscosity and diffusion, !
!               surface stress and fluxes, the common-depth pressure    !
!               gradient, and the variable-coefficient free-surface     !
!               Helmholtz solve - under forward-backward, theta         !
!               (Casulli) and split-explicit time integration.          !
!                                                                       !
!  It is a line-for-line port of libs/core/model3d_v05.py, which is     !
!  verified (V5-1..V5-6). Every array is (i,j,k) column-major; the      !
!  vertical interface array wf(:,:,1) is the surface and wf(:,:,nz+1)   !
!  the bottom. OpenMP and OpenACC directives sit on the same loops so   !
!  the three builds share one arithmetic (R5).                          !
!  Pipeline: mod_domain5 + mod_helm_var -> mod_model3d_v05 -> cfd_exp3d5!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_model3d_v05
   use mod_kinds,    only: wp
   use mod_config,   only: run_config_t
   use mod_grid,     only: grid_t, omp_min_points
   use mod_vertical, only: thomas_batched
   use mod_domain5,  only: domain5_t
   use mod_helm_var, only: helm_var_t, helm_var_init, helm_var_solve, helm_var_update
   implicit none
   private
   public :: v05_config_t, read_v05_config, stepper5_t, stepper5_init, stepper5_step

   ! polyTEOS10-bsq (Roquet et al. 2015) reduced variables and coefficients,
   ! verified in eos_bench.f90 against the published check value 1027.45140.
   real(wp), parameter :: RDELTA_S = 32.0_wp, R1_S0 = 0.875_wp / 35.16504_wp, &
                          R1_T0 = 1.0_wp / 40.0_wp, R1_Z0 = 1.0e-4_wp
   real(wp), parameter :: &
      R00 =  4.6494977072e+01_wp, R01 = -5.2099962525e+00_wp, R02 =  2.2601900708e-01_wp, &
      R03 =  6.4326772569e-02_wp, R04 =  1.5616995503e-02_wp, R05 = -1.7243708991e-03_wp
   real(wp), parameter :: &
      E000 = 8.0189615746e+02_wp, E100 = 8.6672408165e+02_wp, E200 =-1.7864682637e+03_wp, &
      E300 = 2.0375295546e+03_wp, E400 =-1.2849161071e+03_wp, E500 = 4.3227585684e+02_wp, &
      E600 =-6.0579916612e+01_wp, E010 = 2.6010145068e+01_wp, E110 =-6.5281885265e+01_wp, &
      E210 = 8.1770425108e+01_wp, E310 =-5.6888046321e+01_wp, E410 = 1.7681814114e+01_wp, &
      E510 =-1.9193502195e+00_wp, E020 =-3.7074170417e+01_wp, E120 = 6.1548258127e+01_wp, &
      E220 =-6.0362551501e+01_wp, E320 = 2.9130021253e+01_wp, E420 =-5.4723692739e+00_wp, &
      E030 = 2.1661789529e+01_wp, E130 =-3.3449108469e+01_wp, E230 = 1.9717078466e+01_wp, &
      E330 =-3.1742946532e+00_wp, E040 =-8.3627885467e+00_wp, E140 = 1.1311538584e+01_wp, &
      E240 =-5.3563304045e+00_wp, E050 = 5.4048723791e-01_wp, E150 = 4.5111434961e-01_wp, &
      E060 =-1.9098268277e-01_wp, E001 = 1.9681925209e+01_wp, E101 =-4.2549998214e+01_wp, &
      E201 = 5.0774768218e+01_wp, E301 =-3.0938076334e+01_wp, E401 = 6.6051753097e+00_wp, &
      E011 =-1.3336301113e+01_wp, E111 =-4.4870114575e+00_wp, E211 = 5.0042598061e+00_wp, &
      E311 =-6.5399043664e-01_wp, E021 = 6.7080479603e+00_wp, E121 = 3.5063081279e+00_wp, &
      E221 =-1.8795372996e+00_wp, E031 =-2.4649669534e+00_wp, E131 =-5.5077101279e-01_wp, &
      E041 = 5.5927935970e-01_wp, E002 = 2.0660924175e+00_wp, E102 =-4.9527603989e+00_wp, &
      E202 = 2.5019633803e+00_wp, E012 = 2.0564311499e+00_wp, E112 =-2.1311365518e-01_wp, &
      E022 =-1.2419983026e+00_wp, E003 =-2.3342758797e-02_wp, E103 =-1.8507636718e-02_wp, &
      E013 = 3.7969820455e-01_wp

   ! The spec v0.5 additions, read from the &v05_nml group that
   ! tools/toml2nml.py --v05 writes.
   type :: v05_config_t
      character(len=16) :: vcoord = 'zlevel'
      character(len=16) :: bc_x = 'periodic', bc_y = 'periodic'
      character(len=16) :: face_rule = 'min'
      real(wp) :: min_partial = 0.1_wp
      character(len=16) :: eos = 'linear'
      character(len=16) :: pgf = 'remove'
      logical  :: pgf_correction = .true.
      character(len=16) :: advection = 'none'
      character(len=16) :: tracers = 'buoyancy'
      real(wp) :: alpha_t = 2.0e-4_wp, beta_s = 7.4e-4_wp, t0 = 10.0_wp, s0 = 35.0_wp
      real(wp) :: cp = 3990.0_wp, a_h = 0.0_wp, k_h = 0.0_wp
      real(wp) :: q_heat = 0.0_wp, q_salt = 0.0_wp
      ! spec v0.6 closure (S11.2)
      character(len=16) :: closure = 'none'
      real(wp) :: c_k = 0.1_wp, c_eps = 0.7_wp, pr_t = 1.0_wp, z_0 = 0.1_wp
      real(wp) :: e_min = 1.0e-6_wp, l_min = 0.01_wp, e_bb = 3.75_wp
      real(wp) :: n2_min = 1.0e-8_wp
      character(len=16) :: mxl = 'integral'      ! S11.2 axis: integral | recursive
      character(len=256) :: domain_file = '', init_file = ''
   end type v05_config_t

   type :: stepper5_t
      type(run_config_t) :: cfg
      type(v05_config_t) :: v5
      type(grid_t)       :: g
      type(domain5_t)    :: d
      type(helm_var_t)   :: solver
      integer  :: nx = 0, ny = 0, nz = 0
      real(wp) :: dt = 0.0_wp, dz_nom = 0.0_wp
      logical  :: is_theta = .false., is_split = .false., is_ts = .false.
      logical  :: do_adv = .false., is_up3 = .false., is_tvd = .false., is_tke = .false.
      integer  :: n_split = 0, solver_rebuilds = 0
      integer  :: solver_iterations = 0, barotropic_substeps = 0, tridiagonal_solves = 0
      ! Interface arrays (nx,ny,nz+1)
      real(wp), allocatable :: nu3(:,:,:), kap3(:,:,:), wf(:,:,:), wold(:,:,:), tke(:,:,:)
      ! Tridiagonal coefficients (nx,ny,nz)
      real(wp), allocatable :: msub(:,:,:), mdiag(:,:,:), msup(:,:,:)
      real(wp), allocatable :: bsub(:,:,:), bdiag(:,:,:), bsup(:,:,:)
      real(wp), allocatable :: cstar(:,:,:), dstar(:,:,:)
      ! 3D work
      real(wp), allocatable :: q(:,:,:), phi(:,:,:), pgx(:,:,:), pgy(:,:,:)
      real(wp), allocatable :: avpu(:,:,:), avpv(:,:,:), ucor(:,:,:), vcor(:,:,:)
      real(wp), allocatable :: dzu(:,:,:), dzv(:,:,:), exu(:,:,:), exv(:,:,:)
      real(wp), allocatable :: gu(:,:,:), gv(:,:,:), ghu(:,:,:), ghv(:,:,:)
      real(wp), allocatable :: uit(:,:,:), vit(:,:,:), div3(:,:,:), wc(:,:,:)
      real(wp), allocatable :: trhs(:,:,:), tnew(:,:,:), tend(:,:,:), dzb(:,:,:)
      real(wp), allocatable :: w3a(:,:,:), w3b(:,:,:), w3c(:,:,:), w3d(:,:,:)
      ! 2D work
      real(wp), allocatable :: ku(:,:), kv(:,:), tau_u(:,:), tau_v(:,:)
      real(wp), allocatable :: uint(:,:), vint(:,:), divold(:,:), divg(:,:)
      real(wp), allocatable :: rhs2(:,:), etanew(:,:), gx2(:,:), gy2(:,:)
      real(wp), allocatable :: bu(:,:), bv(:,:), bui(:,:), bvi(:,:)
      real(wp), allocatable :: bavu(:,:), bavv(:,:), bucor(:,:), bvcor(:,:)
      real(wp), allocatable :: bfx(:,:), bfy(:,:), corx(:,:), cory(:,:)
      real(wp), allocatable :: umean(:,:), vmean(:,:), flux_t(:,:), flux_s(:,:), zero2(:,:)
   end type stepper5_t

contains

   ! =================================================================== config
   subroutine read_v05_config(path, v5)
      character(len=*),   intent(in)  :: path
      type(v05_config_t), intent(out) :: v5
      character(len=16)  :: vcoord, bc_x, bc_y, face_rule, eos, pgf, advection, tracers, closure, mxl
      real(wp) :: min_partial, alpha_t, beta_s, t0, s0, cp, a_h, k_h, q_heat, q_salt
      real(wp) :: c_k, c_eps, pr_t, z_0, e_min, l_min, e_bb, n2_min
      logical  :: pgf_correction
      character(len=256) :: domain_file, init_file
      integer :: unit, ios
      namelist /v05_nml/ vcoord, bc_x, bc_y, face_rule, min_partial, eos, pgf, &
                         pgf_correction, advection, tracers, alpha_t, beta_s, t0, s0, &
                         cp, a_h, k_h, q_heat, q_salt, closure, c_k, c_eps, pr_t, z_0, &
                         e_min, l_min, e_bb, n2_min, mxl, domain_file, init_file
      vcoord = v5%vcoord; bc_x = v5%bc_x; bc_y = v5%bc_y; face_rule = v5%face_rule
      min_partial = v5%min_partial; eos = v5%eos; pgf = v5%pgf
      pgf_correction = v5%pgf_correction; advection = v5%advection; tracers = v5%tracers
      alpha_t = v5%alpha_t; beta_s = v5%beta_s; t0 = v5%t0; s0 = v5%s0; cp = v5%cp
      a_h = v5%a_h; k_h = v5%k_h; q_heat = v5%q_heat; q_salt = v5%q_salt
      closure = v5%closure; c_k = v5%c_k; c_eps = v5%c_eps; pr_t = v5%pr_t; z_0 = v5%z_0
      e_min = v5%e_min; l_min = v5%l_min; e_bb = v5%e_bb
      n2_min = v5%n2_min; mxl = v5%mxl
      domain_file = v5%domain_file; init_file = v5%init_file
      open(newunit=unit, file=path, status='old', action='read', iostat=ios)
      if (ios /= 0) then
         write(*, '(a)') 'FATAL: cannot open namelist '//trim(path)
         error stop 1
      end if
      read(unit, nml=v05_nml, iostat=ios)
      close(unit)
      if (ios /= 0) then
         write(*, '(a)') 'FATAL: cannot read &v05_nml from '//trim(path)// &
                         ' (generate it with tools/toml2nml.py --v05)'
         error stop 1
      end if
      v5%vcoord = vcoord; v5%bc_x = bc_x; v5%bc_y = bc_y; v5%face_rule = face_rule
      v5%min_partial = min_partial; v5%eos = eos; v5%pgf = pgf
      v5%pgf_correction = pgf_correction; v5%advection = advection; v5%tracers = tracers
      v5%alpha_t = alpha_t; v5%beta_s = beta_s; v5%t0 = t0; v5%s0 = s0; v5%cp = cp
      v5%a_h = a_h; v5%k_h = k_h; v5%q_heat = q_heat; v5%q_salt = q_salt
      v5%closure = closure; v5%c_k = c_k; v5%c_eps = c_eps; v5%pr_t = pr_t; v5%z_0 = z_0
      v5%e_min = e_min; v5%l_min = l_min; v5%e_bb = e_bb
      v5%n2_min = n2_min; v5%mxl = mxl
      if (trim(mxl) /= 'integral' .and. trim(mxl) /= 'recursive') then
         write(*, '(a)') 'FATAL: mxl must be integral or recursive'
         error stop 1
      end if
      v5%domain_file = domain_file; v5%init_file = init_file
      if (trim(vcoord) /= 'zlevel') then
         write(*, '(a)') 'FATAL: the compiled v0.5 backend implements vcoord=zlevel only'
         error stop 1
      end if
      if (trim(bc_x) == 'open' .or. trim(bc_y) == 'open') then
         write(*, '(a)') 'FATAL: the compiled v0.5 backend implements periodic|closed walls only'
         error stop 1
      end if
   end subroutine read_v05_config

   ! ==================================================================== init
   subroutine stepper5_init(self, cfg, v5, g, d, dt)
      type(stepper5_t),   intent(inout) :: self
      type(run_config_t), intent(in)    :: cfg
      type(v05_config_t), intent(in)    :: v5
      type(grid_t),       intent(in)    :: g
      type(domain5_t),    intent(in)    :: d
      real(wp),           intent(in)    :: dt
      integer  :: nx, ny, nz, i, j, k, ip, jp
      real(wp) :: hmax, c, dt_baro, coef

      self%cfg = cfg;  self%v5 = v5;  self%g = g;  self%d = d;  self%dt = dt
      nx = g%nx;  ny = g%ny;  nz = cfg%nz
      self%nx = nx;  self%ny = ny;  self%nz = nz
      self%dz_nom = cfg%h0 / real(nz, wp)          ! grid.dz of the reference
      self%is_theta = trim(cfg%scheme_name) == 'theta'
      self%is_split = trim(cfg%scheme_name) == 'split_explicit'
      self%is_ts    = trim(v5%tracers) == 'TS'
      self%do_adv   = trim(v5%advection) /= 'none'
      self%is_up3   = trim(v5%advection) == 'up3' .or. trim(v5%advection) == 'up3_tvd'
      self%is_tvd   = trim(v5%advection) == 'up3_tvd'
      self%is_tke   = trim(v5%closure) == 'tke'

      allocate(self%nu3(nx,ny,nz+1), self%kap3(nx,ny,nz+1), self%wf(nx,ny,nz+1), self%wold(nx,ny,nz+1))
      allocate(self%tke(nx,ny,nz+1))
      self%tke = v5%e_min
      allocate(self%msub(nx,ny,nz), self%mdiag(nx,ny,nz), self%msup(nx,ny,nz))
      allocate(self%bsub(nx,ny,nz), self%bdiag(nx,ny,nz), self%bsup(nx,ny,nz))
      allocate(self%cstar(nx,ny,nz), self%dstar(nx,ny,nz))
      allocate(self%q(nx,ny,nz), self%phi(nx,ny,nz), self%pgx(nx,ny,nz), self%pgy(nx,ny,nz))
      allocate(self%avpu(nx,ny,nz), self%avpv(nx,ny,nz), self%ucor(nx,ny,nz), self%vcor(nx,ny,nz))
      allocate(self%dzu(nx,ny,nz), self%dzv(nx,ny,nz), self%exu(nx,ny,nz), self%exv(nx,ny,nz))
      allocate(self%gu(nx,ny,nz), self%gv(nx,ny,nz), self%ghu(nx,ny,nz), self%ghv(nx,ny,nz))
      allocate(self%uit(nx,ny,nz), self%vit(nx,ny,nz), self%div3(nx,ny,nz), self%wc(nx,ny,nz))
      allocate(self%trhs(nx,ny,nz), self%tnew(nx,ny,nz), self%tend(nx,ny,nz), self%dzb(nx,ny,nz))
      allocate(self%w3a(nx,ny,nz), self%w3b(nx,ny,nz), self%w3c(nx,ny,nz), self%w3d(nx,ny,nz))
      allocate(self%ku(nx,ny), self%kv(nx,ny), self%tau_u(nx,ny), self%tau_v(nx,ny))
      allocate(self%uint(nx,ny), self%vint(nx,ny), self%divold(nx,ny), self%divg(nx,ny))
      allocate(self%rhs2(nx,ny), self%etanew(nx,ny), self%gx2(nx,ny), self%gy2(nx,ny))
      allocate(self%bu(nx,ny), self%bv(nx,ny), self%bui(nx,ny), self%bvi(nx,ny))
      allocate(self%bavu(nx,ny), self%bavv(nx,ny), self%bucor(nx,ny), self%bvcor(nx,ny))
      allocate(self%bfx(nx,ny), self%bfy(nx,ny), self%corx(nx,ny), self%cory(nx,ny))
      allocate(self%umean(nx,ny), self%vmean(nx,ny), self%flux_t(nx,ny), self%flux_s(nx,ny), self%zero2(nx,ny))

      self%nu3 = cfg%nu;  self%kap3 = cfg%kappa
      self%zero2 = 0.0_wp
      self%flux_t = v5%q_heat / (cfg%rho0 * v5%cp)
      self%flux_s = v5%q_salt
      do j = 1, ny
         do i = 1, nx
            self%tau_u(i, j) = cfg%tau_x / cfg%rho0 * d%mask3u(i, j, 1)
            self%tau_v(i, j) = cfg%tau_y / cfg%rho0 * d%mask3v(i, j, 1)
         end do
      end do

      call build_coefficients(self)

      if (self%is_split) then
         hmax = 0.0_wp
         do j = 1, ny
            do i = 1, nx
               if (d%mask(i, j) > 0.0_wp) hmax = max(hmax, d%h(i, j))
            end do
         end do
         c = sqrt(cfg%g * hmax)
         dt_baro = 1.0_wp / (c * sqrt(1.0_wp / g%dx**2 + 1.0_wp / g%dy**2))
         if (cfg%n_split > 0) then
            self%n_split = cfg%n_split
         else
            self%n_split = max(1, ceiling(1.2_wp * dt / dt_baro))
         end if
      end if
   end subroutine stepper5_init

   ! Tridiagonal coefficients, q = A^-1 1, Ku/Kv and the Helmholtz operator
   ! for the CURRENT nu3/kap3 - once for constant coefficients, every step
   ! under the TKE closure (spec S11.5).
   subroutine build_coefficients(self)
      type(stepper5_t), intent(inout) :: self
      integer  :: i, j, k, ip, jp, nx, ny, nz
      real(wp) :: coef
      nx = self%nx;  ny = self%ny;  nz = self%nz
      call diffusion_coeffs_var(self, self%nu3, self%d%dz3, self%cfg%theta_v, self%cfg%bottom_drag, &
                                self%msub, self%mdiag, self%msup)
      call diffusion_coeffs_var(self, self%kap3, self%d%dz3, self%cfg%theta_v, 0.0_wp, &
                                self%bsub, self%bdiag, self%bsup)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               self%w3a(i, j, k) = 1.0_wp
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp parallel default(shared)
      call thomas_batched(self%g, nz, self%msub, self%mdiag, self%msup, self%w3a, self%q, &
                          self%cstar, self%dstar)
      !$omp end parallel
      self%tridiagonal_solves = self%tridiagonal_solves + 1
      !$omp parallel do collapse(2) default(shared) private(i,j,k,ip,jp)
      !$acc parallel loop collapse(2) gang vector private(k,ip,jp)
      do j = 1, ny
         do i = 1, nx
            jp = merge(1, j + 1, j == ny)
            ip = merge(1, i + 1, i == nx)
            self%ku(i, j) = 0.0_wp;  self%kv(i, j) = 0.0_wp
            do k = 1, nz
               self%ku(i, j) = self%ku(i, j) + self%d%dz3u(i, j, k) * min(self%q(i, j, k), self%q(ip, j, k))
               self%kv(i, j) = self%kv(i, j) + self%d%dz3v(i, j, k) * min(self%q(i, j, k), self%q(i, jp, k))
            end do
            self%ku(i, j) = self%ku(i, j) * self%d%masku(i, j)
            self%kv(i, j) = self%kv(i, j) * self%d%maskv(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end parallel do
      if (self%is_theta) then
         if (self%solver%nx == 0) then
            coef = self%cfg%g * self%cfg%theta**2 * self%dt**2
            call helm_var_init(self%solver, nx, ny, self%g%dx, self%g%dy, coef, self%ku, self%kv, &
                               self%d%mask, trim(self%cfg%solver_kind), self%cfg%rtol, self%cfg%max_iter)
         else
            call helm_var_update(self%solver, self%ku, self%kv)
            self%solver_rebuilds = self%solver_rebuilds + 1
         end if
      end if
   end subroutine build_coefficients

   ! ========================================================= TKE closure
   ! spec S11.2 / libs/core/closure.py, one column per thread: interface
   ! shear and N^2, Gaspar lengths, explicit production, linearised implicit
   ! dissipation, implicit vertical diffusion of e (inline Thomas), then the
   ! new K_m/K_h into nu3/kap3. Column arrays are sized by MAXNZ1.
   subroutine closure_step(self, u, v, b)
      type(stepper5_t), intent(inout) :: self
      real(wp),         intent(in)    :: u(:,:,:), v(:,:,:), b(:,:,:)
      integer, parameter :: MAXNZ1 = 65      ! nz <= 64: private column arrays stay in local memory
      integer  :: i, j, k, im, jm, nz, jj, dir, kk
      real(wp) :: nu_b, kap_b, dt, ustar2, e_surf
      real(wp) :: zs(MAXNZ1), zb(MAXNZ1), wet(MAXNZ1), dzi(MAXNZ1), n2(MAXNZ1), sh2(MAXNZ1)
      real(wp) :: lup(MAXNZ1), ldn(MAXNZ1), lk(MAXNZ1), leps(MAXNZ1), km(MAXNZ1)
      real(wp) :: sub(MAXNZ1), dia(MAXNZ1), sup(MAXNZ1), rhs(MAXNZ1), cst(MAXNZ1), xx(MAXNZ1)
      real(wp) :: hcol, uc_a, uc_b, vc_a, vc_b, inv_dzi, a_up, a_dn, klay_a, klay_b
      real(wp) :: budget, l, n2_j, dz_j, cost, aa, s_part, step, denom, prod, diss
      logical  :: recursive_mxl
      nz = self%nz;  dt = self%dt
      recursive_mxl = (trim(self%v5%mxl) == 'recursive')
      if (nz + 1 > MAXNZ1) then
         write(*, '(a)') 'FATAL: closure_step supports nz <= 64'; error stop 1
      end if
      nu_b = self%cfg%nu;  kap_b = self%cfg%kappa
      ustar2 = sqrt(self%cfg%tau_x**2 + self%cfg%tau_y**2) / self%cfg%rho0
      !$omp parallel do collapse(2) default(shared) schedule(static) &
      !$omp private(i,j,k,im,jm,jj,dir,kk,e_surf,zs,zb,wet,dzi,n2,sh2,lup,ldn,lk,leps,km, &
      !$omp         sub,dia,sup,rhs,cst,xx,hcol,uc_a,uc_b,vc_a,vc_b,inv_dzi,a_up,a_dn,klay_a,klay_b, &
      !$omp         budget,l,n2_j,dz_j,cost,aa,s_part,step,denom,prod,diss)
      !$acc parallel loop collapse(2) gang vector copyin(recursive_mxl) &
      !$acc private(k,im,jm,jj,dir,kk,e_surf,zs,zb,wet,dzi,n2,sh2,lup,ldn,lk,leps,km, &
      !$acc         sub,dia,sup,rhs,cst,xx,hcol,uc_a,uc_b,vc_a,vc_b,inv_dzi,a_up,a_dn,klay_a,klay_b, &
      !$acc         budget,l,n2_j,dz_j,cost,aa,s_part,step,denom,prod,diss)
      do j = 1, self%ny
         do i = 1, self%nx
            im = merge(self%nx, i - 1, i == 1);  jm = merge(self%ny, j - 1, j == 1)
            ! interface geometry (1 = surface, nz+1 = bottom)
            zs(1) = 0.0_wp
            do k = 1, nz
               zs(k + 1) = zs(k) + self%d%dz3(i, j, k)
            end do
            hcol = zs(nz + 1)
            do k = 1, nz + 1
               zb(k) = max(hcol - zs(k), 0.0_wp)
               wet(k) = 0.0_wp;  n2(k) = 0.0_wp;  sh2(k) = 0.0_wp
            end do
            dzi(1) = self%d%dz3(i, j, 1);  dzi(nz + 1) = self%d%dz3(i, j, nz)
            do k = 2, nz
               dzi(k) = 0.5_wp * (self%d%dz3(i, j, k - 1) + self%d%dz3(i, j, k))
               wet(k) = self%d%mask3(i, j, k - 1) * self%d%mask3(i, j, k)
               inv_dzi = safe_inv(dzi(k))
               uc_a = 0.5_wp * (u(i, j, k - 1) + u(im, j, k - 1));  uc_b = 0.5_wp * (u(i, j, k) + u(im, j, k))
               vc_a = 0.5_wp * (v(i, j, k - 1) + v(i, jm, k - 1));  vc_b = 0.5_wp * (v(i, j, k) + v(i, jm, k))
               sh2(k) = (((uc_a - uc_b) * inv_dzi)**2 + ((vc_a - vc_b) * inv_dzi)**2) * wet(k)
               n2(k) = ((b(i, j, k - 1) - b(i, j, k)) * inv_dzi) * wet(k)
            end do
            ! Gaspar lengths, marching one interface at a time (closure.py::gaspar_lengths)
            do k = 1, nz + 1
               lup(k) = 0.0_wp;  ldn(k) = 0.0_wp
            end do
            if (recursive_mxl) then
               ! S11.2 `recursive`: buoyancy length capped by both walls, then
               ! limited to grow no faster than the grid (two O(nz) sweeps)
               do k = 1, nz + 1
                  l = sqrt(2.0_wp * max(self%tke(i, j, k), self%v5%e_min) / max(n2(k), self%v5%n2_min))
                  lup(k) = min(l, min(zs(k) + self%v5%z_0, zb(k) + self%v5%z_0))
               end do
               do k = 2, nz + 1
                  lup(k) = min(lup(k), lup(k - 1) + dzi(k))
               end do
               do k = nz, 1, -1
                  lup(k) = min(lup(k), lup(k + 1) + dzi(k + 1))
               end do
               do k = 1, nz + 1
                  lup(k) = lup(k) * wet(k);  ldn(k) = lup(k)
               end do
            else
            do dir = 1, 2
               do k = 1, nz + 1
                  l = 0.0_wp;  budget = max(self%tke(i, j, k), self%v5%e_min)
                  do jj = 1, nz
                     if (dir == 1) then
                        kk = k - jj
                     else
                        kk = k + jj
                     end if
                     if (kk >= 1 .and. kk <= nz + 1) then
                        n2_j = max(n2(kk), 0.0_wp);  dz_j = dzi(kk)
                     else
                        n2_j = 0.0_wp;  dz_j = 0.0_wp
                     end if
                     cost = n2_j * dz_j * (l + 0.5_wp * dz_j)
                     if (n2_j > 0.0_wp) then
                        aa = n2_j
                     else
                        aa = 1.0_wp
                     end if
                     s_part = sqrt(max(l * l + 2.0_wp * budget / aa, 0.0_wp)) - l
                     if (cost <= budget) then
                        step = dz_j
                     else if (n2_j > 0.0_wp) then
                        step = min(max(s_part, 0.0_wp), dz_j)
                     else
                        step = dz_j
                     end if
                     if (budget > 0.0_wp) l = l + step
                     if (cost <= budget) then
                        budget = budget - cost
                     else
                        budget = 0.0_wp
                     end if
                  end do
                  if (dir == 1) then
                     lup(k) = min(l, zs(k) + self%v5%z_0) * wet(k)
                  else
                     ldn(k) = min(l, zb(k) + self%v5%z_0) * wet(k)
                  end if
               end do
            end do
            end if
            do k = 1, nz + 1
               lk(k) = max(self%v5%l_min, sqrt(lup(k) * ldn(k)))
               leps(k) = max(self%v5%l_min, min(lup(k), ldn(k)))
            end do
            ! tridiagonal for e on interfaces: identity rows at the surface, bottom and dry
            e_surf = self%v5%e_bb * ustar2
            do k = 1, nz + 1
               prod = (self%nu3(i, j, k) - nu_b) * sh2(k) - (self%kap3(i, j, k) - kap_b) * n2(k)
               diss = self%v5%c_eps * sqrt(max(self%tke(i, j, k), self%v5%e_min)) / leps(k)
               if (k >= 2 .and. k <= nz) then
                  klay_a = 0.5_wp * (self%nu3(i, j, k - 1) + self%nu3(i, j, k))
                  klay_b = 0.5_wp * (self%nu3(i, j, k) + self%nu3(i, j, k + 1))
                  inv_dzi = safe_inv(dzi(k))
                  a_up = dt * klay_a * safe_inv(self%d%dz3(i, j, k - 1)) * inv_dzi
                  a_dn = dt * klay_b * safe_inv(self%d%dz3(i, j, k)) * inv_dzi
               else
                  a_up = 0.0_wp;  a_dn = 0.0_wp
               end if
               sub(k) = -a_up;  sup(k) = -a_dn
               dia(k) = 1.0_wp + a_up + a_dn + dt * diss
               rhs(k) = self%tke(i, j, k) + dt * prod
               if (wet(k) <= 0.0_wp) then
                  dia(k) = 1.0_wp;  sub(k) = 0.0_wp;  sup(k) = 0.0_wp;  rhs(k) = self%v5%e_min
               end if
            end do
            if (self%d%mask3(i, j, 1) > 0.0_wp) then
               rhs(1) = e_surf
            else
               rhs(1) = self%v5%e_min
            end if
            cst(1) = sup(1) / dia(1);  xx(1) = rhs(1) / dia(1)
            do k = 2, nz + 1
               denom = dia(k) - sub(k) * cst(k - 1)
               cst(k) = sup(k) / denom
               xx(k) = (rhs(k) - sub(k) * xx(k - 1)) / denom
            end do
            do k = nz, 1, -1
               xx(k) = xx(k) - cst(k) * xx(k + 1)
            end do
            do k = 1, nz + 1
               xx(k) = max(xx(k), self%v5%e_min)
               if (wet(k) <= 0.0_wp) xx(k) = self%v5%e_min
            end do
            xx(1) = rhs(1)
            do k = 1, nz + 1
               self%tke(i, j, k) = xx(k)
               km(k) = self%v5%c_k * lk(k) * sqrt(xx(k)) * wet(k) + nu_b
               self%nu3(i, j, k) = km(k)
               self%kap3(i, j, k) = (km(k) - nu_b) / self%v5%pr_t + kap_b
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end parallel do
      self%tridiagonal_solves = self%tridiagonal_solves + 1
   end subroutine closure_step

   ! ============================================================ vertical ops
   ! Tridiagonal coefficients of A = I - theta_v dt Dz on the partial-cell
   ! grid (libs/core/vertical_var.py::diffusion_coeffs_var).
   subroutine diffusion_coeffs_var(self, nu, dz3, thv, drag, sub, diag, sup)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: nu(:,:,:), dz3(:,:,:), thv, drag
      real(wp),         intent(out) :: sub(:,:,:), diag(:,:,:), sup(:,:,:)
      integer  :: i, j, k, nz
      real(wp) :: inv_dz, dzi_top, dzi_bot, at, ab
      logical  :: bottom, dry
      nz = self%nz
      !$acc parallel loop collapse(3) gang vector private(inv_dz,dzi_top,dzi_bot,at,ab,bottom,dry)
      do k = 1, nz
         do j = 1, self%ny
            do i = 1, self%nx
               dry = self%d%mask3(i, j, k) <= 0.0_wp
               if (k < nz) then
                  bottom = (.not. dry) .and. (self%d%mask3(i, j, k+1) <= 0.0_wp)
               else
                  bottom = .not. dry
               end if
               inv_dz = self%d%inv_dz3(i, j, k)
               if (k == 1) then
                  dzi_top = dz3(i, j, 1)
               else
                  dzi_top = 0.5_wp * (dz3(i, j, k-1) + dz3(i, j, k))
               end if
               if (k == nz) then
                  dzi_bot = dz3(i, j, nz)
               else
                  dzi_bot = 0.5_wp * (dz3(i, j, k) + dz3(i, j, k+1))
               end if
               at = thv * self%dt * inv_dz * nu(i, j, k)   * safe_inv(dzi_top)
               ab = thv * self%dt * inv_dz * nu(i, j, k+1) * safe_inv(dzi_bot)
               sub(i, j, k) = -at
               sup(i, j, k) = -ab
               diag(i, j, k) = 1.0_wp + at + ab
               if (k == 1) then
                  diag(i, j, k) = 1.0_wp + ab
                  sub(i, j, k) = 0.0_wp
               end if
               if (bottom) then
                  diag(i, j, k) = 1.0_wp + at + thv * self%dt * drag * inv_dz
                  sup(i, j, k) = 0.0_wp
               end if
               if (dry) then
                  diag(i, j, k) = 1.0_wp;  sub(i, j, k) = 0.0_wp;  sup(i, j, k) = 0.0_wp
               end if
            end do
         end do
      end do
      !$acc end parallel loop
   end subroutine diffusion_coeffs_var

   ! Called inside OpenACC loops, so it must be a device routine: without the
   ! directive nvfortran 25.11 dies with "could not get result type from opc".
   pure function safe_inv(x) result(r)
      !$acc routine seq
      real(wp), intent(in) :: x
      real(wp) :: r
      if (x > 0.0_wp) then
         r = 1.0_wp / x
      else
         r = 0.0_wp
      end if
   end function safe_inv

   ! Explicit Dz[field] on thickness dzx with the cell mask deciding the
   ! bottom (vertical_var.py::apply_diffusion_var). Orphaned OpenMP.
   subroutine apply_diffusion_var(self, field, nu, dzx, sflux, drag, out)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: field(:,:,:), nu(:,:,:), dzx(:,:,:), sflux(:,:), drag
      real(wp),         intent(out) :: out(:,:,:)
      integer  :: i, j, k, nz
      real(wp) :: ftop, fbot, dzi
      logical  :: bottom_here, bottom_above
      nz = self%nz
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(ftop,fbot,dzi,bottom_here,bottom_above)
      do k = 1, nz
         do j = 1, self%ny
            do i = 1, self%nx
               if (k < nz) then
                  bottom_here = self%d%mask3(i, j, k) > 0.0_wp .and. self%d%mask3(i, j, k+1) <= 0.0_wp
               else
                  bottom_here = self%d%mask3(i, j, k) > 0.0_wp
               end if
               ! flux through the top interface of layer k
               if (k == 1) then
                  ftop = sflux(i, j)
               else
                  if (k - 1 < nz) then
                     bottom_above = self%d%mask3(i, j, k-1) > 0.0_wp .and. self%d%mask3(i, j, k) <= 0.0_wp
                  else
                     bottom_above = .false.
                  end if
                  dzi = 0.5_wp * (dzx(i, j, k-1) + dzx(i, j, k))
                  ftop = nu(i, j, k) * (field(i, j, k-1) - field(i, j, k)) * safe_inv(dzi)
                  if (bottom_above) ftop = 0.0_wp
               end if
               ! flux through the bottom interface of layer k
               if (k == nz) then
                  fbot = 0.0_wp
               else
                  dzi = 0.5_wp * (dzx(i, j, k) + dzx(i, j, k+1))
                  fbot = nu(i, j, k+1) * (field(i, j, k) - field(i, j, k+1)) * safe_inv(dzi)
                  if (bottom_here) fbot = 0.0_wp
               end if
               out(i, j, k) = (ftop - fbot) * safe_inv(dzx(i, j, k))
               if (bottom_here) out(i, j, k) = out(i, j, k) - drag * field(i, j, k) * safe_inv(dzx(i, j, k))
               if (self%d%mask3(i, j, k) <= 0.0_wp) out(i, j, k) = 0.0_wp
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine apply_diffusion_var

   ! w at interfaces from the LAYER TRANSPORT divergence (vertical_var.py).
   subroutine w_from_transport(self, u, v, wf)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: u(:,:,:), v(:,:,:)
      real(wp),         intent(out) :: wf(:,:,:)
      integer  :: i, j, k, im, jm, nz
      real(wp) :: dv
      nz = self%nz
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(k,im,jm,dv)
      do j = 1, self%ny
         do i = 1, self%nx
            im = merge(self%nx, i - 1, i == 1)
            jm = merge(self%ny, j - 1, j == 1)
            wf(i, j, nz + 1) = 0.0_wp
            do k = nz, 1, -1
               dv = ((self%d%dz3u(i, j, k) * u(i, j, k) - self%d%dz3u(im, j, k) * u(im, j, k)) / self%g%dx &
                   + (self%d%dz3v(i, j, k) * v(i, j, k) - self%d%dz3v(i, jm, k) * v(i, jm, k)) / self%g%dy) &
                   * self%d%mask3(i, j, k)
               wf(i, j, k) = wf(i, j, k + 1) - dv
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine w_from_transport

   ! ============================================================ horizontal
   ! Masked Coriolis averages (operators_masked.py), 3D masks.
   subroutine avg_v_to_u_m(self, v, mv, mu, out)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: v(:,:,:), mv(:,:,:), mu(:,:,:)
      real(wp),         intent(out) :: out(:,:,:)
      integer  :: i, j, k, ip, jm
      real(wp) :: num, den
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(ip,jm,num,den)
      do k = 1, self%nz
         do j = 1, self%ny
            do i = 1, self%nx
               ip = merge(1, i + 1, i == self%nx)
               jm = merge(self%ny, j - 1, j == 1)
               num = mv(i, j, k) * v(i, j, k) + mv(i, jm, k) * v(i, jm, k) &
                   + mv(ip, j, k) * v(ip, j, k) + mv(ip, jm, k) * v(ip, jm, k)
               den = mv(i, j, k) + mv(i, jm, k) + mv(ip, j, k) + mv(ip, jm, k)
               if (den > 0.0_wp) then
                  out(i, j, k) = num / den * mu(i, j, k)
               else
                  out(i, j, k) = 0.0_wp
               end if
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine avg_v_to_u_m

   subroutine avg_u_to_v_m(self, u, mu, mv, out)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: u(:,:,:), mu(:,:,:), mv(:,:,:)
      real(wp),         intent(out) :: out(:,:,:)
      integer  :: i, j, k, im, jp
      real(wp) :: num, den
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(im,jp,num,den)
      do k = 1, self%nz
         do j = 1, self%ny
            do i = 1, self%nx
               im = merge(self%nx, i - 1, i == 1)
               jp = merge(1, j + 1, j == self%ny)
               num = mu(i, j, k) * u(i, j, k) + mu(im, j, k) * u(im, j, k) &
                   + mu(i, jp, k) * u(i, jp, k) + mu(im, jp, k) * u(im, jp, k)
               den = mu(i, j, k) + mu(im, j, k) + mu(i, jp, k) + mu(im, jp, k)
               if (den > 0.0_wp) then
                  out(i, j, k) = num / den * mv(i, j, k)
               else
                  out(i, j, k) = 0.0_wp
               end if
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine avg_u_to_v_m

   ! 2D versions for the barotropic substep (2D masks).
   subroutine avg2_v_to_u(self, v, out)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: v(:,:)
      real(wp),         intent(out) :: out(:,:)
      integer  :: i, j, ip, jm
      real(wp) :: num, den
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(ip,jm,num,den)
      do j = 1, self%ny
         do i = 1, self%nx
            ip = merge(1, i + 1, i == self%nx)
            jm = merge(self%ny, j - 1, j == 1)
            num = self%d%maskv(i, j) * v(i, j) + self%d%maskv(i, jm) * v(i, jm) &
                + self%d%maskv(ip, j) * v(ip, j) + self%d%maskv(ip, jm) * v(ip, jm)
            den = self%d%maskv(i, j) + self%d%maskv(i, jm) + self%d%maskv(ip, j) + self%d%maskv(ip, jm)
            if (den > 0.0_wp) then
               out(i, j) = num / den * self%d%masku(i, j)
            else
               out(i, j) = 0.0_wp
            end if
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine avg2_v_to_u

   subroutine avg2_u_to_v(self, u, out)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: u(:,:)
      real(wp),         intent(out) :: out(:,:)
      integer  :: i, j, im, jp
      real(wp) :: num, den
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(im,jp,num,den)
      do j = 1, self%ny
         do i = 1, self%nx
            im = merge(self%nx, i - 1, i == 1)
            jp = merge(1, j + 1, j == self%ny)
            num = self%d%masku(i, j) * u(i, j) + self%d%masku(im, j) * u(im, j) &
                + self%d%masku(i, jp) * u(i, jp) + self%d%masku(im, jp) * u(im, jp)
            den = self%d%masku(i, j) + self%d%masku(im, j) + self%d%masku(i, jp) + self%d%masku(im, jp)
            if (den > 0.0_wp) then
               out(i, j) = num / den * self%d%maskv(i, j)
            else
               out(i, j) = 0.0_wp
            end if
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine avg2_u_to_v

   ! Depth-integrated transport of a 3D face velocity.
   subroutine depth_transport(self, u, dzx, out)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: u(:,:,:), dzx(:,:,:)
      real(wp),         intent(out) :: out(:,:)
      integer :: i, j, k
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(k)
      do j = 1, self%ny
         do i = 1, self%nx
            out(i, j) = 0.0_wp
            do k = 1, self%nz
               out(i, j) = out(i, j) + dzx(i, j, k) * u(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine depth_transport

   ! Masked 2D divergence (u, v already carry their face masks).
   subroutine div2_m(self, u, v, out)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: u(:,:), v(:,:)
      real(wp),         intent(out) :: out(:,:)
      integer :: i, j, im, jm
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(im,jm)
      do j = 1, self%ny
         do i = 1, self%nx
            im = merge(self%nx, i - 1, i == 1)
            jm = merge(self%ny, j - 1, j == 1)
            out(i, j) = self%d%mask(i, j) * ((u(i, j) - u(im, j)) / self%g%dx &
                                           + (v(i, j) - v(i, jm)) / self%g%dy)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine div2_m

   ! Masked 2D gradients of eta at u and v faces.
   subroutine grad2_m(self, eta, gx, gy)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: eta(:,:)
      real(wp),         intent(out) :: gx(:,:), gy(:,:)
      integer :: i, j, ip, jp
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(ip,jp)
      do j = 1, self%ny
         do i = 1, self%nx
            ip = merge(1, i + 1, i == self%nx)
            jp = merge(1, j + 1, j == self%ny)
            gx(i, j) = self%d%masku(i, j) * (eta(ip, j) - eta(i, j)) / self%g%dx
            gy(i, j) = self%d%maskv(i, j) * (eta(i, jp) - eta(i, j)) / self%g%dy
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine grad2_m

   ! Masked horizontal Laplacian of a 3D cell or face field (D o G).
   subroutine laplacian_h_m(self, a, mu, mv, mc, out)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: a(:,:,:), mu(:,:,:), mv(:,:,:), mc(:,:,:)
      real(wp),         intent(out) :: out(:,:,:)
      integer  :: i, j, k, ip, im, jp, jm
      real(wp) :: fxe, fxw, fyn, fys
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(ip,im,jp,jm,fxe,fxw,fyn,fys)
      do k = 1, self%nz
         do j = 1, self%ny
            do i = 1, self%nx
               ip = merge(1, i + 1, i == self%nx);  im = merge(self%nx, i - 1, i == 1)
               jp = merge(1, j + 1, j == self%ny);  jm = merge(self%ny, j - 1, j == 1)
               fxe = mu(i, j, k)  * (a(ip, j, k) - a(i, j, k))  / self%g%dx
               fxw = mu(im, j, k) * (a(i, j, k)  - a(im, j, k)) / self%g%dx
               fyn = mv(i, j, k)  * (a(i, jp, k) - a(i, j, k))  / self%g%dy
               fys = mv(i, jm, k) * (a(i, j, k)  - a(i, jm, k)) / self%g%dy
               out(i, j, k) = mc(i, j, k) * ((fxe - fxw) / self%g%dx + (fyn - fys) / self%g%dy)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine laplacian_h_m

   ! ======================================================== pressure gradient
   ! Baroclinic pressure gradient at a common face depth (spec S10.7).
   subroutine pressure_gradients(self, b)
      type(stepper5_t), intent(inout) :: self
      real(wp),         intent(in)    :: b(:,:,:)
      integer  :: i, j, k, ip, jp
      real(wp) :: above, tot, mean, zf, left, right
      logical  :: do_remove, do_corr
      ! Character intrinsics inside an OpenACC region are what nvfortran
      ! 25.9/25.11 choke on ("could not get result type from opc", bisected
      ! to this routine); the two switches are evaluated on the host.
      do_remove = trim(self%v5%pgf) == 'remove'
      do_corr = self%v5%pgf_correction
      ! Phi[k] = 0.5 dz3[k] b[k] + sum_{k'<k} dz3[k'] b[k'], depth mean removed if asked.
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(k,above,tot,mean)
      do j = 1, self%ny
         do i = 1, self%nx
            above = 0.0_wp;  tot = 0.0_wp;  mean = 0.0_wp
            do k = 1, self%nz
               self%phi(i, j, k) = 0.5_wp * self%d%dz3(i, j, k) * b(i, j, k) + above
               above = above + self%d%dz3(i, j, k) * b(i, j, k)
               tot = tot + self%d%dz3(i, j, k)
               mean = mean + self%d%dz3(i, j, k) * self%phi(i, j, k)
            end do
            if (do_remove) then
               mean = mean * safe_inv(tot)
               do k = 1, self%nz
                  self%phi(i, j, k) = self%phi(i, j, k) - mean
               end do
            end if
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(ip,jp,zf,left,right)
      do k = 1, self%nz
         do j = 1, self%ny
            do i = 1, self%nx
               ip = merge(1, i + 1, i == self%nx)
               jp = merge(1, j + 1, j == self%ny)
               if (do_corr) then
                  zf = 0.5_wp * (self%d%zc(i, j, k) + self%d%zc(ip, j, k))
                  left  = self%phi(i, j, k)  + b(i, j, k)  * (zf - self%d%zc(i, j, k))
                  right = self%phi(ip, j, k) + b(ip, j, k) * (zf - self%d%zc(ip, j, k))
                  self%pgx(i, j, k) = self%d%mask3u(i, j, k) * (right - left) / self%g%dx
                  zf = 0.5_wp * (self%d%zc(i, j, k) + self%d%zc(i, jp, k))
                  left  = self%phi(i, j, k)  + b(i, j, k)  * (zf - self%d%zc(i, j, k))
                  right = self%phi(i, jp, k) + b(i, jp, k) * (zf - self%d%zc(i, jp, k))
                  self%pgy(i, j, k) = self%d%mask3v(i, j, k) * (right - left) / self%g%dy
               else
                  self%pgx(i, j, k) = self%d%mask3u(i, j, k) * (self%phi(ip, j, k) - self%phi(i, j, k)) / self%g%dx
                  self%pgy(i, j, k) = self%d%mask3v(i, j, k) * (self%phi(i, jp, k) - self%phi(i, j, k)) / self%g%dy
               end if
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine pressure_gradients

   ! ================================================================ advection
   ! Second-order centred flux-form momentum advection with the NOMINAL dz
   ! in the vertical term, exactly as libs/core/advection.py does; the
   ! result is then masked. Reproducing the reference is the point.
   subroutine momentum_advection(self, u, v, wf, exu, exv)
      type(stepper5_t), intent(in)    :: self
      real(wp),         intent(in)    :: u(:,:,:), v(:,:,:), wf(:,:,:)
      real(wp),         intent(inout) :: exu(:,:,:), exv(:,:,:)
      integer  :: i, j, k, ip, im, jp, jm, nz
      real(wp) :: ubc_e, ubc, fyu_n, fyu_s, fzt, fzb, vbc_n, vbc, fxv_e, fxv_w, wb
      nz = self%nz
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(ip,im,jp,jm,ubc_e,ubc,fyu_n,fyu_s,fzt,fzb,vbc_n,vbc,fxv_e,fxv_w,wb)
      do k = 1, nz
         do j = 1, self%ny
            do i = 1, self%nx
               ip = merge(1, i + 1, i == self%nx);  im = merge(self%nx, i - 1, i == 1)
               jp = merge(1, j + 1, j == self%ny);  jm = merge(self%ny, j - 1, j == 1)
               ! ---- u
               ubc   = 0.5_wp * (u(im, j, k) + u(i, j, k))       ! cell centre i
               ubc_e = 0.5_wp * (u(i, j, k) + u(ip, j, k))       ! cell centre i+1
               fyu_n = 0.5_wp * (v(i, j, k) + v(ip, j, k)) * 0.5_wp * (u(i, j, k) + u(i, jp, k))
               fyu_s = 0.5_wp * (v(i, jm, k) + v(ip, jm, k)) * 0.5_wp * (u(i, jm, k) + u(i, j, k))
               fzt = 0.0_wp;  fzb = 0.0_wp
               if (k > 1) then
                  wb = 0.5_wp * (wf(i, j, k) + wf(ip, j, k))
                  fzt = wb * 0.5_wp * (u(i, j, k-1) + u(i, j, k))
               end if
               if (k < nz) then
                  wb = 0.5_wp * (wf(i, j, k+1) + wf(ip, j, k+1))
                  fzb = wb * 0.5_wp * (u(i, j, k) + u(i, j, k+1))
               end if
               exu(i, j, k) = exu(i, j, k) - self%d%mask3u(i, j, k) * ( &
                  (ubc_e**2 - ubc**2) / self%g%dx + (fyu_n - fyu_s) / self%g%dy + (fzt - fzb) / self%dz_nom)
               ! ---- v
               vbc   = 0.5_wp * (v(i, jm, k) + v(i, j, k))
               vbc_n = 0.5_wp * (v(i, j, k) + v(i, jp, k))
               fxv_e = 0.5_wp * (u(i, j, k) + u(i, jp, k)) * 0.5_wp * (v(i, j, k) + v(ip, j, k))
               fxv_w = 0.5_wp * (u(im, j, k) + u(im, jp, k)) * 0.5_wp * (v(im, j, k) + v(i, j, k))
               fzt = 0.0_wp;  fzb = 0.0_wp
               if (k > 1) then
                  wb = 0.5_wp * (wf(i, j, k) + wf(i, jp, k))
                  fzt = wb * 0.5_wp * (v(i, j, k-1) + v(i, j, k))
               end if
               if (k < nz) then
                  wb = 0.5_wp * (wf(i, j, k+1) + wf(i, jp, k+1))
                  fzb = wb * 0.5_wp * (v(i, j, k) + v(i, j, k+1))
               end if
               exv(i, j, k) = exv(i, j, k) - self%d%mask3v(i, j, k) * ( &
                  (vbc_n**2 - vbc**2) / self%g%dy + (fxv_e - fxv_w) / self%g%dx + (fzt - fzb) / self%dz_nom)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine momentum_advection

   ! Third-order upwind-biased face value (spec S11.3, advection_v06.py::kappa_face):
   ! the face between c and cp for the advecting velocity vel; cm = c[i-1],
   ! cpp = c[i+2]; ok_pos/ok_neg say whether the far cell is wet (else upwind1).
   pure function kface(cm, c, cp, cpp, vel, ok_pos, ok_neg, limited) result(f)
      !$acc routine seq
      real(wp), intent(in) :: cm, c, cp, cpp, vel
      logical,  intent(in) :: ok_pos, ok_neg, limited
      real(wp), parameter  :: KAP = 1.0_wp / 3.0_wp
      real(wp) :: f, dc, r_pos, r_neg, phi_pos, phi_neg, f_pos, f_neg
      if (limited) then
         dc = cp - c
         if (dc == 0.0_wp) then
            r_pos = 0.0_wp;  r_neg = 0.0_wp
         else
            r_pos = (c - cm) / dc;  r_neg = (cpp - cp) / dc
         end if
         phi_pos = max(0.0_wp, max(min(2.0_wp * r_pos, 1.0_wp), min(r_pos, 2.0_wp)))
         phi_neg = max(0.0_wp, max(min(2.0_wp * r_neg, 1.0_wp), min(r_neg, 2.0_wp)))
         f_pos = c + 0.5_wp * phi_pos * dc
         f_neg = cp - 0.5_wp * phi_neg * dc
      else
         f_pos = c + 0.25_wp * ((1.0_wp - KAP) * (c - cm) + (1.0_wp + KAP) * (cp - c))
         f_neg = cp - 0.25_wp * ((1.0_wp - KAP) * (cpp - cp) + (1.0_wp + KAP) * (cp - c))
      end if
      if (.not. ok_pos) f_pos = c
      if (.not. ok_neg) f_neg = cp
      if (vel > 0.0_wp) then
         f = f_pos
      else
         f = f_neg
      end if
   end function kface

   ! Momentum advection with kappa-interpolated advected velocity
   ! (advection_v06.py::momentum_advection_up3).
   subroutine momentum_advection_up3(self, u, v, wf, exu, exv)
      type(stepper5_t), intent(in)    :: self
      real(wp),         intent(in)    :: u(:,:,:), v(:,:,:), wf(:,:,:)
      real(wp),         intent(inout) :: exu(:,:,:), exv(:,:,:)
      integer  :: i, j, k, ip, im, ipp, imm, jp, jm, jpp, jmm, nz, nx, ny
      real(wp) :: ubc, ubc_e, uac, uac_e, vbar, vbar_s, uay, uay_s, fzt, fzb, wb
      real(wp) :: vbc, vbc_n, vac, vac_n, ubar, ubar_w, vax, vax_w, cmv, cppv
      logical  :: lim, okp, okn
      nz = self%nz;  nx = self%nx;  ny = self%ny;  lim = self%is_tvd
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(ip,im,ipp,imm,jp,jm,jpp,jmm,ubc,ubc_e,uac,uac_e,vbar,vbar_s,uay,uay_s,fzt,fzb,wb,vbc,vbc_n,vac,vac_n,ubar,ubar_w,vax,vax_w,cmv,cppv,okp,okn)
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               ip = merge(1, i + 1, i == nx);  im = merge(nx, i - 1, i == 1)
               ipp = merge(ip + 1 - nx, ip + 1, ip == nx);  imm = merge(im - 1 + nx, im - 1, im == 1)
               jp = merge(1, j + 1, j == ny);  jm = merge(ny, j - 1, j == 1)
               jpp = merge(jp + 1 - ny, jp + 1, jp == ny);  jmm = merge(jm - 1 + ny, jm - 1, jm == 1)
               ! ---- u: x flux at centres i and i+1
               ubc   = 0.5_wp * (u(im, j, k) + u(i, j, k))
               ubc_e = 0.5_wp * (u(i, j, k) + u(ip, j, k))
               uac   = kface(u(imm, j, k), u(im, j, k), u(i, j, k), u(ip, j, k), ubc, &
                             self%d%mask3u(imm, j, k) > 0.0_wp, self%d%mask3u(ip, j, k) > 0.0_wp, lim)
               uac_e = kface(u(im, j, k), u(i, j, k), u(ip, j, k), u(ipp, j, k), ubc_e, &
                             self%d%mask3u(im, j, k) > 0.0_wp, self%d%mask3u(ipp, j, k) > 0.0_wp, lim)
               ! y flux at corners j+1/2 and j-1/2
               vbar   = 0.5_wp * (v(i, j, k) + v(ip, j, k))
               vbar_s = 0.5_wp * (v(i, jm, k) + v(ip, jm, k))
               uay   = kface(u(i, jm, k), u(i, j, k), u(i, jp, k), u(i, jpp, k), vbar, &
                             self%d%mask3u(i, jm, k) > 0.0_wp, self%d%mask3u(i, jpp, k) > 0.0_wp, lim)
               uay_s = kface(u(i, jmm, k), u(i, jm, k), u(i, j, k), u(i, jp, k), vbar_s, &
                             self%d%mask3u(i, jmm, k) > 0.0_wp, self%d%mask3u(i, jp, k) > 0.0_wp, lim)
               fzt = 0.0_wp;  fzb = 0.0_wp
               if (k > 1) then
                  wb = 0.5_wp * (wf(i, j, k) + wf(ip, j, k))
                  cmv = 0.0_wp;  okp = .false.;  cppv = 0.0_wp;  okn = .false.
                  if (k + 1 <= nz) then
                     cmv = u(i, j, k + 1);  okp = self%d%mask3u(i, j, k + 1) > 0.0_wp
                  end if
                  if (k - 2 >= 1) then
                     cppv = u(i, j, k - 2);  okn = self%d%mask3u(i, j, k - 2) > 0.0_wp
                  end if
                  fzt = wb * kface(cmv, u(i, j, k), u(i, j, k - 1), cppv, wb, okp, okn, lim)
               end if
               if (k < nz) then
                  wb = 0.5_wp * (wf(i, j, k + 1) + wf(ip, j, k + 1))
                  cmv = 0.0_wp;  okp = .false.;  cppv = 0.0_wp;  okn = .false.
                  if (k + 2 <= nz) then
                     cmv = u(i, j, k + 2);  okp = self%d%mask3u(i, j, k + 2) > 0.0_wp
                  end if
                  if (k - 1 >= 1) then
                     cppv = u(i, j, k - 1);  okn = self%d%mask3u(i, j, k - 1) > 0.0_wp
                  end if
                  fzb = wb * kface(cmv, u(i, j, k + 1), u(i, j, k), cppv, wb, okp, okn, lim)
               end if
               exu(i, j, k) = exu(i, j, k) - self%d%mask3u(i, j, k) * ( &
                  (ubc_e * uac_e - ubc * uac) / self%g%dx + (vbar * uay - vbar_s * uay_s) / self%g%dy &
                  + (fzt - fzb) / self%dz_nom)
               ! ---- v
               vbc   = 0.5_wp * (v(i, jm, k) + v(i, j, k))
               vbc_n = 0.5_wp * (v(i, j, k) + v(i, jp, k))
               vac   = kface(v(i, jmm, k), v(i, jm, k), v(i, j, k), v(i, jp, k), vbc, &
                             self%d%mask3v(i, jmm, k) > 0.0_wp, self%d%mask3v(i, jp, k) > 0.0_wp, lim)
               vac_n = kface(v(i, jm, k), v(i, j, k), v(i, jp, k), v(i, jpp, k), vbc_n, &
                             self%d%mask3v(i, jm, k) > 0.0_wp, self%d%mask3v(i, jpp, k) > 0.0_wp, lim)
               ubar   = 0.5_wp * (u(i, j, k) + u(i, jp, k))
               ubar_w = 0.5_wp * (u(im, j, k) + u(im, jp, k))
               vax   = kface(v(im, j, k), v(i, j, k), v(ip, j, k), v(ipp, j, k), ubar, &
                             self%d%mask3v(im, j, k) > 0.0_wp, self%d%mask3v(ipp, j, k) > 0.0_wp, lim)
               vax_w = kface(v(imm, j, k), v(im, j, k), v(i, j, k), v(ip, j, k), ubar_w, &
                             self%d%mask3v(imm, j, k) > 0.0_wp, self%d%mask3v(ip, j, k) > 0.0_wp, lim)
               fzt = 0.0_wp;  fzb = 0.0_wp
               if (k > 1) then
                  wb = 0.5_wp * (wf(i, j, k) + wf(i, jp, k))
                  cmv = 0.0_wp;  okp = .false.;  cppv = 0.0_wp;  okn = .false.
                  if (k + 1 <= nz) then
                     cmv = v(i, j, k + 1);  okp = self%d%mask3v(i, j, k + 1) > 0.0_wp
                  end if
                  if (k - 2 >= 1) then
                     cppv = v(i, j, k - 2);  okn = self%d%mask3v(i, j, k - 2) > 0.0_wp
                  end if
                  fzt = wb * kface(cmv, v(i, j, k), v(i, j, k - 1), cppv, wb, okp, okn, lim)
               end if
               if (k < nz) then
                  wb = 0.5_wp * (wf(i, j, k + 1) + wf(i, jp, k + 1))
                  cmv = 0.0_wp;  okp = .false.;  cppv = 0.0_wp;  okn = .false.
                  if (k + 2 <= nz) then
                     cmv = v(i, j, k + 2);  okp = self%d%mask3v(i, j, k + 2) > 0.0_wp
                  end if
                  if (k - 1 >= 1) then
                     cppv = v(i, j, k - 1);  okn = self%d%mask3v(i, j, k - 1) > 0.0_wp
                  end if
                  fzb = wb * kface(cmv, v(i, j, k + 1), v(i, j, k), cppv, wb, okp, okn, lim)
               end if
               exv(i, j, k) = exv(i, j, k) - self%d%mask3v(i, j, k) * ( &
                  (vbc_n * vac_n - vbc * vac) / self%g%dy + (ubar * vax - ubar_w * vax_w) / self%g%dx &
                  + (fzt - fzb) / self%dz_nom)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine momentum_advection_up3

   ! Flux-form tracer advection with real face thickness; the surface
   ! interface carries the flux w[0]*c[0] (model3d_v05.py::_tracer_advection).
   subroutine tracer_advection(self, c, u, v, wf, out)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: c(:,:,:), u(:,:,:), v(:,:,:), wf(:,:,:)
      real(wp),         intent(out) :: out(:,:,:)
      integer  :: i, j, k, ip, im, ipp, imm, jp, jm, jpp, jmm, nz
      real(wp) :: fxe, fxw, fyn, fys, cxe, cxw, cyn, cys, cft, cfb, hdiv, vdiv, cmv, cppv
      logical  :: upwind, up3, lim, okp, okn
      nz = self%nz
      upwind = trim(self%v5%advection) == 'upwind1'
      up3 = self%is_up3;  lim = self%is_tvd
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(ip,im,ipp,imm,jp,jm,jpp,jmm,fxe,fxw,fyn,fys,cxe,cxw,cyn,cys,cft,cfb,hdiv,vdiv,cmv,cppv,okp,okn)
      do k = 1, nz
         do j = 1, self%ny
            do i = 1, self%nx
               ip = merge(1, i + 1, i == self%nx);  im = merge(self%nx, i - 1, i == 1)
               jp = merge(1, j + 1, j == self%ny);  jm = merge(self%ny, j - 1, j == 1)
               if (up3) then
                  ipp = merge(1, ip + 1, ip == self%nx);  imm = merge(self%nx, im - 1, im == 1)
                  jpp = merge(1, jp + 1, jp == self%ny);  jmm = merge(self%ny, jm - 1, jm == 1)
                  cxe = kface(c(im, j, k), c(i, j, k), c(ip, j, k), c(ipp, j, k), u(i, j, k), &
                              self%d%mask3(im, j, k) > 0.0_wp, self%d%mask3(ipp, j, k) > 0.0_wp, lim)
                  cxw = kface(c(imm, j, k), c(im, j, k), c(i, j, k), c(ip, j, k), u(im, j, k), &
                              self%d%mask3(imm, j, k) > 0.0_wp, self%d%mask3(ip, j, k) > 0.0_wp, lim)
                  cyn = kface(c(i, jm, k), c(i, j, k), c(i, jp, k), c(i, jpp, k), v(i, j, k), &
                              self%d%mask3(i, jm, k) > 0.0_wp, self%d%mask3(i, jpp, k) > 0.0_wp, lim)
                  cys = kface(c(i, jmm, k), c(i, jm, k), c(i, j, k), c(i, jp, k), v(i, jm, k), &
                              self%d%mask3(i, jmm, k) > 0.0_wp, self%d%mask3(i, jp, k) > 0.0_wp, lim)
               else if (upwind) then
                  cxe = merge(c(i, j, k),  c(ip, j, k), u(i, j, k)  > 0.0_wp)
                  cxw = merge(c(im, j, k), c(i, j, k),  u(im, j, k) > 0.0_wp)
                  cyn = merge(c(i, j, k),  c(i, jp, k), v(i, j, k)  > 0.0_wp)
                  cys = merge(c(i, jm, k), c(i, j, k),  v(i, jm, k) > 0.0_wp)
               else
                  cxe = 0.5_wp * (c(i, j, k) + c(ip, j, k))
                  cxw = 0.5_wp * (c(im, j, k) + c(i, j, k))
                  cyn = 0.5_wp * (c(i, j, k) + c(i, jp, k))
                  cys = 0.5_wp * (c(i, jm, k) + c(i, j, k))
               end if
               fxe = self%d%dz3u(i, j, k)  * u(i, j, k)  * cxe
               fxw = self%d%dz3u(im, j, k) * u(im, j, k) * cxw
               fyn = self%d%dz3v(i, j, k)  * v(i, j, k)  * cyn
               fys = self%d%dz3v(i, jm, k) * v(i, jm, k) * cys
               hdiv = ((fxe - fxw) / self%g%dx + (fyn - fys) / self%g%dy) * self%d%inv_dz3(i, j, k)
               ! interface above layer k (k = 1: the surface carries c(1))
               if (k == 1) then
                  cft = c(i, j, 1)
               else if (up3) then
                  cmv = 0.0_wp;  okp = .false.;  cppv = 0.0_wp;  okn = .false.
                  if (k + 1 <= nz) then
                     cmv = c(i, j, k + 1);  okp = self%d%mask3(i, j, k + 1) > 0.0_wp
                  end if
                  if (k - 2 >= 1) then
                     cppv = c(i, j, k - 2);  okn = self%d%mask3(i, j, k - 2) > 0.0_wp
                  end if
                  cft = kface(cmv, c(i, j, k), c(i, j, k - 1), cppv, wf(i, j, k), okp, okn, lim)
               else
                  cft = 0.5_wp * (c(i, j, k-1) + c(i, j, k))
               end if
               ! interface below layer k (k = nz: bottom, no flux)
               if (k == nz) then
                  cfb = 0.0_wp
               else if (up3) then
                  cmv = 0.0_wp;  okp = .false.;  cppv = 0.0_wp;  okn = .false.
                  if (k + 2 <= nz) then
                     cmv = c(i, j, k + 2);  okp = self%d%mask3(i, j, k + 2) > 0.0_wp
                  end if
                  if (k - 1 >= 1) then
                     cppv = c(i, j, k - 1);  okn = self%d%mask3(i, j, k - 1) > 0.0_wp
                  end if
                  cfb = kface(cmv, c(i, j, k + 1), c(i, j, k), cppv, wf(i, j, k + 1), okp, okn, lim)
               else
                  cfb = 0.5_wp * (c(i, j, k) + c(i, j, k+1))
               end if
               vdiv = (wf(i, j, k) * cft - wf(i, j, k+1) * cfb) * self%d%inv_dz3(i, j, k)
               out(i, j, k) = (hdiv + vdiv) * self%d%mask3(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine tracer_advection

   ! One tracer step: explicit advection and horizontal diffusion, then the
   ! implicit vertical solve carrying the surface flux. Orphaned OpenMP.
   subroutine advance_tracer(self, c, u, v, wf, sflux, source, cnew)
      type(stepper5_t), intent(inout) :: self
      real(wp),         intent(in)    :: c(:,:,:), u(:,:,:), v(:,:,:), wf(:,:,:), sflux(:,:)
      real(wp),         intent(in)    :: source(:,:,:)
      real(wp),         intent(out)   :: cnew(:,:,:)
      real(wp) :: dt, thv
      integer  :: i, j, k
      dt = self%dt;  thv = self%cfg%theta_v
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, self%nz
         do j = 1, self%ny
            do i = 1, self%nx
               self%tend(i, j, k) = source(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      if (self%do_adv) then
         call tracer_advection(self, c, u, v, wf, self%w3b)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  self%tend(i, j, k) = self%tend(i, j, k) - self%w3b(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end if
      if (self%v5%k_h > 0.0_wp) then
         call laplacian_h_m(self, c, self%d%mask3u, self%d%mask3v, self%d%mask3, self%w3b)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  self%tend(i, j, k) = self%tend(i, j, k) + self%v5%k_h * self%w3b(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end if
      call apply_diffusion_var(self, c, self%kap3, self%d%dz3, sflux, 0.0_wp, self%dzb)
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, self%nz
         do j = 1, self%ny
            do i = 1, self%nx
               self%trhs(i, j, k) = c(i, j, k) + dt * self%tend(i, j, k) &
                                  + (1.0_wp - thv) * dt * self%dzb(i, j, k)
               if (k == 1) self%trhs(i, j, k) = self%trhs(i, j, k) &
                                  + thv * dt * sflux(i, j) * self%d%inv_dz3(i, j, 1)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      call thomas_batched(self%g, self%nz, self%bsub, self%bdiag, self%bsup, &
                          self%trhs, cnew, self%cstar, self%dstar)
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, self%nz
         do j = 1, self%ny
            do i = 1, self%nx
               cnew(i, j, k) = cnew(i, j, k) * self%d%mask3(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      self%tridiagonal_solves = self%tridiagonal_solves + 1
   end subroutine advance_tracer

   ! ===================================================================== EOS
   ! b = -g rho'/rho0 at the three levels of spec S10.6 (libs/core/eos.py).
   ! One plain loop per level: nvfortran 25.11 raised an internal compiler
   ! error on a select-case-plus-function-call inside an OpenACC loop, so the
   ! polyTEOS10 polynomial is written out in its loop body.
   subroutine buoyancy_eos(self, t, s, b)
      type(stepper5_t), intent(in)  :: self
      real(wp),         intent(in)  :: t(:,:,:), s(:,:,:)
      real(wp),         intent(out) :: b(:,:,:)
      integer  :: i, j, k
      real(wp) :: g_, rho0, rp, ta, sa, z, zh, ss, tt, r0, rz0, rz1, rz2, rz3
      real(wp), parameter :: SA0 = 1.6550e-1_wp, SB0 = 7.6554e-1_wp, SL1 = 5.9520e-2_wp, &
                             SL2 = 7.4914e-4_wp, SM1 = 1.4970e-4_wp, SM2 = 1.1090e-5_wp, &
                             SNU = 2.4341e-3_wp
      g_ = self%cfg%g;  rho0 = self%cfg%rho0
      if (trim(self%v5%eos) == 'linear') then
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector private(rp)
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  rp = rho0 * (-self%v5%alpha_t * (t(i, j, k) - self%v5%t0) &
                               + self%v5%beta_s * (s(i, j, k) - self%v5%s0))
                  b(i, j, k) = -g_ * rp / rho0 * self%d%mask3(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      else if (trim(self%v5%eos) == 'seos') then
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector private(rp,ta,sa,z)
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  z = self%d%zc(i, j, k)
                  ta = t(i, j, k) - 10.0_wp;  sa = s(i, j, k) - 35.0_wp
                  rp = -SA0 * (1.0_wp + 0.5_wp * SL1 * ta + SM1 * z) * ta &
                     +  SB0 * (1.0_wp - 0.5_wp * SL2 * sa - SM2 * z) * sa &
                     -  SNU * ta * sa
                  b(i, j, k) = -g_ * rp / rho0 * self%d%mask3(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      else if (trim(self%v5%eos) == 'teos10') then
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector private(rp,z,zh,ss,tt,r0,rz0,rz1,rz2,rz3)
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  z = self%d%zc(i, j, k)
                  zh = z * R1_Z0
                  ss = sqrt((s(i, j, k) + RDELTA_S) * R1_S0)
                  tt = t(i, j, k) * R1_T0
                  r0 = (((((R05 * zh + R04) * zh + R03) * zh + R02) * zh + R01) * zh + R00) * zh
                  rz3 = E013 * tt + E103 * ss + E003
                  rz2 = (E022 * tt + E112 * ss + E012) * tt + (E202 * ss + E102) * ss + E002
                  rz1 = (((E041 * tt + E131 * ss + E031) * tt + (E221 * ss + E121) * ss + E021) * tt &
                         + ((E311 * ss + E211) * ss + E111) * ss + E011) * tt &
                      + (((E401 * ss + E301) * ss + E201) * ss + E101) * ss + E001
                  rz0 = ((((( E060 * tt + E150 * ss + E050 ) * tt + (E240 * ss + E140) * ss + E040 ) * tt &
                           + ((E330 * ss + E230) * ss + E130) * ss + E030 ) * tt &
                          + (((E420 * ss + E320) * ss + E220) * ss + E120) * ss + E020 ) * tt &
                         + ((((E510 * ss + E410) * ss + E310) * ss + E210) * ss + E110) * ss + E010 ) * tt &
                      + (((((E600 * ss + E500) * ss + E400) * ss + E300) * ss + E200) * ss + E100) * ss + E000
                  rp = ((rz3 * zh + rz2) * zh + rz1) * zh + rz0 + r0 - rho0
                  b(i, j, k) = -g_ * rp / rho0 * self%d%mask3(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      else
         write(*, '(a)') 'FATAL: unknown eos '//trim(self%v5%eos); error stop 1
      end if
   end subroutine buoyancy_eos

   ! ==================================================================== step
   subroutine stepper5_step(self, u, v, b, eta, t, s)
      type(stepper5_t), intent(inout) :: self
      real(wp),         intent(inout) :: u(:,:,:), v(:,:,:), b(:,:,:), eta(:,:)
      real(wp),         intent(inout) :: t(:,:,:), s(:,:,:)
      real(wp) :: g_, f, dt, th, tc, thv, c1, c2
      integer  :: m, i, j, k, nz, nx, ny

      g_ = self%cfg%g;  f = self%cfg%f0;  dt = self%dt
      th = self%cfg%theta;  tc = self%cfg%theta_cor;  thv = self%cfg%theta_v
      if (.not. self%is_theta) th = 0.0_wp     ! fb: full explicit gradient (N14)
      c1 = g_ * dt * (1.0_wp - th);  c2 = g_ * dt * th
      nz = self%nz;  nx = self%nx;  ny = self%ny

      ! spec S11.2: closure first, then every coefficient that depends on K
      if (self%is_tke) then
         call closure_step(self, u, v, b)
         call build_coefficients(self)
      end if

      ! -------------------------------------------------- shared predictor
      !$omp parallel default(shared) private(m, i, j, k) if(nx*ny >= omp_min_points)
      call pressure_gradients(self, b)
      call avg_v_to_u_m(self, v, self%d%mask3v, self%d%mask3u, self%avpv)
      call avg_u_to_v_m(self, u, self%d%mask3u, self%d%mask3v, self%avpu)
      call apply_diffusion_var(self, u, self%nu3, self%d%dz3u, self%tau_u, self%cfg%bottom_drag, self%dzu)
      call apply_diffusion_var(self, v, self%nu3, self%d%dz3v, self%tau_v, self%cfg%bottom_drag, self%dzv)
      call w_from_transport(self, u, v, self%wold)

      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               self%exu(i, j, k) = 0.0_wp
               self%exv(i, j, k) = 0.0_wp
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      if (self%is_up3) then
         call momentum_advection_up3(self, u, v, self%wold, self%exu, self%exv)
      else if (self%do_adv) then
         call momentum_advection(self, u, v, self%wold, self%exu, self%exv)
      end if
      if (self%v5%a_h > 0.0_wp) then
         call laplacian_h_m(self, u, self%d%mask3u, self%d%mask3v, self%d%mask3u, self%w3c)
         call laplacian_h_m(self, v, self%d%mask3u, self%d%mask3v, self%d%mask3v, self%w3d)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, nz
            do j = 1, ny
               do i = 1, nx
                  self%exu(i, j, k) = self%exu(i, j, k) + self%v5%a_h * self%w3c(i, j, k)
                  self%exv(i, j, k) = self%exv(i, j, k) + self%v5%a_h * self%w3d(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end if
      ! stress (theta_v-implicit surface part) + explicit extras, into w3c/w3d
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               self%w3c(i, j, k) = dt * self%exu(i, j, k)
               self%w3d(i, j, k) = dt * self%exv(i, j, k)
               if (k == 1) then
                  self%w3c(i, j, k) = self%w3c(i, j, k) + thv * dt * self%tau_u(i, j) * safe_inv(self%d%dz3u(i, j, 1))
                  self%w3d(i, j, k) = self%w3d(i, j, k) + thv * dt * self%tau_v(i, j) * safe_inv(self%d%dz3v(i, j, 1))
               end if
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel

      if (self%is_split) then
         call step_split(self, u, v, b, eta, t, s)
         return
      end if

      ! ------------------------------------------------- theta / fb branch
      !$omp parallel default(shared) private(m, i, j, k) if(nx*ny >= omp_min_points)
      call grad2_m(self, eta, self%gx2, self%gy2)
      call depth_transport(self, u, self%d%dz3u, self%uint)
      call depth_transport(self, v, self%d%dz3v, self%vint)
      call div2_m(self, self%uint, self%vint, self%divold)
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               self%uit(i, j, k) = u(i, j, k)
               self%vit(i, j, k) = v(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel

      do m = 1, self%cfg%n_picard
         !$omp parallel default(shared) private(i, j, k) if(nx*ny >= omp_min_points)
         call avg_v_to_u_m(self, self%vit, self%d%mask3v, self%d%mask3u, self%vcor)
         call avg_u_to_v_m(self, self%uit, self%d%mask3u, self%d%mask3v, self%ucor)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, nz
            do j = 1, ny
               do i = 1, nx
                  self%gu(i, j, k) = u(i, j, k) &
                     + dt * (f * ((1.0_wp - tc) * self%avpv(i, j, k) + tc * self%vcor(i, j, k)) + self%pgx(i, j, k)) &
                     - c1 * self%gx2(i, j) * self%d%mask3u(i, j, k) &
                     + (1.0_wp - thv) * dt * self%dzu(i, j, k) + self%w3c(i, j, k)
                  self%gv(i, j, k) = v(i, j, k) &
                     + dt * (-f * ((1.0_wp - tc) * self%avpu(i, j, k) + tc * self%ucor(i, j, k)) + self%pgy(i, j, k)) &
                     - c1 * self%gy2(i, j) * self%d%mask3v(i, j, k) &
                     + (1.0_wp - thv) * dt * self%dzv(i, j, k) + self%w3d(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         call thomas_batched(self%g, nz, self%msub, self%mdiag, self%msup, self%gu, self%ghu, self%cstar, self%dstar)
         call thomas_batched(self%g, nz, self%msub, self%mdiag, self%msup, self%gv, self%ghv, self%cstar, self%dstar)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, nz
            do j = 1, ny
               do i = 1, nx
                  self%ghu(i, j, k) = self%ghu(i, j, k) * self%d%mask3(i, j, k)
                  self%ghv(i, j, k) = self%ghv(i, j, k) * self%d%mask3(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         if (self%is_theta) then
            call depth_transport(self, self%ghu, self%d%dz3u, self%uint)
            call depth_transport(self, self%ghv, self%d%dz3v, self%vint)
            call div2_m(self, self%uint, self%vint, self%divg)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  self%rhs2(i, j) = eta(i, j) - dt * ((1.0_wp - th) * self%divold(i, j) + th * self%divg(i, j))
               end do
            end do
            !$acc end parallel loop
            !$omp end do
         end if
         !$omp end parallel

         if (self%is_theta) then
            call helm_var_solve(self%solver, self%rhs2, self%etanew)   ! own regions
            !$omp parallel default(shared) private(i, j, k) if(nx*ny >= omp_min_points)
            call grad2_m(self, self%etanew, self%gx2, self%gy2)
            !$omp do collapse(3) schedule(static)
            !$acc parallel loop collapse(3) gang vector
            do k = 1, nz
               do j = 1, ny
                  do i = 1, nx
                     self%uit(i, j, k) = (self%ghu(i, j, k) - c2 * self%q(i, j, k) * self%gx2(i, j) &
                                          * self%d%mask3u(i, j, k)) * self%d%mask3(i, j, k)
                     self%vit(i, j, k) = (self%ghv(i, j, k) - c2 * self%q(i, j, k) * self%gy2(i, j) &
                                          * self%d%mask3v(i, j, k)) * self%d%mask3(i, j, k)
                  end do
               end do
            end do
            !$acc end parallel loop
            !$omp end do
            if (m < self%cfg%n_picard) call grad2_m(self, eta, self%gx2, self%gy2)
            !$omp end parallel
         else
            !$omp parallel default(shared) private(i, j, k) if(nx*ny >= omp_min_points)
            !$omp do collapse(3) schedule(static)
            !$acc parallel loop collapse(3) gang vector
            do k = 1, nz
               do j = 1, ny
                  do i = 1, nx
                     self%uit(i, j, k) = self%ghu(i, j, k)
                     self%vit(i, j, k) = self%ghv(i, j, k)
                  end do
               end do
            end do
            !$acc end parallel loop
            !$omp end do
            !$omp end parallel
         end if
         self%tridiagonal_solves = self%tridiagonal_solves + 2
      end do

      !$omp parallel default(shared) private(i, j, k) if(nx*ny >= omp_min_points)
      if (.not. self%is_theta) then
         call depth_transport(self, self%uit, self%d%dz3u, self%uint)
         call depth_transport(self, self%vit, self%d%dz3v, self%vint)
         call div2_m(self, self%uint, self%vint, self%divg)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               self%etanew(i, j) = eta(i, j) - dt * self%divg(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end if
      call w_from_transport(self, self%uit, self%vit, self%wf)
      !$omp end parallel

      call advance_tracers(self, b, t, s, self%uit, self%vit, self%wf)

      !$omp parallel default(shared) private(i, j, k) if(nx*ny >= omp_min_points)
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               u(i, j, k) = self%uit(i, j, k)
               v(i, j, k) = self%vit(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, ny
         do i = 1, nx
            eta(i, j) = self%etanew(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel
   end subroutine stepper5_step

   ! Tracers: T/S with the equation of state, or buoyancy with the N^2 source.
   subroutine advance_tracers(self, b, t, s, u, v, wf)
      type(stepper5_t), intent(inout) :: self
      real(wp),         intent(inout) :: b(:,:,:), t(:,:,:), s(:,:,:)
      real(wp),         intent(in)    :: u(:,:,:), v(:,:,:), wf(:,:,:)
      integer :: i, j, k
      !$omp parallel default(shared) private(i, j, k) if(self%nx*self%ny >= omp_min_points)
      if (self%is_ts) then
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  self%w3a(i, j, k) = 0.0_wp
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         call advance_tracer(self, t, u, v, wf, self%flux_t, self%w3a, self%tnew)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  t(i, j, k) = self%tnew(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         call advance_tracer(self, s, u, v, wf, self%flux_s, self%w3a, self%tnew)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  s(i, j, k) = self%tnew(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         call buoyancy_eos(self, t, s, b)
      else
         ! source = -N2 * w at centres
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  self%w3a(i, j, k) = -self%cfg%n2 * 0.5_wp * (wf(i, j, k) + wf(i, j, k + 1))
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         call advance_tracer(self, b, u, v, wf, self%zero2, self%w3a, self%tnew)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, self%nz
            do j = 1, self%ny
               do i = 1, self%nx
                  b(i, j, k) = self%tnew(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end if
      !$omp end parallel
   end subroutine advance_tracers

   ! ============================================================ split-explicit
   subroutine step_split(self, u, v, b, eta, t, s)
      type(stepper5_t), intent(inout) :: self
      real(wp),         intent(inout) :: u(:,:,:), v(:,:,:), b(:,:,:), eta(:,:)
      real(wp),         intent(inout) :: t(:,:,:), s(:,:,:)
      real(wp) :: g_, f, dt, tc, thv, ddt
      integer  :: m, q, i, j, k, nz, nx, ny

      g_ = self%cfg%g;  f = self%cfg%f0;  dt = self%dt
      tc = self%cfg%theta_cor;  thv = self%cfg%theta_v
      nz = self%nz;  nx = self%nx;  ny = self%ny
      ddt = dt / real(self%n_split, wp)

      !$omp parallel default(shared) private(m, q, i, j, k) if(nx*ny >= omp_min_points)
      call depth_transport(self, u, self%d%dz3u, self%bu)      ! U0
      call depth_transport(self, v, self%d%dz3v, self%bv)      ! V0
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               self%uit(i, j, k) = u(i, j, k)
               self%vit(i, j, k) = v(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      do m = 1, self%cfg%n_picard
         call avg_v_to_u_m(self, self%vit, self%d%mask3v, self%d%mask3u, self%vcor)
         call avg_u_to_v_m(self, self%uit, self%d%mask3u, self%d%mask3v, self%ucor)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, nz
            do j = 1, ny
               do i = 1, nx
                  self%gu(i, j, k) = u(i, j, k) &
                     + dt * (f * ((1.0_wp - tc) * self%avpv(i, j, k) + tc * self%vcor(i, j, k)) + self%pgx(i, j, k)) &
                     + (1.0_wp - thv) * dt * self%dzu(i, j, k) + self%w3c(i, j, k)
                  self%gv(i, j, k) = v(i, j, k) &
                     + dt * (-f * ((1.0_wp - tc) * self%avpu(i, j, k) + tc * self%ucor(i, j, k)) + self%pgy(i, j, k)) &
                     + (1.0_wp - thv) * dt * self%dzv(i, j, k) + self%w3d(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         call thomas_batched(self%g, nz, self%msub, self%mdiag, self%msup, self%gu, self%ghu, self%cstar, self%dstar)
         call thomas_batched(self%g, nz, self%msub, self%mdiag, self%msup, self%gv, self%ghv, self%cstar, self%dstar)
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, nz
            do j = 1, ny
               do i = 1, nx
                  self%ghu(i, j, k) = self%ghu(i, j, k) * self%d%mask3(i, j, k)
                  self%ghv(i, j, k) = self%ghv(i, j, k) * self%d%mask3(i, j, k)
                  self%uit(i, j, k) = self%ghu(i, j, k)
                  self%vit(i, j, k) = self%ghv(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end do

      ! Coriolis removed from the barotropic forcing with the SAME 2D transport
      ! operator the substeps add back (spec S8.3 variable-depth form, N15).
      call depth_transport(self, self%ghu, self%d%dz3u, self%umean)
      call depth_transport(self, self%ghv, self%d%dz3v, self%vmean)
      call avg2_v_to_u(self, self%bv, self%bavv)        ! A_u[V0]
      call avg2_u_to_v(self, self%bu, self%bavu)        ! A_v[U0]
      call avg2_v_to_u(self, self%vmean, self%bvcor)    ! A_u[sum dz3v gh_v]
      call avg2_u_to_v(self, self%umean, self%bucor)    ! A_v[sum dz3u gh_u]
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, ny
         do i = 1, nx
            self%corx(i, j) =  f * ((1.0_wp - tc) * self%bavv(i, j) + tc * self%bvcor(i, j))
            self%cory(i, j) = -f * ((1.0_wp - tc) * self%bavu(i, j) + tc * self%bucor(i, j))
            self%bfx(i, j) = (self%umean(i, j) - self%bu(i, j)) / dt - self%corx(i, j)
            self%bfy(i, j) = (self%vmean(i, j) - self%bv(i, j)) / dt - self%cory(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      ! Barotropic substeps on the masked grid with the effective depth Ku/Kv.
      do m = 1, self%n_split
         call avg2_v_to_u(self, self%bv, self%bavv)
         call avg2_u_to_v(self, self%bu, self%bavu)
         call grad2_m(self, eta, self%gx2, self%gy2)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               self%bui(i, j) = self%bu(i, j)
               self%bvi(i, j) = self%bv(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         do q = 1, self%cfg%n_picard
            call avg2_v_to_u(self, self%bvi, self%bvcor)
            call avg2_u_to_v(self, self%bui, self%bucor)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  self%bui(i, j) = (self%bu(i, j) + ddt * ( &
                     f * ((1.0_wp - tc) * self%bavv(i, j) + tc * self%bvcor(i, j)) &
                     - g_ * self%ku(i, j) * self%gx2(i, j) + self%bfx(i, j))) * self%d%masku(i, j)
                  self%bvi(i, j) = (self%bv(i, j) + ddt * ( &
                     -f * ((1.0_wp - tc) * self%bavu(i, j) + tc * self%bucor(i, j)) &
                     - g_ * self%kv(i, j) * self%gy2(i, j) + self%bfy(i, j))) * self%d%maskv(i, j)
               end do
            end do
            !$acc end parallel loop
            !$omp end do
         end do
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               self%bu(i, j) = self%bui(i, j)
               self%bv(i, j) = self%bvi(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         call div2_m(self, self%bu, self%bv, self%divg)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               eta(i, j) = eta(i, j) - ddt * self%divg(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end do

      ! Replace the depth mean with the barotropic result (real thickness).
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               ! face masks, not the cell mask (N15)
               self%uit(i, j, k) = (self%ghu(i, j, k) - self%umean(i, j) * self%d%inv_hcu(i, j) &
                                    + self%bu(i, j) * self%d%inv_hcu(i, j)) * self%d%mask3u(i, j, k)
               self%vit(i, j, k) = (self%ghv(i, j, k) - self%vmean(i, j) * self%d%inv_hcv(i, j) &
                                    + self%bv(i, j) * self%d%inv_hcv(i, j)) * self%d%mask3v(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      call w_from_transport(self, self%uit, self%vit, self%wf)
      !$omp end parallel

      call advance_tracers(self, b, t, s, self%uit, self%vit, self%wf)

      !$omp parallel default(shared) private(i, j, k) if(nx*ny >= omp_min_points)
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               u(i, j, k) = self%uit(i, j, k)
               v(i, j, k) = self%vit(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel
      self%barotropic_substeps = self%barotropic_substeps + self%n_split
      self%tridiagonal_solves = self%tridiagonal_solves + 2 * self%cfg%n_picard
   end subroutine step_split

end module mod_model3d_v05
