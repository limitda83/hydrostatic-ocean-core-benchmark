!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_model3d                                                  !
!  Description: 3D hydrostatic linear Boussinesq stepper following the  !
!               Casulli decomposition of docs/03_discretization_spec.md !
!               S7.5 - two batched tridiagonal solves, one 2D           !
!               free-surface Helmholtz, then back-substitution.         !
!               With nu = 0 and b = 0 it reduces exactly to the 2D      !
!               scheme, which is what verification case V3D-1 checks.   !
!  Pipeline: mod_vertical/mod_operators/mod_solvers -> mod_model3d      !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_model3d
   use mod_kinds,     only: wp
   use mod_grid,      only: grid_t, omp_min_points
   use mod_config,    only: run_config_t
   use mod_operators, only: gradx_u, grady_v, div_eta, avg_v_to_u, avg_u_to_v, &
                            gradx_u3, grady_v3, div_eta3, avg_v_to_u3, avg_u_to_v3
   use mod_solvers,   only: pcg_solver_t
   use mod_vertical,  only: thomas_batched, diffusion_coeffs, apply_diffusion, &
                            w_from_divergence, w_at_centres, &
                            buoyancy_potential, remove_depth_mean, depth_integral
   implicit none
   private
   public :: stepper3d_t, stepper3d_init, stepper3d_step, stepper3d_step_split

   type :: stepper3d_t
      type(run_config_t) :: cfg
      type(grid_t)       :: g
      type(pcg_solver_t) :: solver
      integer  :: nz = 0
      logical  :: is_theta = .true.
      logical  :: is_split = .false.
      integer  :: n_split = 0
      integer  :: barotropic_substeps = 0
      real(wp) :: dt = 0.0_wp, dz = 0.0_wp, h_eff = 0.0_wp
      integer  :: tridiagonal_solves = 0
      ! Tridiagonal coefficients (momentum and buoyancy) and the q profile.
      real(wp), allocatable :: nu(:,:,:), kap(:,:,:)
      real(wp), allocatable :: msub(:,:,:), mdiag(:,:,:), msup(:,:,:)
      real(wp), allocatable :: bsub(:,:,:), bdiag(:,:,:), bsup(:,:,:)
      real(wp), allocatable :: q(:,:,:)
      ! 3D workspace; allocated once so the time loop never allocates.
      real(wp), allocatable :: phi(:,:,:), pgx(:,:,:), pgy(:,:,:)
      real(wp), allocatable :: gx3(:,:,:), gy3(:,:,:)
      real(wp), allocatable :: avpv(:,:,:), avpu(:,:,:), vcor(:,:,:), ucor(:,:,:)
      real(wp), allocatable :: gu(:,:,:), gv(:,:,:), ghu(:,:,:), ghv(:,:,:)
      real(wp), allocatable :: dzu(:,:,:), dzv(:,:,:), dzb(:,:,:)
      real(wp), allocatable :: uit(:,:,:), vit(:,:,:), brhs(:,:,:)
      real(wp), allocatable :: div3(:,:,:), w(:,:,:), wc(:,:,:)
      real(wp), allocatable :: cstar(:,:,:), dstar(:,:,:), ones(:,:,:)
      ! 2D workspace.
      real(wp), allocatable :: gx2(:,:), gy2(:,:), work2(:,:)
      real(wp), allocatable :: uint(:,:), vint(:,:), divold(:,:), divg(:,:)
      real(wp), allocatable :: rhs2(:,:), etanew(:,:), tau_u(:,:), tau_v(:,:)
      ! Barotropic workspace for split_explicit (spec S8).
      real(wp), allocatable :: bu(:,:), bv(:,:), bfx(:,:), bfy(:,:)
      real(wp), allocatable :: bavv(:,:), bavu(:,:), bvcor(:,:), bucor(:,:)
      real(wp), allocatable :: bui(:,:), bvi(:,:), bgxe(:,:), bgye(:,:)
      real(wp), allocatable :: bacu(:,:), bacv(:,:), corx(:,:), cory(:,:)
      real(wp), allocatable :: umean(:,:), vmean(:,:)
      real(wp), allocatable :: zero2(:,:)   ! zero surface flux for buoyancy
   end type stepper3d_t

contains

   subroutine stepper3d_init(self, cfg, g, dt)
      type(stepper3d_t),  intent(out) :: self
      type(run_config_t), intent(in)  :: cfg
      type(grid_t),       intent(in)  :: g
      real(wp),           intent(in)  :: dt
      integer  :: nx, ny, nz, i, j, k
      real(wp) :: coef, hmin, hmax, acc, dt_baro

      self%cfg = cfg
      self%g = g
      self%dt = dt
      self%nz = cfg%nz
      self%dz = cfg%h0 / real(cfg%nz, wp)
      self%is_theta = (trim(cfg%scheme_name) == 'theta')
      self%is_split = (trim(cfg%scheme_name) == 'split_explicit')
      nx = g%nx;  ny = g%ny;  nz = cfg%nz

      allocate(self%nu(nx, ny, nz + 1), self%kap(nx, ny, nz + 1))
      allocate(self%msub(nx, ny, nz), self%mdiag(nx, ny, nz), self%msup(nx, ny, nz))
      allocate(self%bsub(nx, ny, nz), self%bdiag(nx, ny, nz), self%bsup(nx, ny, nz))
      allocate(self%q(nx, ny, nz), self%ones(nx, ny, nz))
      allocate(self%phi(nx, ny, nz), self%pgx(nx, ny, nz), self%pgy(nx, ny, nz))
      allocate(self%gx3(nx, ny, nz), self%gy3(nx, ny, nz))
      allocate(self%avpv(nx, ny, nz), self%avpu(nx, ny, nz))
      allocate(self%vcor(nx, ny, nz), self%ucor(nx, ny, nz))
      allocate(self%gu(nx, ny, nz), self%gv(nx, ny, nz))
      allocate(self%ghu(nx, ny, nz), self%ghv(nx, ny, nz))
      allocate(self%dzu(nx, ny, nz), self%dzv(nx, ny, nz), self%dzb(nx, ny, nz))
      allocate(self%uit(nx, ny, nz), self%vit(nx, ny, nz), self%brhs(nx, ny, nz))
      allocate(self%div3(nx, ny, nz), self%w(nx, ny, nz + 1), self%wc(nx, ny, nz))
      allocate(self%cstar(nx, ny, nz), self%dstar(nx, ny, nz))
      allocate(self%gx2(nx, ny), self%gy2(nx, ny), self%work2(nx, ny))
      allocate(self%uint(nx, ny), self%vint(nx, ny))
      allocate(self%divold(nx, ny), self%divg(nx, ny))
      allocate(self%rhs2(nx, ny), self%etanew(nx, ny))
      allocate(self%tau_u(nx, ny), self%tau_v(nx, ny), self%zero2(nx, ny))
      allocate(self%bu(nx, ny), self%bv(nx, ny), self%bfx(nx, ny), self%bfy(nx, ny))
      allocate(self%bavv(nx, ny), self%bavu(nx, ny), self%bvcor(nx, ny), self%bucor(nx, ny))
      allocate(self%bui(nx, ny), self%bvi(nx, ny), self%bgxe(nx, ny), self%bgye(nx, ny))
      allocate(self%bacu(nx, ny), self%bacv(nx, ny), self%corx(nx, ny), self%cory(nx, ny))
      allocate(self%umean(nx, ny), self%vmean(nx, ny))

      self%nu = cfg%nu
      self%kap = cfg%kappa
      self%ones = 1.0_wp
      self%tau_u = cfg%tau_x / cfg%rho0
      self%tau_v = cfg%tau_y / cfg%rho0
      self%zero2 = 0.0_wp

      !$omp parallel default(shared) if(nx*ny >= omp_min_points)
      call diffusion_coeffs(g, nz, self%nu, self%dz, dt, cfg%theta_v, &
                            cfg%bottom_drag, self%msub, self%mdiag, self%msup)
      call diffusion_coeffs(g, nz, self%kap, self%dz, dt, cfg%theta_v, &
                            0.0_wp, self%bsub, self%bdiag, self%bsup)
      call thomas_batched(g, nz, self%msub, self%mdiag, self%msup, self%ones, &
                          self%q, self%cstar, self%dstar)
      call depth_integral(g, nz, self%q, self%dz, self%work2)
      !$omp end parallel
      self%tridiagonal_solves = 1

      hmin = minval(self%work2)
      hmax = maxval(self%work2)
      if (hmax - hmin > 1.0e-10_wp * max(1.0_wp, abs(hmax))) then
         write(*, '(a,es12.5)') 'FATAL: spec v0.2 assumes a horizontally uniform '// &
            'effective depth (variable-coefficient Helmholtz is Phase 2b); spread=', &
            hmax - hmin
         error stop 1
      end if
      acc = 0.0_wp
      do j = 1, ny
         do i = 1, nx
            acc = acc + self%work2(i, j)
         end do
      end do
      self%h_eff = acc / real(nx * ny, wp)

      if (self%is_split) then
         ! Barotropic substeps must satisfy the surface-gravity CFL (spec S8.4);
         ! the baroclinic step is limited only by the internal wave speed.
         dt_baro = 2.0_wp / (sqrt(cfg%g * cfg%h0) &
                             * sqrt(4.0_wp / g%dx**2 + 4.0_wp / g%dy**2))
         if (cfg%n_split > 0) then
            self%n_split = cfg%n_split
         else
            self%n_split = max(1, ceiling(1.2_wp * dt / dt_baro))
         end if
         if (dt / real(self%n_split, wp) > dt_baro) then
            write(*, '(a,i0,a,es12.5,a,es12.5)') &
               'FATAL: n_split=', self%n_split, ' leaves a barotropic substep of ', &
               dt / real(self%n_split, wp), ' s above the stability limit ', dt_baro
            error stop 1
         end if
      end if

      if (self%is_theta) then
         if (trim(cfg%solver_kind) /= 'pcg_jacobi') then
            write(*, '(a)') 'FATAL: the compiled 3D backend implements '// &
                            'solver_kind=pcg_jacobi only; got '//trim(cfg%solver_kind)
            error stop 1
         end if
         coef = cfg%g * self%h_eff * cfg%theta**2 * dt**2
         call self%solver%init(g, coef, cfg%rtol, cfg%max_iter)
      end if
   end subroutine stepper3d_init

   subroutine barotropic_substeps(self, eta)
      ! Forward-backward substepping of the depth-integrated transports and the
      ! free surface (spec S8.3 step 4). Entirely local stencils: no elliptic
      ! solve and no global reduction, which is the whole point of the scheme.
      type(stepper3d_t), intent(inout) :: self
      real(wp),          intent(inout) :: eta(:,:)
      real(wp) :: g_, f, h_, ddt, tc
      integer  :: m, q, i, j, nx, ny

      g_ = self%cfg%g;  f = self%cfg%f0;  h_ = self%cfg%h0
      tc = self%cfg%theta_cor
      ddt = self%dt / real(self%n_split, wp)
      nx = self%g%nx;  ny = self%g%ny

      do m = 1, self%n_split
         call avg_v_to_u(self%g, self%bv, self%bavv)
         call avg_u_to_v(self%g, self%bu, self%bavu)
         call gradx_u(self%g, eta, self%bgxe)
         call grady_v(self%g, eta, self%bgye)
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

         ! theta_cor-weighted Picard Coriolis, as in the 2D scheme (spec S3.1).
         ! A sequential Coriolis here is first order and caps the whole scheme.
         do q = 1, self%cfg%n_picard
            call avg_v_to_u(self%g, self%bvi, self%bvcor)
            call avg_u_to_v(self%g, self%bui, self%bucor)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  self%bui(i, j) = self%bu(i, j) + ddt * ( &
                     f * ((1.0_wp - tc) * self%bavv(i, j) + tc * self%bvcor(i, j)) &
                     - g_ * h_ * self%bgxe(i, j) + self%bfx(i, j))
                  self%bvi(i, j) = self%bv(i, j) + ddt * ( &
                     -f * ((1.0_wp - tc) * self%bavu(i, j) + tc * self%bucor(i, j)) &
                     - g_ * h_ * self%bgye(i, j) + self%bfy(i, j))
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

         call div_eta(self%g, self%bu, self%bv, self%divg)
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
   end subroutine barotropic_substeps

   subroutine stepper3d_step_split(self, u, v, b, eta)
      type(stepper3d_t), intent(inout) :: self
      real(wp),          intent(inout) :: u(:,:,:), v(:,:,:), b(:,:,:), eta(:,:)
      real(wp) :: g_, f, dt, tc, thv, dz, n2, inv_nz, h_
      integer  :: m, i, j, k, nz, nx, ny

      g_ = self%cfg%g;  f = self%cfg%f0;  dt = self%dt
      tc = self%cfg%theta_cor;  thv = self%cfg%theta_v
      n2 = self%cfg%n2;  dz = self%dz;  h_ = self%cfg%h0
      nz = self%nz;  nx = self%g%nx;  ny = self%g%ny
      inv_nz = 1.0_wp / real(nz, wp)

      !$omp parallel default(shared) private(m, i, j, k) if(nx*ny >= omp_min_points)

      call buoyancy_potential(self%g, nz, b, dz, self%phi)
      call remove_depth_mean(self%g, nz, self%phi, self%work2)
      call gradx_u3(self%g, nz, self%phi, self%pgx)
      call grady_v3(self%g, nz, self%phi, self%pgy)
      call avg_v_to_u3(self%g, nz, v, self%avpv)
      call avg_u_to_v3(self%g, nz, u, self%avpu)
      call depth_integral(self%g, nz, u, dz, self%uint)
      call depth_integral(self%g, nz, v, dz, self%vint)
      call apply_diffusion(self%g, nz, u, self%nu, dz, self%tau_u, &
                           self%cfg%bottom_drag, self%dzu)
      call apply_diffusion(self%g, nz, v, self%nu, dz, self%tau_v, &
                           self%cfg%bottom_drag, self%dzv)

      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, ny
         do i = 1, nx
            self%bu(i, j) = self%uint(i, j)
            self%bv(i, j) = self%vint(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do

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
         call avg_v_to_u3(self%g, nz, self%vit, self%vcor)
         call avg_u_to_v3(self%g, nz, self%uit, self%ucor)
         ! Baroclinic predictor WITHOUT the barotropic pressure gradient.
         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, nz
            do j = 1, ny
               do i = 1, nx
                  self%gu(i, j, k) = u(i, j, k) + dt * ( &
                     f * ((1.0_wp - tc) * self%avpv(i, j, k) + tc * self%vcor(i, j, k)) &
                     + self%pgx(i, j, k)) + (1.0_wp - thv) * dt * self%dzu(i, j, k)
                  self%gv(i, j, k) = v(i, j, k) + dt * ( &
                     -f * ((1.0_wp - tc) * self%avpu(i, j, k) + tc * self%ucor(i, j, k)) &
                     + self%pgy(i, j, k)) + (1.0_wp - thv) * dt * self%dzv(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               self%gu(i, j, 1) = self%gu(i, j, 1) + thv * dt * self%tau_u(i, j) / dz
               self%gv(i, j, 1) = self%gv(i, j, 1) + thv * dt * self%tau_v(i, j) / dz
            end do
         end do
         !$acc end parallel loop
         !$omp end do

         call thomas_batched(self%g, nz, self%msub, self%mdiag, self%msup, &
                             self%gu, self%ghu, self%cstar, self%dstar)
         call thomas_batched(self%g, nz, self%msub, self%mdiag, self%msup, &
                             self%gv, self%ghv, self%cstar, self%dstar)

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
      end do

      ! Depth-integrated Coriolis actually applied above, so it can be removed
      ! from the barotropic forcing (spec S8.2 - otherwise it is counted twice).
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(k)
      do j = 1, ny
         do i = 1, nx
            self%corx(i, j) = 0.0_wp
            self%cory(i, j) = 0.0_wp
            do k = 1, nz
               self%corx(i, j) = self%corx(i, j) + f * ((1.0_wp - tc) * self%avpv(i, j, k) &
                                                        + tc * self%vcor(i, j, k))
               self%cory(i, j) = self%cory(i, j) - f * ((1.0_wp - tc) * self%avpu(i, j, k) &
                                                        + tc * self%ucor(i, j, k))
            end do
            self%corx(i, j) = dz * self%corx(i, j)
            self%cory(i, j) = dz * self%cory(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      call depth_integral(self%g, nz, self%ghu, dz, self%umean)
      call depth_integral(self%g, nz, self%ghv, dz, self%vmean)
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, ny
         do i = 1, nx
            self%bfx(i, j) = (self%umean(i, j) - self%bu(i, j)) / dt - self%corx(i, j)
            self%bfy(i, j) = (self%vmean(i, j) - self%bv(i, j)) / dt - self%cory(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      call barotropic_substeps(self, eta)

      ! Replace the depth mean of the baroclinic solution with the barotropic
      ! result, so that the depth integral matches exactly (spec S8.3 step 5).
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(k)
      do j = 1, ny
         do i = 1, nx
            self%umean(i, j) = 0.0_wp
            self%vmean(i, j) = 0.0_wp
            do k = 1, nz
               self%umean(i, j) = self%umean(i, j) + self%ghu(i, j, k)
               self%vmean(i, j) = self%vmean(i, j) + self%ghv(i, j, k)
            end do
            self%umean(i, j) = self%umean(i, j) * inv_nz
            self%vmean(i, j) = self%vmean(i, j) * inv_nz
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               self%uit(i, j, k) = self%ghu(i, j, k) - self%umean(i, j) &
                                 + self%bu(i, j) / h_
               self%vit(i, j, k) = self%ghv(i, j, k) - self%vmean(i, j) &
                                 + self%bv(i, j) / h_
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      call div_eta3(self%g, nz, self%uit, self%vit, self%div3)
      call w_from_divergence(self%g, nz, self%div3, dz, self%w)
      call w_at_centres(self%g, nz, self%w, self%wc)
      call apply_diffusion(self%g, nz, b, self%kap, dz, self%zero2, 0.0_wp, self%dzb)
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               self%brhs(i, j, k) = b(i, j, k) - dt * n2 * self%wc(i, j, k) &
                                  + (1.0_wp - thv) * dt * self%dzb(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      call thomas_batched(self%g, nz, self%bsub, self%bdiag, self%bsup, &
                          self%brhs, b, self%cstar, self%dstar)

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
      self%tridiagonal_solves = self%tridiagonal_solves + 2 * self%cfg%n_picard + 1
   end subroutine stepper3d_step_split

   subroutine stepper3d_step(self, u, v, b, eta)
      type(stepper3d_t), intent(inout) :: self
      real(wp),          intent(inout) :: u(:,:,:), v(:,:,:), b(:,:,:), eta(:,:)
      real(wp) :: g_, f, dt, th, tc, thv, dz, c1, c2, n2
      integer  :: m, i, j, k, nz, nx, ny

      g_ = self%cfg%g;  f = self%cfg%f0;  dt = self%dt
      th = self%cfg%theta;  tc = self%cfg%theta_cor;  thv = self%cfg%theta_v
      if (.not. self%is_theta) th = 0.0_wp     ! fb: full explicit gradient (N14)
      n2 = self%cfg%n2;  dz = self%dz
      c1 = g_ * dt * (1.0_wp - th)
      c2 = g_ * dt * th
      nz = self%nz;  nx = self%g%nx;  ny = self%g%ny

      if (self%is_split) then
         call stepper3d_step_split(self, u, v, b, eta)
         return
      end if

      !$omp parallel default(shared) private(m, i, j, k) if(nx*ny >= omp_min_points)

      ! Baroclinic pressure gradient from the depth-mean-free potential.
      call buoyancy_potential(self%g, nz, b, dz, self%phi)
      call remove_depth_mean(self%g, nz, self%phi, self%work2)
      call gradx_u3(self%g, nz, self%phi, self%pgx)
      call grady_v3(self%g, nz, self%phi, self%pgy)

      call gradx_u(self%g, eta, self%gx2)
      call grady_v(self%g, eta, self%gy2)
      call avg_v_to_u3(self%g, nz, v, self%avpv)
      call avg_u_to_v3(self%g, nz, u, self%avpu)
      call depth_integral(self%g, nz, u, dz, self%uint)
      call depth_integral(self%g, nz, v, dz, self%vint)
      call div_eta(self%g, self%uint, self%vint, self%divold)

      call apply_diffusion(self%g, nz, u, self%nu, dz, self%tau_u, &
                           self%cfg%bottom_drag, self%dzu)
      call apply_diffusion(self%g, nz, v, self%nu, dz, self%tau_v, &
                           self%cfg%bottom_drag, self%dzv)

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
         call avg_v_to_u3(self%g, nz, self%vit, self%vcor)
         call avg_u_to_v3(self%g, nz, self%uit, self%ucor)

         !$omp do collapse(3) schedule(static)
         !$acc parallel loop collapse(3) gang vector
         do k = 1, nz
            do j = 1, ny
               do i = 1, nx
                  self%gu(i, j, k) = u(i, j, k) &
                     + dt * (f * ((1.0_wp - tc) * self%avpv(i, j, k) &
                                  + tc * self%vcor(i, j, k)) + self%pgx(i, j, k)) &
                     - c1 * self%gx2(i, j) &
                     + (1.0_wp - thv) * dt * self%dzu(i, j, k)
                  self%gv(i, j, k) = v(i, j, k) &
                     + dt * (-f * ((1.0_wp - tc) * self%avpu(i, j, k) &
                                   + tc * self%ucor(i, j, k)) + self%pgy(i, j, k)) &
                     - c1 * self%gy2(i, j) &
                     + (1.0_wp - thv) * dt * self%dzv(i, j, k)
               end do
            end do
         end do
         !$acc end parallel loop
         !$omp end do

         ! theta_v-implicit part of the surface stress.
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               self%gu(i, j, 1) = self%gu(i, j, 1) + thv * dt * self%tau_u(i, j) / dz
               self%gv(i, j, 1) = self%gv(i, j, 1) + thv * dt * self%tau_v(i, j) / dz
            end do
         end do
         !$acc end parallel loop
         !$omp end do

         call thomas_batched(self%g, nz, self%msub, self%mdiag, self%msup, &
                             self%gu, self%ghu, self%cstar, self%dstar)
         call thomas_batched(self%g, nz, self%msub, self%mdiag, self%msup, &
                             self%gv, self%ghv, self%cstar, self%dstar)

         if (self%is_theta) then
            call depth_integral(self%g, nz, self%ghu, dz, self%uint)
            call depth_integral(self%g, nz, self%ghv, dz, self%vint)
            call div_eta(self%g, self%uint, self%vint, self%divg)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  self%rhs2(i, j) = eta(i, j) - dt * ((1.0_wp - th) * self%divold(i, j) &
                                                      + th * self%divg(i, j))
               end do
            end do
            !$acc end parallel loop
            !$omp end do

            call self%solver%solve(self%g, self%rhs2, self%etanew)
            call gradx_u(self%g, self%etanew, self%gx2)
            call grady_v(self%g, self%etanew, self%gy2)

            !$omp do collapse(3) schedule(static)
            !$acc parallel loop collapse(3) gang vector
            do k = 1, nz
               do j = 1, ny
                  do i = 1, nx
                     self%uit(i, j, k) = self%ghu(i, j, k) &
                        - c2 * self%q(i, j, k) * self%gx2(i, j)
                     self%vit(i, j, k) = self%ghv(i, j, k) &
                        - c2 * self%q(i, j, k) * self%gy2(i, j)
                  end do
               end do
            end do
            !$acc end parallel loop
            !$omp end do
            ! Restore grad(eta^n) for the next Picard pass.
            if (m < self%cfg%n_picard) then
               call gradx_u(self%g, eta, self%gx2)
               call grady_v(self%g, eta, self%gy2)
            end if
         else
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
         end if
      end do

      if (.not. self%is_theta) then
         call depth_integral(self%g, nz, self%uit, dz, self%uint)
         call depth_integral(self%g, nz, self%vit, dz, self%vint)
         call div_eta(self%g, self%uint, self%vint, self%divg)
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

      ! w from the updated flow, then buoyancy stepped forward-backward.
      call div_eta3(self%g, nz, self%uit, self%vit, self%div3)
      call w_from_divergence(self%g, nz, self%div3, dz, self%w)
      call w_at_centres(self%g, nz, self%w, self%wc)
      call apply_diffusion(self%g, nz, b, self%kap, dz, self%zero2, 0.0_wp, self%dzb)

      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               self%brhs(i, j, k) = b(i, j, k) - dt * n2 * self%wc(i, j, k) &
                                  + (1.0_wp - thv) * dt * self%dzb(i, j, k)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      call thomas_batched(self%g, nz, self%bsub, self%bdiag, self%bsup, &
                          self%brhs, b, self%cstar, self%dstar)

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
      self%tridiagonal_solves = self%tridiagonal_solves + 2 * self%cfg%n_picard + 1
   end subroutine stepper3d_step

end module mod_model3d
