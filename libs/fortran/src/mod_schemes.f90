!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_schemes                                                  !
!  Description: Forward-backward and Casulli-type theta time            !
!               integration (docs/03_discretization_spec.md S3), with   !
!               the theta_cor-weighted Picard Coriolis treatment.       !
!               Term order follows S6.1 so the result matches           !
!               libs/core/schemes.py to round-off.                      !
!                                                                       !
!  OpenMP: exactly ONE parallel region per timestep. Everything called  !
!  from inside it (operators, PCG) uses orphaned worksharing, so the    !
!  team is created once per step instead of ~20 times. The earlier      !
!  per-loop structure created ~54,000 teams per benchmark sweep and     !
!  collapsed past 16 threads on a 192-core node (docs/11 section 4).    !
!  Pipeline: mod_operators/mod_solvers -> mod_schemes -> program        !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_schemes
   use mod_kinds,     only: wp
   use mod_grid,      only: grid_t, omp_min_points
   use mod_config,    only: run_config_t
   use mod_operators, only: gradx_u, grady_v, div_eta, avg_v_to_u, avg_u_to_v
   use mod_solvers,   only: pcg_solver_t
   implicit none
   private
   public :: stepper_t, stepper_init, stepper_step, explicit_dt_max

   type :: stepper_t
      type(run_config_t) :: cfg
      type(grid_t)       :: g
      type(pcg_solver_t) :: solver
      logical  :: is_theta = .true.
      real(wp) :: dt = 0.0_wp
      ! Workspace, allocated once - the time loop must never allocate.
      real(wp), allocatable :: gx_eta(:, :), gy_eta(:, :), div_old(:, :)
      real(wp), allocatable :: av_prev_v(:, :), av_prev_u(:, :)
      real(wp), allocatable :: v_cor(:, :), u_cor(:, :)
      real(wp), allocatable :: gu(:, :), gv(:, :), rhs(:, :), tmp(:, :)
      real(wp), allocatable :: u_it(:, :), v_it(:, :)
   end type stepper_t

contains

   pure function explicit_dt_max(g, cfg) result(dt_max)
      ! Forward-backward stability limit (spec S3.2); used as the common
      ! yardstick so cfl_factor is comparable across schemes.
      type(grid_t),       intent(in) :: g
      type(run_config_t), intent(in) :: cfg
      real(wp) :: dt_max, c, s_max, dt_gravity, dt_coriolis
      c = sqrt(cfg%g * cfg%h0)
      s_max = sqrt(4.0_wp / g%dx**2 + 4.0_wp / g%dy**2)
      dt_gravity = 2.0_wp / (c * s_max)
      if (cfg%f0 /= 0.0_wp) then
         dt_coriolis = 2.0_wp / abs(cfg%f0)
      else
         dt_coriolis = huge(1.0_wp)
      end if
      dt_max = min(dt_gravity, dt_coriolis)
   end function explicit_dt_max

   subroutine stepper_init(self, cfg, g, dt)
      type(stepper_t),    intent(out) :: self
      type(run_config_t), intent(in)  :: cfg
      type(grid_t),       intent(in)  :: g
      real(wp),           intent(in)  :: dt
      real(wp) :: coef
      integer  :: nx, ny

      self%cfg = cfg
      self%g = g
      self%dt = dt
      self%is_theta = (trim(cfg%scheme_name) == 'theta')
      nx = g%nx;  ny = g%ny

      allocate(self%gx_eta(nx, ny), self%gy_eta(nx, ny), self%div_old(nx, ny), &
               self%av_prev_v(nx, ny), self%av_prev_u(nx, ny), &
               self%v_cor(nx, ny), self%u_cor(nx, ny), &
               self%gu(nx, ny), self%gv(nx, ny), self%rhs(nx, ny), &
               self%tmp(nx, ny), self%u_it(nx, ny), self%v_it(nx, ny))

      if (self%is_theta) then
         if (trim(cfg%solver_kind) /= 'pcg_jacobi') then
            write(*, '(a)') 'FATAL: the Fortran backend implements solver_kind='// &
                            'pcg_jacobi only (fft is reference-only); got '// &
                            trim(cfg%solver_kind)
            error stop 1
         end if
         coef = cfg%g * cfg%h0 * cfg%theta**2 * dt**2
         call self%solver%init(g, coef, cfg%rtol, cfg%max_iter)
      end if
   end subroutine stepper_init

   subroutine stepper_step(self, u, v, eta)
      type(stepper_t), intent(inout) :: self
      real(wp),        intent(inout) :: u(:, :), v(:, :), eta(:, :)
      real(wp) :: g_, f, h_, dt, th, tc, c1, c2
      integer  :: m, i, j, nx, ny

      g_ = self%cfg%g;  f = self%cfg%f0;  h_ = self%cfg%h0
      dt = self%dt;     th = self%cfg%theta;  tc = self%cfg%theta_cor
      c1 = g_ * dt * (1.0_wp - th)
      c2 = g_ * dt * th
      nx = self%g%nx;  ny = self%g%ny

      !$omp parallel default(shared) private(m, i, j) &
      !$omp          if(nx*ny >= omp_min_points)

      call gradx_u(self%g, eta, self%gx_eta)
      call grady_v(self%g, eta, self%gy_eta)
      ! Hoisted out of the Picard loop: these depend only on the level-n
      ! state, so the arithmetic per element is identical to the reference.
      call avg_v_to_u(self%g, v, self%av_prev_v)
      call avg_u_to_v(self%g, u, self%av_prev_u)

      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, ny
         do i = 1, nx
            self%u_it(i, j) = u(i, j)
            self%v_it(i, j) = v(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      if (self%is_theta) then
         call div_eta(self%g, u, v, self%div_old)

         do m = 1, self%cfg%n_picard
            call avg_v_to_u(self%g, self%v_it, self%v_cor)
            call avg_u_to_v(self%g, self%u_it, self%u_cor)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  self%gu(i, j) = u(i, j) + dt * f * ((1.0_wp - tc) * self%av_prev_v(i, j) &
                                                      + tc * self%v_cor(i, j)) &
                                  - c1 * self%gx_eta(i, j)
                  self%gv(i, j) = v(i, j) - dt * f * ((1.0_wp - tc) * self%av_prev_u(i, j) &
                                                      + tc * self%u_cor(i, j)) &
                                  - c1 * self%gy_eta(i, j)
               end do
            end do
            !$acc end parallel loop
            !$omp end do

            call div_eta(self%g, self%gu, self%gv, self%tmp)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  self%rhs(i, j) = eta(i, j) - dt * h_ * &
                     ((1.0_wp - th) * self%div_old(i, j) + th * self%tmp(i, j))
               end do
            end do
            !$acc end parallel loop
            !$omp end do

            call self%solver%solve(self%g, self%rhs, self%tmp)   ! tmp <- eta^{n+1}
            call gradx_u(self%g, self%tmp, self%gx_eta)          ! reuse as scratch
            call grady_v(self%g, self%tmp, self%gy_eta)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  self%u_it(i, j) = self%gu(i, j) - c2 * self%gx_eta(i, j)
                  self%v_it(i, j) = self%gv(i, j) - c2 * self%gy_eta(i, j)
               end do
            end do
            !$acc end parallel loop
            !$omp end do
            ! gx_eta/gy_eta must return to grad(eta^n) for the next Picard pass.
            if (m < self%cfg%n_picard) then
               call gradx_u(self%g, eta, self%gx_eta)
               call grady_v(self%g, eta, self%gy_eta)
            end if
         end do

         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               u(i, j) = self%u_it(i, j)
               v(i, j) = self%v_it(i, j)
               eta(i, j) = self%tmp(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do

      else   ! forward-backward
         do m = 1, self%cfg%n_picard
            call avg_v_to_u(self%g, self%v_it, self%v_cor)
            call avg_u_to_v(self%g, self%u_it, self%u_cor)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  self%gu(i, j) = u(i, j) + dt * (f * ((1.0_wp - tc) * self%av_prev_v(i, j) &
                                     + tc * self%v_cor(i, j)) - g_ * self%gx_eta(i, j))
                  self%gv(i, j) = v(i, j) + dt * (-f * ((1.0_wp - tc) * self%av_prev_u(i, j) &
                                     + tc * self%u_cor(i, j)) - g_ * self%gy_eta(i, j))
               end do
            end do
            !$acc end parallel loop
            !$omp end do
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  self%u_it(i, j) = self%gu(i, j)
                  self%v_it(i, j) = self%gv(i, j)
               end do
            end do
            !$acc end parallel loop
            !$omp end do
         end do

         call div_eta(self%g, self%u_it, self%v_it, self%tmp)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               eta(i, j) = eta(i, j) - dt * h_ * self%tmp(i, j)
               u(i, j) = self%u_it(i, j)
               v(i, j) = self%v_it(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end if

      !$omp end parallel
   end subroutine stepper_step

end module mod_schemes
