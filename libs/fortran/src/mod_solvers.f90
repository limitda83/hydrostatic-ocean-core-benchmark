!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_solvers                                                  !
!  Description: Matrix-free preconditioned conjugate gradient for the   !
!               free-surface Helmholtz system                           !
!               ( I - g H theta^2 dt^2 L ) eta = rhs (spec S4.2).       !
!               Two global reductions per iteration - the scalability   !
!               bottleneck measured by RQ2.                             !
!                                                                       !
!  OpenMP: all worksharing here is ORPHANED, so the whole solve runs    !
!  inside the single parallel region that mod_schemes opens per         !
!  timestep. Two consequences drive the structure below:                !
!    1. Reduction targets must be SHARED. A routine's own locals are    !
!       private per thread inside a parallel region, so reducing into   !
!       them would silently give each thread its own partial sum. The   !
!       module-level accumulators red_a/red_b/red_c exist for this.     !
!    2. Scalars derived from a reduction (alpha, beta) are computed     !
!       redundantly into thread-private locals after the worksharing    !
!       barrier - identical values, no race, no extra barrier.          !
!  Pipeline: mod_operators -> mod_solvers -> mod_schemes                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_solvers
   use mod_kinds,     only: wp
   use mod_grid,      only: grid_t, omp_min_points
   use mod_operators, only: laplacian_eta
   implicit none
   private
   public :: pcg_solver_t

   ! Shared reduction accumulators (see note 1 in the header). Module scope
   ! makes them shared in the enclosing parallel region; they are never
   ! threadprivate. Only one solve is ever in flight at a time.
   ! NOTE: deliberately NOT '!$acc declare create'. An OpenACC reduction
   ! clause copies its result back to the host variable itself; giving the
   ! variable a resident device copy first makes the host keep reading a
   ! stale value. That silently produced norm_b = 0, an immediate return from
   ! the solver, and eta = 0 everywhere - the R2 gate caught it as a relative
   ! L2 error of exactly 1.0.
   real(wp), save :: red_a = 0.0_wp, red_b = 0.0_wp, red_c = 0.0_wp

   type :: pcg_solver_t
      real(wp) :: coef = 0.0_wp        ! g*H*theta^2*dt^2 in A = I - coef*L
      real(wp) :: inv_diag = 1.0_wp    ! Jacobi preconditioner
      real(wp) :: rtol = 1.0e-12_wp
      integer  :: max_iter = 2000
      integer  :: total_iterations = 0
      integer  :: failures = 0
      ! Persistent workspace: allocated once so the time loop never allocates.
      real(wp), allocatable :: r(:, :), z(:, :), p(:, :), ap(:, :)
      real(wp), allocatable :: wx(:, :), wy(:, :)
   contains
      procedure :: init  => pcg_init
      procedure :: solve => pcg_solve
   end type pcg_solver_t

contains

   subroutine pcg_init(self, g, coef, rtol, max_iter)
      class(pcg_solver_t), intent(inout) :: self
      type(grid_t),        intent(in)    :: g
      real(wp),            intent(in)    :: coef, rtol
      integer,             intent(in)    :: max_iter
      self%coef = coef
      self%rtol = rtol
      self%max_iter = max_iter
      self%inv_diag = 1.0_wp / (1.0_wp + coef * (2.0_wp / g%dx**2 + 2.0_wp / g%dy**2))
      allocate(self%r(g%nx, g%ny), self%z(g%nx, g%ny), self%p(g%nx, g%ny), &
               self%ap(g%nx, g%ny), self%wx(g%nx, g%ny), self%wy(g%nx, g%ny))
   end subroutine pcg_init

   subroutine apply(self, g, x, ax)
      class(pcg_solver_t), intent(inout) :: self
      type(grid_t),        intent(in)    :: g
      real(wp),            intent(in)    :: x(:, :)
      real(wp),            intent(inout) :: ax(:, :)
      integer :: i, j
      call laplacian_eta(g, x, ax, self%wx, self%wy)
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            ax(i, j) = x(i, j) - self%coef * ax(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine apply

   subroutine pcg_solve(self, g, rhs, x)
      class(pcg_solver_t), intent(inout) :: self
      type(grid_t),        intent(in)    :: g
      real(wp),            intent(in)    :: rhs(:, :)
      real(wp),            intent(inout) :: x(:, :)
      ! All of these are thread-private: locals of a routine called from
      ! inside a parallel region get one instance per thread.
      real(wp) :: norm_b, rz, rz_new, alpha, beta, residual
      integer  :: iteration, used, i, j
      logical  :: converged

      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            x(i, j) = 0.0_wp
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      call apply(self, g, x, self%ap)

      !$omp single
      red_a = 0.0_wp
      !$omp end single
      !$omp do collapse(2) schedule(static) reduction(+:red_a)
      !$acc parallel loop collapse(2) gang vector reduction(+:red_a)
      do j = 1, g%ny
         do i = 1, g%nx
            red_a = red_a + rhs(i, j) * rhs(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      norm_b = sqrt(red_a)
      ! Identical on every thread, so the early exit never diverges.
      if (norm_b == 0.0_wp) return

      !$omp single
      red_b = 0.0_wp
      !$omp end single
      !$omp do collapse(2) schedule(static) reduction(+:red_b)
      !$acc parallel loop collapse(2) gang vector reduction(+:red_b)
      do j = 1, g%ny
         do i = 1, g%nx
            self%r(i, j) = rhs(i, j) - self%ap(i, j)
            self%z(i, j) = self%inv_diag * self%r(i, j)
            self%p(i, j) = self%z(i, j)
            red_b = red_b + self%r(i, j) * self%z(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      rz = red_b

      converged = .false.
      used = self%max_iter
      residual = huge(1.0_wp)
      do iteration = 1, self%max_iter
         call apply(self, g, self%p, self%ap)

         !$omp single
         red_a = 0.0_wp
         !$omp end single
         !$omp do collapse(2) schedule(static) reduction(+:red_a)
         !$acc parallel loop collapse(2) gang vector reduction(+:red_a)
         do j = 1, g%ny
            do i = 1, g%nx
               red_a = red_a + self%p(i, j) * self%ap(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         alpha = rz / red_a

         !$omp single
         red_b = 0.0_wp
         !$omp end single
         !$omp do collapse(2) schedule(static) reduction(+:red_b)
         !$acc parallel loop collapse(2) gang vector reduction(+:red_b)
         do j = 1, g%ny
            do i = 1, g%nx
               x(i, j) = x(i, j) + alpha * self%p(i, j)
               self%r(i, j) = self%r(i, j) - alpha * self%ap(i, j)
               red_b = red_b + self%r(i, j) * self%r(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         residual = sqrt(red_b) / norm_b
         if (residual < self%rtol) then
            converged = .true.
            used = iteration
            exit
         end if

         !$omp single
         red_c = 0.0_wp
         !$omp end single
         !$omp do collapse(2) schedule(static) reduction(+:red_c)
         !$acc parallel loop collapse(2) gang vector reduction(+:red_c)
         do j = 1, g%ny
            do i = 1, g%nx
               self%z(i, j) = self%inv_diag * self%r(i, j)
               red_c = red_c + self%r(i, j) * self%z(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         rz_new = red_c
         beta = rz_new / rz

         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, g%ny
            do i = 1, g%nx
               self%p(i, j) = self%z(i, j) + beta * self%p(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         rz = rz_new
      end do

      ! Shared counters: exactly one thread updates them.
      !$omp single
      self%total_iterations = self%total_iterations + used
      if (.not. converged) then
         self%failures = self%failures + 1
         write(*, '(a,i0,a,es12.5,a,es12.5)') &
            'ERROR: PCG failed to converge: ', self%max_iter, &
            ' iterations, residual=', residual, ' > rtol=', self%rtol
      end if
      !$omp end single
   end subroutine pcg_solve

end module mod_solvers
