!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_helm_var                                                 !
!  Description: Variable-coefficient free-surface Helmholtz solver of   !
!               spec S10.5,                                             !
!                   eta - coef * D[ K G[eta] ] = rhs                    !
!               on a masked C-grid, with PCG-Jacobi, PCG-RBGS and a     !
!               geometric multigrid V-cycle. This is the solver of      !
!               libs/fortran/src/helmholtz_bench.f90 refactored into a  !
!               type so the v0.5 model can call it; the arithmetic is   !
!               unchanged, so the iteration counts verified there (R2)  !
!               carry over.                                             !
!                                                                       !
!  The solver opens its own OpenMP parallel regions. A caller inside a  !
!  parallel region must close it first (mod_model3d_v05 does).          !
!  Pipeline: mod_domain5 -> mod_helm_var -> mod_model3d_v05             !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_helm_var
   use mod_kinds, only: wp
   implicit none
   private
   public :: helm_var_t, helm_var_init, helm_var_solve, helm_var_free, helm_var_update

   integer, parameter :: MAXLVL = 14

   type :: level_t
      integer  :: nx = 0, ny = 0
      real(wp) :: dx = 0.0_wp, dy = 0.0_wp
      real(wp), allocatable :: ku(:, :), kv(:, :), msk(:, :), dinv(:, :)
      real(wp), allocatable :: x(:, :), b(:, :), r(:, :)
   end type level_t

   type :: helm_var_t
      integer  :: nx = 0, ny = 0, nlvl = 0
      real(wp) :: coef = 0.0_wp, rtol = 1.0e-12_wp
      integer  :: max_iter = 2000
      character(len=16) :: kind = 'pcg_jacobi'
      integer  :: total_iterations = 0, failures = 0
      type(level_t) :: lvl(MAXLVL)
      real(wp), allocatable :: r(:, :), z(:, :), p(:, :), ap(:, :), res(:, :)
   end type helm_var_t

contains

   ! ----------------------------------------------------------------- setup
   subroutine level_setup(lv, coef)
      type(level_t), intent(inout) :: lv
      real(wp),      intent(in)    :: coef
      integer  :: i, j, im, jm
      real(wp) :: cx, cy, d
      cx = coef / lv%dx**2
      cy = coef / lv%dy**2
      if (.not. allocated(lv%dinv)) allocate(lv%dinv(lv%nx, lv%ny))
      !$acc parallel loop collapse(2) gang vector private(im,jm,d)
      do j = 1, lv%ny
         do i = 1, lv%nx
            jm = merge(lv%ny, j - 1, j == 1)
            im = merge(lv%nx, i - 1, i == 1)
            d = 1.0_wp + cx * (lv%ku(i, j) + lv%ku(im, j)) &
                       + cy * (lv%kv(i, j) + lv%kv(i, jm))
            if (lv%msk(i, j) <= 0.0_wp) d = 1.0_wp
            lv%dinv(i, j) = 1.0_wp / d
         end do
      end do
      !$acc end parallel loop
   end subroutine level_setup

   subroutine helm_var_init(self, nx, ny, dx, dy, coef, ku, kv, msk, kind, rtol, max_iter)
      type(helm_var_t), intent(inout) :: self
      integer,          intent(in)    :: nx, ny, max_iter
      real(wp),         intent(in)    :: dx, dy, coef, rtol
      real(wp),         intent(in)    :: ku(nx, ny), kv(nx, ny), msk(nx, ny)
      character(len=*), intent(in)    :: kind
      integer :: l, i, j, cx, cy

      self%nx = nx;  self%ny = ny;  self%coef = coef
      self%rtol = rtol;  self%max_iter = max_iter;  self%kind = kind
      self%total_iterations = 0;  self%failures = 0

      self%lvl(1)%nx = nx;  self%lvl(1)%ny = ny
      self%lvl(1)%dx = dx;  self%lvl(1)%dy = dy
      if (allocated(self%lvl(1)%ku)) deallocate(self%lvl(1)%ku, self%lvl(1)%kv, self%lvl(1)%msk)
      allocate(self%lvl(1)%ku(nx, ny), self%lvl(1)%kv(nx, ny), self%lvl(1)%msk(nx, ny))
      self%lvl(1)%ku = ku;  self%lvl(1)%kv = kv;  self%lvl(1)%msk = msk
      call level_setup(self%lvl(1), coef)
      if (.not. allocated(self%lvl(1)%x)) &
         allocate(self%lvl(1)%x(nx, ny), self%lvl(1)%b(nx, ny), self%lvl(1)%r(nx, ny))
      self%nlvl = 1

      ! Coarse faces are the average of the two fine faces they replace and a
      ! coarse cell is wet if any child is (spec S10.5) - identical to the
      ! NumPy and CUDA hierarchies, which is why the V-cycle counts agree.
      if (trim(kind) == 'multigrid') then
         do l = 2, MAXLVL
            if (modulo(self%lvl(l-1)%nx, 2) /= 0 .or. modulo(self%lvl(l-1)%ny, 2) /= 0) exit
            if (self%lvl(l-1)%nx / 2 < 8 .or. self%lvl(l-1)%ny / 2 < 8) exit
            cx = self%lvl(l-1)%nx / 2;  cy = self%lvl(l-1)%ny / 2
            self%lvl(l)%nx = cx;  self%lvl(l)%ny = cy
            self%lvl(l)%dx = 2.0_wp * self%lvl(l-1)%dx
            self%lvl(l)%dy = 2.0_wp * self%lvl(l-1)%dy
            if (allocated(self%lvl(l)%ku)) deallocate(self%lvl(l)%ku, self%lvl(l)%kv, self%lvl(l)%msk)
            allocate(self%lvl(l)%ku(cx, cy), self%lvl(l)%kv(cx, cy), self%lvl(l)%msk(cx, cy))
            do j = 1, cy
               do i = 1, cx
                  self%lvl(l)%ku(i, j) = 0.5_wp * (self%lvl(l-1)%ku(2*i, 2*j-1) &
                                                 + self%lvl(l-1)%ku(2*i, 2*j))
                  self%lvl(l)%kv(i, j) = 0.5_wp * (self%lvl(l-1)%kv(2*i-1, 2*j) &
                                                 + self%lvl(l-1)%kv(2*i, 2*j))
                  self%lvl(l)%msk(i, j) = max(max(self%lvl(l-1)%msk(2*i-1, 2*j-1),  &
                                                  self%lvl(l-1)%msk(2*i, 2*j-1)),   &
                                              max(self%lvl(l-1)%msk(2*i-1, 2*j),    &
                                                  self%lvl(l-1)%msk(2*i, 2*j)))
               end do
            end do
            call level_setup(self%lvl(l), coef)
            if (.not. allocated(self%lvl(l)%x)) &
               allocate(self%lvl(l)%x(cx, cy), self%lvl(l)%b(cx, cy), self%lvl(l)%r(cx, cy))
            self%nlvl = l
         end do
      end if

      if (.not. allocated(self%r)) &
         allocate(self%r(nx, ny), self%z(nx, ny), self%p(nx, ny), self%ap(nx, ny), &
                  self%res(nx, ny))
      do l = 1, self%nlvl
         !$acc enter data copyin(self%lvl(l)%ku, self%lvl(l)%kv, self%lvl(l)%msk, self%lvl(l)%dinv)
         !$acc enter data create(self%lvl(l)%x, self%lvl(l)%b, self%lvl(l)%r)
      end do
      !$acc enter data create(self%r, self%z, self%p, self%ap, self%res)
   end subroutine helm_var_init

   ! Rebuild the face coefficients of every level for new Ku/Kv (masks and
   ! shapes unchanged): the per-step cost of a step-dependent vertical
   ! viscosity under the theta scheme (spec S11.5).
   subroutine helm_var_update(self, ku, kv)
      type(helm_var_t), intent(inout) :: self
      real(wp),         intent(in)    :: ku(:, :), kv(:, :)
      integer :: l, i, j
      ! Device-resident update (managed memory): copy the fine coefficients,
      ! coarsen level by level, refresh dinv - no host round trip per step.
      !$acc parallel loop collapse(2) gang vector
      do j = 1, self%lvl(1)%ny
         do i = 1, self%lvl(1)%nx
            self%lvl(1)%ku(i, j) = ku(i, j);  self%lvl(1)%kv(i, j) = kv(i, j)
         end do
      end do
      !$acc end parallel loop
      call level_setup(self%lvl(1), self%coef)
      do l = 2, self%nlvl
         !$acc parallel loop collapse(2) gang vector
         do j = 1, self%lvl(l)%ny
            do i = 1, self%lvl(l)%nx
               self%lvl(l)%ku(i, j) = 0.5_wp * (self%lvl(l-1)%ku(2*i, 2*j-1) + self%lvl(l-1)%ku(2*i, 2*j))
               self%lvl(l)%kv(i, j) = 0.5_wp * (self%lvl(l-1)%kv(2*i-1, 2*j) + self%lvl(l-1)%kv(2*i, 2*j))
            end do
         end do
         !$acc end parallel loop
         call level_setup(self%lvl(l), self%coef)
      end do
   end subroutine helm_var_update

   subroutine helm_var_free(self)
      type(helm_var_t), intent(inout) :: self
      integer :: l
      !$acc exit data delete(self%r, self%z, self%p, self%ap, self%res)
      do l = self%nlvl, 1, -1
         !$acc exit data delete(self%lvl(l)%x, self%lvl(l)%b, self%lvl(l)%r)
         !$acc exit data delete(self%lvl(l)%ku, self%lvl(l)%kv, self%lvl(l)%msk, self%lvl(l)%dinv)
      end do
   end subroutine helm_var_free

   ! --------------------------------------------------------------- kernels
   ! ax = x - coef * D[ K G[x] ], periodic index wrap, dry cells decoupled.
   subroutine helm_apply(lv, coef, x, ax)
      type(level_t), intent(in)    :: lv
      real(wp),      intent(in)    :: coef, x(:, :)
      real(wp),      intent(inout) :: ax(:, :)
      integer  :: i, j, ip, im, jp, jm
      real(wp) :: cx, cy, lap
      cx = coef / lv%dx**2
      cy = coef / lv%dy**2
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(ip,im,jp,jm,lap)
      do j = 1, lv%ny
         do i = 1, lv%nx
            ip = merge(1, i + 1, i == lv%nx)
            im = merge(lv%nx, i - 1, i == 1)
            jp = merge(1, j + 1, j == lv%ny)
            jm = merge(lv%ny, j - 1, j == 1)
            lap = cx * (lv%ku(i, j) * (x(ip, j) - x(i, j))                &
                      - lv%ku(im, j) * (x(i, j) - x(im, j)))              &
                + cy * (lv%kv(i, j) * (x(i, jp) - x(i, j))                &
                      - lv%kv(i, jm) * (x(i, j) - x(i, jm)))
            if (lv%msk(i, j) > 0.0_wp) then
               ax(i, j) = x(i, j) - lap
            else
               ax(i, j) = x(i, j)
            end if
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine helm_apply

   subroutine rbgs_colour(lv, coef, x, b, colour)
      type(level_t), intent(in)    :: lv
      real(wp),      intent(in)    :: coef, b(:, :)
      real(wp),      intent(inout) :: x(:, :)
      integer,       intent(in)    :: colour
      integer  :: i, j, ip, im, jp, jm
      real(wp) :: cx, cy, nb
      cx = coef / lv%dx**2
      cy = coef / lv%dy**2
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(ip,im,jp,jm,nb)
      do j = 1, lv%ny
         do i = 1, lv%nx
            if (modulo(i + j, 2) == colour .and. lv%msk(i, j) > 0.0_wp) then
               ip = merge(1, i + 1, i == lv%nx)
               im = merge(lv%nx, i - 1, i == 1)
               jp = merge(1, j + 1, j == lv%ny)
               jm = merge(lv%ny, j - 1, j == 1)
               nb = cx * (lv%ku(i, j) * x(ip, j) + lv%ku(im, j) * x(im, j)) &
                  + cy * (lv%kv(i, j) * x(i, jp) + lv%kv(i, jm) * x(i, jm))
               x(i, j) = lv%dinv(i, j) * (b(i, j) + nb)
            end if
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine rbgs_colour

   real(wp) function dot(a, b) result(s)
      real(wp), intent(in) :: a(:, :), b(:, :)
      integer  :: i, j
      ! Combined construct: called from serial context between the hoisted
      ! parallel regions, and an orphaned reduction into a routine local is
      ! invalid OpenMP (each thread would get its own partial sum).
      s = 0.0_wp
      !$omp parallel do collapse(2) schedule(static) reduction(+:s)
      !$acc parallel loop collapse(2) gang vector reduction(+:s)
      do j = 1, size(a, 2)
         do i = 1, size(a, 1)
            s = s + a(i, j) * b(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end parallel do
   end function dot

   ! ---------------------------------------------------------------- solve
   subroutine precondition(self, use_rbgs, r, z)
      type(helm_var_t), intent(inout) :: self
      logical,          intent(in)    :: use_rbgs
      real(wp),         intent(in)    :: r(:, :)
      real(wp),         intent(inout) :: z(:, :)
      integer :: i, j
      if (.not. use_rbgs) then
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, self%ny
            do i = 1, self%nx
               z(i, j) = self%lvl(1)%dinv(i, j) * r(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      else
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, self%ny
            do i = 1, self%nx
               z(i, j) = 0.0_wp
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         ! Symmetric sweep keeps M symmetric, which CG requires.
         call rbgs_colour(self%lvl(1), self%coef, z, r, 0)
         call rbgs_colour(self%lvl(1), self%coef, z, r, 1)
         call rbgs_colour(self%lvl(1), self%coef, z, r, 1)
         call rbgs_colour(self%lvl(1), self%coef, z, r, 0)
      end if
   end subroutine precondition

   subroutine pcg(self, use_rbgs, rhs, x, iters, converged)
      type(helm_var_t), intent(inout) :: self
      logical,          intent(in)    :: use_rbgs
      real(wp),         intent(in)    :: rhs(:, :)
      real(wp),         intent(inout) :: x(:, :)
      integer,          intent(out)   :: iters
      logical,          intent(out)   :: converged
      real(wp) :: norm_b, rz, rz_new, alpha, beta, tol, residual
      integer  :: it, i, j, nx, ny
      nx = self%nx;  ny = self%ny
      !$omp parallel default(shared)
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector present(self%r)
      do j = 1, ny
         do i = 1, nx
            x(i, j) = 0.0_wp
            self%r(i, j) = rhs(i, j) * self%lvl(1)%msk(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel
      norm_b = sqrt(dot(self%r, self%r))
      tol = self%rtol * norm_b
      iters = 0;  converged = .true.
      if (norm_b == 0.0_wp) return
      !$omp parallel default(shared)
      call precondition(self, use_rbgs, self%r, self%z)
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, ny
         do i = 1, nx
            self%p(i, j) = self%z(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel
      rz = dot(self%r, self%z)
      converged = .false.
      residual = norm_b
      do it = 1, self%max_iter
         !$omp parallel default(shared)
         call helm_apply(self%lvl(1), self%coef, self%p, self%ap)
         !$omp end parallel
         alpha = rz / dot(self%p, self%ap)
         !$omp parallel default(shared)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               x(i, j) = x(i, j) + alpha * self%p(i, j)
               self%r(i, j) = self%r(i, j) - alpha * self%ap(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         !$omp end parallel
         residual = sqrt(dot(self%r, self%r))
         iters = it
         if (residual <= tol) then
            converged = .true.
            exit
         end if
         !$omp parallel default(shared)
         call precondition(self, use_rbgs, self%r, self%z)
         !$omp end parallel
         rz_new = dot(self%r, self%z)
         beta = rz_new / rz
         !$omp parallel default(shared)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               self%p(i, j) = self%z(i, j) + beta * self%p(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         !$omp end parallel
         rz = rz_new
      end do
   end subroutine pcg

   recursive subroutine vcycle(self, l)
      type(helm_var_t), intent(inout) :: self
      integer,          intent(in)    :: l
      integer :: i, j, s
      do s = 1, 2
         call rbgs_colour(self%lvl(l), self%coef, self%lvl(l)%x, self%lvl(l)%b, 0)
         call rbgs_colour(self%lvl(l), self%coef, self%lvl(l)%x, self%lvl(l)%b, 1)
      end do
      if (l < self%nlvl) then
         call helm_apply(self%lvl(l), self%coef, self%lvl(l)%x, self%lvl(l)%r)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, self%lvl(l)%ny
            do i = 1, self%lvl(l)%nx
               self%lvl(l)%r(i, j) = (self%lvl(l)%b(i, j) - self%lvl(l)%r(i, j)) &
                                   * self%lvl(l)%msk(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, self%lvl(l+1)%ny
            do i = 1, self%lvl(l+1)%nx
               self%lvl(l+1)%b(i, j) = 0.25_wp * (self%lvl(l)%r(2*i-1, 2*j-1)  &
                                                + self%lvl(l)%r(2*i,   2*j-1)  &
                                                + self%lvl(l)%r(2*i-1, 2*j)    &
                                                + self%lvl(l)%r(2*i,   2*j))
               self%lvl(l+1)%x(i, j) = 0.0_wp
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         call vcycle(self, l + 1)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, self%lvl(l)%ny
            do i = 1, self%lvl(l)%nx
               self%lvl(l)%x(i, j) = self%lvl(l)%x(i, j)                          &
                  + self%lvl(l+1)%x((i + 1) / 2, (j + 1) / 2) * self%lvl(l)%msk(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end if
      do s = 1, 2
         call rbgs_colour(self%lvl(l), self%coef, self%lvl(l)%x, self%lvl(l)%b, 1)
         call rbgs_colour(self%lvl(l), self%coef, self%lvl(l)%x, self%lvl(l)%b, 0)
      end do
   end subroutine vcycle

   subroutine multigrid_solve(self, rhs, x, iters, converged)
      type(helm_var_t), intent(inout) :: self
      real(wp),         intent(in)    :: rhs(:, :)
      real(wp),         intent(inout) :: x(:, :)
      integer,          intent(out)   :: iters
      logical,          intent(out)   :: converged
      real(wp) :: norm_b, tol, residual
      integer  :: it, i, j, nx, ny
      nx = self%nx;  ny = self%ny
      !$omp parallel default(shared)
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, ny
         do i = 1, nx
            self%lvl(1)%x(i, j) = 0.0_wp
            self%lvl(1)%b(i, j) = rhs(i, j) * self%lvl(1)%msk(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel
      norm_b = sqrt(dot(self%lvl(1)%b, self%lvl(1)%b))
      tol = self%rtol * norm_b
      iters = 0;  converged = .true.
      if (norm_b == 0.0_wp) then
         x = 0.0_wp
         return
      end if
      converged = .false.
      do it = 1, self%max_iter
         !$omp parallel default(shared)
         call vcycle(self, 1)
         call helm_apply(self%lvl(1), self%coef, self%lvl(1)%x, self%res)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               self%res(i, j) = self%lvl(1)%b(i, j) - self%res(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         !$omp end parallel
         residual = sqrt(dot(self%res, self%res))
         iters = it
         if (residual <= tol) then
            converged = .true.
            exit
         end if
      end do
      !$omp parallel default(shared)
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, ny
         do i = 1, nx
            x(i, j) = self%lvl(1)%x(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel
   end subroutine multigrid_solve

   subroutine helm_var_solve(self, rhs, x)
      type(helm_var_t), intent(inout) :: self
      real(wp),         intent(in)    :: rhs(:, :)
      real(wp),         intent(inout) :: x(:, :)
      integer :: iters
      logical :: converged
      select case (trim(self%kind))
      case ('pcg_jacobi'); call pcg(self, .false., rhs, x, iters, converged)
      case ('pcg_rbgs');   call pcg(self, .true.,  rhs, x, iters, converged)
      case ('multigrid');  call multigrid_solve(self, rhs, x, iters, converged)
      case default
         write(*, '(a)') 'FATAL: unknown solver_kind '//trim(self%kind)// &
                         ' (v0.5 backend: pcg_jacobi|pcg_rbgs|multigrid)'
         error stop 1
      end select
      self%total_iterations = self%total_iterations + iters
      if (.not. converged) self%failures = self%failures + 1
   end subroutine helm_var_solve

end module mod_helm_var
