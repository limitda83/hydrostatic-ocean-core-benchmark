!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_operators                                                !
!  Description: Discrete C-grid operators of docs/03_discretization_    !
!               spec.md S2.1, byte-for-byte equivalent to               !
!               libs/core/operators.py. Term order is fixed by S6.1.    !
!                                                                       !
!  IMPORTANT: the worksharing directives here are ORPHANED (!$omp do,   !
!  not !$omp parallel do). Callers are expected to already be inside a  !
!  parallel region - mod_schemes opens exactly one per timestep. Called !
!  from serial code they simply execute serially, which is valid        !
!  OpenMP, so nothing else has to change.                               !
!  Pipeline: mod_grid -> mod_operators -> mod_schemes / mod_solvers     !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_operators
   use mod_kinds, only: wp
   use mod_grid,  only: grid_t, omp_min_points
   implicit none
   private
   public :: gradx_u, grady_v, div_eta, avg_v_to_u, avg_u_to_v, laplacian_eta
   public :: gradx_u3, grady_v3, div_eta3, avg_v_to_u3, avg_u_to_v3

contains

   subroutine gradx_u(g, eta, out)
      type(grid_t), intent(in)  :: g
      real(wp),     intent(in)  :: eta(:, :)
      real(wp),     intent(out) :: out(:, :)
      integer :: i, j
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            out(i, j) = (eta(g%ip(i), j) - eta(i, j)) / g%dx
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine gradx_u

   subroutine grady_v(g, eta, out)
      type(grid_t), intent(in)  :: g
      real(wp),     intent(in)  :: eta(:, :)
      real(wp),     intent(out) :: out(:, :)
      integer :: i, j
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            out(i, j) = (eta(i, g%jp(j)) - eta(i, j)) / g%dy
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine grady_v

   subroutine div_eta(g, u, v, out)
      type(grid_t), intent(in)  :: g
      real(wp),     intent(in)  :: u(:, :), v(:, :)
      real(wp),     intent(out) :: out(:, :)
      integer :: i, j
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            out(i, j) = (u(i, j) - u(g%im(i), j)) / g%dx &
                      + (v(i, j) - v(i, g%jm(j))) / g%dy
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine div_eta

   subroutine avg_v_to_u(g, v, out)
      type(grid_t), intent(in)  :: g
      real(wp),     intent(in)  :: v(:, :)
      real(wp),     intent(out) :: out(:, :)
      integer :: i, j
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            out(i, j) = 0.25_wp * (v(i, j) + v(i, g%jm(j)) &
                                 + v(g%ip(i), j) + v(g%ip(i), g%jm(j)))
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine avg_v_to_u

   subroutine avg_u_to_v(g, u, out)
      type(grid_t), intent(in)  :: g
      real(wp),     intent(in)  :: u(:, :)
      real(wp),     intent(out) :: out(:, :)
      integer :: i, j
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            out(i, j) = 0.25_wp * (u(i, j) + u(g%im(i), j) &
                                 + u(i, g%jp(j)) + u(g%im(i), g%jp(j)))
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine avg_u_to_v

   subroutine laplacian_eta(g, eta, out, wx, wy)
      ! L = D o G, never a separate 5-point stencil (spec S2.1), so that
      ! (I - c L) stays symmetric positive definite.
      type(grid_t), intent(in)    :: g
      real(wp),     intent(in)    :: eta(:, :)
      real(wp),     intent(out)   :: out(:, :)
      real(wp),     intent(inout) :: wx(:, :), wy(:, :)   ! scratch, u/v points
      call gradx_u(g, eta, wx)
      call grady_v(g, eta, wy)
      call div_eta(g, wx, wy, out)
   end subroutine laplacian_eta

   ! ------------------------------------------------------------------ 3D
   ! Same stencils applied level by level, but with collapse(3) so one
   ! worksharing region covers the whole 3D field instead of nz of them.

   subroutine gradx_u3(g, nz, eta, out)
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: eta(:, :, :)
      real(wp),     intent(out) :: out(:, :, :)
      integer :: i, j, k
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, g%ny
            do i = 1, g%nx
               out(i, j, k) = (eta(g%ip(i), j, k) - eta(i, j, k)) / g%dx
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine gradx_u3

   subroutine grady_v3(g, nz, eta, out)
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: eta(:, :, :)
      real(wp),     intent(out) :: out(:, :, :)
      integer :: i, j, k
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, g%ny
            do i = 1, g%nx
               out(i, j, k) = (eta(i, g%jp(j), k) - eta(i, j, k)) / g%dy
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine grady_v3

   subroutine div_eta3(g, nz, u, v, out)
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: u(:, :, :), v(:, :, :)
      real(wp),     intent(out) :: out(:, :, :)
      integer :: i, j, k
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, g%ny
            do i = 1, g%nx
               out(i, j, k) = (u(i, j, k) - u(g%im(i), j, k)) / g%dx &
                            + (v(i, j, k) - v(i, g%jm(j), k)) / g%dy
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine div_eta3

   subroutine avg_v_to_u3(g, nz, v, out)
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: v(:, :, :)
      real(wp),     intent(out) :: out(:, :, :)
      integer :: i, j, k
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, g%ny
            do i = 1, g%nx
               out(i, j, k) = 0.25_wp * (v(i, j, k) + v(i, g%jm(j), k) &
                            + v(g%ip(i), j, k) + v(g%ip(i), g%jm(j), k))
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine avg_v_to_u3

   subroutine avg_u_to_v3(g, nz, u, out)
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: u(:, :, :)
      real(wp),     intent(out) :: out(:, :, :)
      integer :: i, j, k
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, g%ny
            do i = 1, g%nx
               out(i, j, k) = 0.25_wp * (u(i, j, k) + u(g%im(i), j, k) &
                            + u(i, g%jp(j), k) + u(g%im(i), g%jp(j), k))
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine avg_u_to_v3

end module mod_operators
