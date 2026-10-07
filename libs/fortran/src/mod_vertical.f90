!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_vertical                                                 !
!  Description: Vertical operators and the batched tridiagonal solve of !
!               docs/03_discretization_spec.md S7.3-S7.4. Arrays are    !
!               (nx, ny, nz) so k is the slowest axis: every step of    !
!               the Thomas recursion then sweeps one contiguous 2D      !
!               plane, which is the layout the vectorised reference     !
!               uses and the one a GPU wants.                           !
!  Pipeline: mod_grid -> mod_vertical -> mod_model3d                    !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_vertical
   use mod_kinds, only: wp
   use mod_grid,  only: grid_t, omp_min_points
   implicit none
   private
   public :: thomas_batched, thomas_column, tridiag_kernel, set_tridiag_kernel
   public :: diffusion_coeffs, apply_diffusion, &
             w_from_divergence, w_at_centres, buoyancy_potential, &
             remove_depth_mean, depth_integral

   ! Which tridiagonal kernel to use:
   !   'plane'  - one worksharing region per k, sweeping a contiguous 2D plane.
   !              Vectorises well on CPU, but on GPU it costs 2*nz kernel
   !              launches per solve and becomes launch-latency bound.
   !   'column' - one parallel region over (i,j), with the whole k recursion
   !              inside a single thread. One kernel per solve. Because k is
   !              the slowest axis, neighbouring threads still read adjacent
   !              memory, so the accesses stay coalesced.
   character(len=8), save :: tridiag_kernel = 'plane'

contains

   subroutine set_tridiag_kernel(name)
      character(len=*), intent(in) :: name
      if (trim(name) == 'plane' .or. trim(name) == 'column') then
         tridiag_kernel = trim(name)
      else
         write(*, '(a)') 'FATAL: tridiag_kernel must be plane|column; got '//trim(name)
         error stop 1
      end if
   end subroutine set_tridiag_kernel

   subroutine thomas_column(g, nz, sub, diag, sup, rhs, x, cstar)
      ! Thread-per-column Thomas: the sequential k recursion lives inside one
      ! kernel launch instead of 2*nz of them.
      type(grid_t), intent(in)    :: g
      integer,      intent(in)    :: nz
      real(wp),     intent(in)    :: sub(:,:,:), diag(:,:,:), sup(:,:,:), rhs(:,:,:)
      real(wp),     intent(out)   :: x(:,:,:)
      real(wp),     intent(inout) :: cstar(:,:,:)
      integer  :: i, j, k
      real(wp) :: denom

      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(k, denom)
      do j = 1, g%ny
         do i = 1, g%nx
            cstar(i, j, 1) = sup(i, j, 1) / diag(i, j, 1)
            x(i, j, 1) = rhs(i, j, 1) / diag(i, j, 1)
            do k = 2, nz
               denom = diag(i, j, k) - sub(i, j, k) * cstar(i, j, k - 1)
               cstar(i, j, k) = sup(i, j, k) / denom
               x(i, j, k) = (rhs(i, j, k) - sub(i, j, k) * x(i, j, k - 1)) / denom
            end do
            do k = nz - 1, 1, -1
               x(i, j, k) = x(i, j, k) - cstar(i, j, k) * x(i, j, k + 1)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine thomas_column

   subroutine thomas_batched(g, nz, sub, diag, sup, rhs, x, cstar, dstar)
      ! One tridiagonal system per column, vectorised over (i,j).
      ! Coefficients are full 3D arrays even when column-independent: the real
      ! case (turbulence closure) gives a different matrix per column, and
      ! factorising once would understate the cost this benchmark measures.
      type(grid_t), intent(in)    :: g
      integer,      intent(in)    :: nz
      real(wp),     intent(in)    :: sub(:,:,:), diag(:,:,:), sup(:,:,:), rhs(:,:,:)
      real(wp),     intent(out)   :: x(:,:,:)
      real(wp),     intent(inout) :: cstar(:,:,:), dstar(:,:,:)   ! scratch
      integer  :: i, j, k
      real(wp) :: denom

      if (tridiag_kernel == 'column') then
         call thomas_column(g, nz, sub, diag, sup, rhs, x, cstar)
         return
      end if

      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            cstar(i, j, 1) = sup(i, j, 1) / diag(i, j, 1)
            dstar(i, j, 1) = rhs(i, j, 1) / diag(i, j, 1)
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      do k = 2, nz
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector private(denom)
         do j = 1, g%ny
            do i = 1, g%nx
               denom = diag(i, j, k) - sub(i, j, k) * cstar(i, j, k - 1)
               cstar(i, j, k) = sup(i, j, k) / denom
               dstar(i, j, k) = (rhs(i, j, k) - sub(i, j, k) * dstar(i, j, k - 1)) / denom
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end do

      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            x(i, j, nz) = dstar(i, j, nz)
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      do k = nz - 1, 1, -1
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, g%ny
            do i = 1, g%nx
               x(i, j, k) = dstar(i, j, k) - cstar(i, j, k) * x(i, j, k + 1)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end do
   end subroutine thomas_batched

   subroutine diffusion_coeffs(g, nz, nu, dz, dt, theta_v, drag, sub, diag, sup)
      ! Coefficients of A = I - theta_v*dt*Dz (spec S7.4). nu lives at
      ! interfaces, nu(:,:,k) being the interface above layer k.
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: nu(:,:,:)        ! (nx, ny, nz+1)
      real(wp),     intent(in)  :: dz, dt, theta_v, drag
      real(wp),     intent(out) :: sub(:,:,:), diag(:,:,:), sup(:,:,:)
      integer  :: i, j, k
      real(wp) :: fac, a_top, a_bot

      fac = theta_v * dt / dz**2
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(a_top, a_bot)
      do k = 1, nz
         do j = 1, g%ny
            do i = 1, g%nx
               a_top = fac * nu(i, j, k)
               a_bot = fac * nu(i, j, k + 1)
               sub(i, j, k) = -a_top
               sup(i, j, k) = -a_bot
               diag(i, j, k) = 1.0_wp + a_top + a_bot
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      ! Surface and bottom: prescribed flux, so the coupling term is dropped.
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            diag(i, j, 1) = 1.0_wp + fac * nu(i, j, 2)
            sub(i, j, 1) = 0.0_wp
            diag(i, j, nz) = 1.0_wp + fac * nu(i, j, nz) + theta_v * dt * drag / dz
            sup(i, j, nz) = 0.0_wp
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine diffusion_coeffs

   subroutine apply_diffusion(g, nz, field, nu, dz, surf_flux, drag, out)
      ! Explicit Dz[field] = d/dz( nu d(field)/dz ), same boundary treatment.
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: field(:,:,:), nu(:,:,:)
      real(wp),     intent(in)  :: dz, drag
      real(wp),     intent(in)  :: surf_flux(:,:)
      real(wp),     intent(out) :: out(:,:,:)
      integer  :: i, j, k
      real(wp) :: ftop, fbot

      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector private(ftop, fbot)
      do k = 1, nz
         do j = 1, g%ny
            do i = 1, g%nx
               if (k == 1) then
                  ftop = surf_flux(i, j)
               else
                  ftop = nu(i, j, k) * (field(i, j, k - 1) - field(i, j, k)) / dz
               end if
               if (k == nz) then
                  fbot = drag * field(i, j, nz)
               else
                  fbot = nu(i, j, k + 1) * (field(i, j, k) - field(i, j, k + 1)) / dz
               end if
               out(i, j, k) = (ftop - fbot) / dz
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine apply_diffusion

   subroutine w_from_divergence(g, nz, div, dz, w)
      ! w at interfaces, integrating up from the bottom where w = 0.
      ! w(:,:,k) is the interface above layer k; w(:,:,nz+1) is the bottom.
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: div(:,:,:)
      real(wp),     intent(in)  :: dz
      real(wp),     intent(out) :: w(:,:,:)        ! (nx, ny, nz+1)
      integer :: i, j, k

      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            w(i, j, nz + 1) = 0.0_wp
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      do k = nz, 1, -1
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, g%ny
            do i = 1, g%nx
               w(i, j, k) = w(i, j, k + 1) - dz * div(i, j, k)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end do
   end subroutine w_from_divergence

   subroutine w_at_centres(g, nz, w, wc)
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: w(:,:,:)
      real(wp),     intent(out) :: wc(:,:,:)
      integer :: i, j, k
      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, g%ny
            do i = 1, g%nx
               wc(i, j, k) = 0.5_wp * (w(i, j, k) + w(i, j, k + 1))
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine w_at_centres

   subroutine buoyancy_potential(g, nz, b, dz, phi)
      ! Phi(k) = dz*( 0.5*b(k) + sum_{k'<k} b(k') ), then the depth mean is
      ! removed by remove_depth_mean (spec S7.2 - a correctness condition).
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: b(:,:,:)
      real(wp),     intent(in)  :: dz
      real(wp),     intent(out) :: phi(:,:,:)
      integer :: i, j, k

      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, g%ny
         do i = 1, g%nx
            phi(i, j, 1) = dz * 0.5_wp * b(i, j, 1)
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      do k = 2, nz
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, g%ny
            do i = 1, g%nx
               phi(i, j, k) = phi(i, j, k - 1) &
                            + dz * 0.5_wp * (b(i, j, k - 1) + b(i, j, k))
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end do
   end subroutine buoyancy_potential

   subroutine remove_depth_mean(g, nz, phi, work)
      type(grid_t), intent(in)    :: g
      integer,      intent(in)    :: nz
      real(wp),     intent(inout) :: phi(:,:,:)
      real(wp),     intent(inout) :: work(:,:)     ! scratch for the mean
      integer  :: i, j, k
      real(wp) :: acc

      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(acc, k)
      do j = 1, g%ny
         do i = 1, g%nx
            acc = 0.0_wp
            do k = 1, nz
               acc = acc + phi(i, j, k)
            end do
            work(i, j) = acc / real(nz, wp)
         end do
      end do
      !$acc end parallel loop
      !$omp end do

      !$omp do collapse(3) schedule(static)
      !$acc parallel loop collapse(3) gang vector
      do k = 1, nz
         do j = 1, g%ny
            do i = 1, g%nx
               phi(i, j, k) = phi(i, j, k) - work(i, j)
            end do
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine remove_depth_mean

   subroutine depth_integral(g, nz, a, dz, out)
      type(grid_t), intent(in)  :: g
      integer,      intent(in)  :: nz
      real(wp),     intent(in)  :: a(:,:,:)
      real(wp),     intent(in)  :: dz
      real(wp),     intent(out) :: out(:,:)
      integer  :: i, j, k
      real(wp) :: acc
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector private(acc, k)
      do j = 1, g%ny
         do i = 1, g%nx
            acc = 0.0_wp
            do k = 1, nz
               acc = acc + a(i, j, k)
            end do
            out(i, j) = dz * acc
         end do
      end do
      !$acc end parallel loop
      !$omp end do
   end subroutine depth_integral

end module mod_vertical
