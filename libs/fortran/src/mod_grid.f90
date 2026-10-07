!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_grid                                                     !
!  Description: Arakawa C-grid geometry, transposed to (i,j) so that i  !
!               is contiguous in column-major storage - the same memory !
!               order as the Python reference's [j,i]                   !
!               (docs/03_discretization_spec.md S0, S2).                !
!  Pipeline: mod_config -> mod_grid -> mod_operators / mod_cases        !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_grid
   use mod_kinds, only: wp
   implicit none
   private
   public :: grid_t, build_grid, omp_min_points, set_omp_min_points

   ! Below this many grid points the OpenMP fork/join cost exceeds the work,
   ! so every parallel region falls back to serial via an if() clause.
   ! Measured on a 12-core M4 Pro at 64x64 = 4096 points: 12 threads ran 22x
   ! SLOWER than serial. The crossover is machine dependent - calibrate on the
   ! bench node and set it from the namelist (docs/09 section 7).
   integer, save :: omp_min_points = 65536

   type :: grid_t
      integer  :: nx = 0, ny = 0
      real(wp) :: lx = 0.0_wp, ly = 0.0_wp
      real(wp) :: dx = 0.0_wp, dy = 0.0_wp
      real(wp) :: cell_area = 0.0_wp
      ! Periodic neighbour index maps, precomputed to keep the stencil loops
      ! free of modulo arithmetic.
      integer, allocatable :: ip(:), im(:)   ! i+1, i-1  (size nx)
      integer, allocatable :: jp(:), jm(:)   ! j+1, j-1  (size ny)
   end type grid_t

contains

   subroutine set_omp_min_points(n)
      integer, intent(in) :: n
      if (n >= 0) omp_min_points = n
   end subroutine set_omp_min_points

   subroutine build_grid(nx, ny, lx, ly, grid)
      integer,  intent(in)  :: nx, ny
      real(wp), intent(in)  :: lx, ly
      type(grid_t), intent(out) :: grid
      integer :: i, j

      grid%nx = nx;  grid%ny = ny
      grid%lx = lx;  grid%ly = ly
      grid%dx = lx / real(nx, wp)
      grid%dy = ly / real(ny, wp)
      grid%cell_area = grid%dx * grid%dy

      allocate(grid%ip(nx), grid%im(nx), grid%jp(ny), grid%jm(ny))
      do i = 1, nx
         grid%ip(i) = merge(1, i + 1, i == nx)
         grid%im(i) = merge(nx, i - 1, i == 1)
      end do
      do j = 1, ny
         grid%jp(j) = merge(1, j + 1, j == ny)
         grid%jm(j) = merge(ny, j - 1, j == 1)
      end do
   end subroutine build_grid

end module mod_grid
