!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_domain5                                                  !
!  Description: The v0.5 domain of spec S10.2-S10.4 for the compiled    !
!               backend: bathymetry and land mask read from the binary  !
!               that tools/toml2nml.py writes, then the face masks,     !
!               face depths, z-level partial-cell thickness, 3D masks   !
!               and layer-centre depths derived from them exactly as    !
!               libs/core/domain.py does.                               !
!                                                                       !
!  The bathymetry is never generated here. One generator in Python,     !
!  read by every backend, is what lets the R2 gate compare the          !
!  backends on a byte-identical problem (docs/25 S1).                   !
!  Pipeline: toml2nml.py -> <prefix>_domain.bin -> mod_domain5          !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_domain5
   use mod_kinds, only: wp, dp
   implicit none
   private
   public :: domain5_t, domain5_build, domain5_read

   type :: domain5_t
      integer  :: nx = 0, ny = 0, nz = 0
      character(len=16) :: face_rule = 'min'
      character(len=16) :: bc_x = 'periodic', bc_y = 'periodic'
      real(wp) :: min_partial = 0.1_wp
      ! 2D [nx,ny]
      real(wp), allocatable :: h(:, :), mask(:, :), masku(:, :), maskv(:, :)
      real(wp), allocatable :: hu(:, :), hv(:, :)
      ! 1D [nz]
      real(wp), allocatable :: dzr(:)
      ! 3D [nx,ny,nz]
      real(wp), allocatable :: dz3(:, :, :), dz3u(:, :, :), dz3v(:, :, :)
      real(wp), allocatable :: mask3(:, :, :), mask3u(:, :, :), mask3v(:, :, :)
      real(wp), allocatable :: inv_dz3(:, :, :), zc(:, :, :)
      ! Wet-column depth at the two face sets (sum of dz3u/dz3v), and 1/it.
      real(wp), allocatable :: hcu(:, :), hcv(:, :), inv_hcu(:, :), inv_hcv(:, :)
   end type domain5_t

contains

   subroutine domain5_read(path, nx, ny, h, mask)
      character(len=*), intent(in)  :: path
      integer,          intent(in)  :: nx, ny
      real(wp),         intent(out) :: h(nx, ny), mask(nx, ny)
      real(dp), allocatable :: buf(:, :)      ! the file is fp64 in every build
      integer :: u, ios
      open(newunit=u, file=path, form='unformatted', access='stream', &
           status='old', action='read', iostat=ios)
      if (ios /= 0) then
         write(*, '(a)') 'FATAL: cannot open domain file '//trim(path)
         error stop 1
      end if
      allocate(buf(nx, ny))
      read(u) buf;  h = real(buf, wp)
      read(u) buf;  mask = real(buf, wp)
      deallocate(buf)
      close(u)
   end subroutine domain5_read

   pure function face_of(a, b, rule) result(f)
      real(wp),         intent(in) :: a, b
      character(len=*), intent(in) :: rule
      real(wp) :: f
      select case (trim(rule))
      case ('mean')
         f = 0.5_wp * (a + b)
      case ('harmonic')
         if (a + b > 0.0_wp) then
            f = 2.0_wp * a * b / (a + b)
         else
            f = 0.0_wp
         end if
      case default
         f = min(a, b)
      end select
   end function face_of

   subroutine domain5_build(d, nx, ny, nz, h, mask, face_rule, min_partial, bc_x, bc_y)
      type(domain5_t),  intent(inout) :: d
      integer,          intent(in)    :: nx, ny, nz
      real(wp),         intent(in)    :: h(nx, ny), mask(nx, ny), min_partial
      character(len=*), intent(in)    :: face_rule, bc_x, bc_y
      integer  :: i, j, k, ip, jp
      real(wp) :: hmax, edge, remaining, thick, above

      d%nx = nx;  d%ny = ny;  d%nz = nz
      d%face_rule = face_rule;  d%min_partial = min_partial
      d%bc_x = bc_x;  d%bc_y = bc_y

      allocate(d%h(nx, ny), d%mask(nx, ny), d%masku(nx, ny), d%maskv(nx, ny))
      allocate(d%hu(nx, ny), d%hv(nx, ny), d%dzr(nz))
      allocate(d%dz3(nx, ny, nz), d%dz3u(nx, ny, nz), d%dz3v(nx, ny, nz))
      allocate(d%mask3(nx, ny, nz), d%mask3u(nx, ny, nz), d%mask3v(nx, ny, nz))
      allocate(d%inv_dz3(nx, ny, nz), d%zc(nx, ny, nz))
      allocate(d%hcu(nx, ny), d%hcv(nx, ny), d%inv_hcu(nx, ny), d%inv_hcv(nx, ny))
      d%h = h;  d%mask = mask

      ! Face masks: wet only if both neighbours are wet; a domain-edge face is
      ! dry unless that direction is periodic (spec S10.4).
      do j = 1, ny
         jp = merge(1, j + 1, j == ny)
         do i = 1, nx
            ip = merge(1, i + 1, i == nx)
            d%masku(i, j) = mask(i, j) * mask(ip, j)
            d%maskv(i, j) = mask(i, j) * mask(i, jp)
            if (trim(bc_x) /= 'periodic' .and. i == nx) d%masku(i, j) = 0.0_wp
            if (trim(bc_y) /= 'periodic' .and. j == ny) d%maskv(i, j) = 0.0_wp
            d%hu(i, j) = d%masku(i, j) * face_of(h(i, j), h(ip, j), face_rule)
            d%hv(i, j) = d%maskv(i, j) * face_of(h(i, j), h(i, jp), face_rule)
         end do
      end do

      ! Reference levels span the DEEPEST wet column (docs/24: otherwise every
      ! column deeper than the nominal depth silently loses its bottom).
      hmax = 0.0_wp
      do j = 1, ny
         do i = 1, nx
            if (mask(i, j) > 0.0_wp) hmax = max(hmax, h(i, j))
         end do
      end do
      d%dzr = hmax / real(nz, wp)

      ! z-level with partial cells: fill fixed levels top-down, the last wet
      ! cell is the remainder, cells thinner than min_partial*dzr are dry.
      do j = 1, ny
         do i = 1, nx
            edge = 0.0_wp
            do k = 1, nz
               remaining = h(i, j) - edge
               thick = min(max(remaining, 0.0_wp), d%dzr(k))
               if (thick > 0.0_wp .and. thick < min_partial * d%dzr(k)) thick = 0.0_wp
               if (mask(i, j) <= 0.0_wp) thick = 0.0_wp
               d%dz3(i, j, k) = thick
               d%mask3(i, j, k) = merge(1.0_wp, 0.0_wp, thick > 0.0_wp)
               edge = edge + d%dzr(k)
            end do
         end do
      end do

      ! 3D face masks and face thickness (minimum rule of S10.2 for the
      ! thickness regardless of face_rule, matching domain.py).
      do k = 1, nz
         do j = 1, ny
            jp = merge(1, j + 1, j == ny)
            do i = 1, nx
               ip = merge(1, i + 1, i == nx)
               d%mask3u(i, j, k) = d%mask3(i, j, k) * d%mask3(ip, j, k) * d%masku(i, j)
               d%mask3v(i, j, k) = d%mask3(i, j, k) * d%mask3(i, jp, k) * d%maskv(i, j)
               d%dz3u(i, j, k) = d%mask3u(i, j, k) * face_of(d%dz3(i, j, k), d%dz3(ip, j, k), face_rule)
               d%dz3v(i, j, k) = d%mask3v(i, j, k) * face_of(d%dz3(i, j, k), d%dz3(i, jp, k), face_rule)
               if (d%dz3(i, j, k) > 0.0_wp) then
                  d%inv_dz3(i, j, k) = 1.0_wp / d%dz3(i, j, k)
               else
                  d%inv_dz3(i, j, k) = 0.0_wp
               end if
            end do
         end do
      end do

      ! Layer-centre depth, positive downward, for the EOS (spec S10.6).
      do j = 1, ny
         do i = 1, nx
            above = 0.0_wp
            do k = 1, nz
               d%zc(i, j, k) = above + 0.5_wp * d%dz3(i, j, k)
               above = above + d%dz3(i, j, k)
            end do
         end do
      end do

      ! Wet column depth at faces and its inverse.
      do j = 1, ny
         do i = 1, nx
            d%hcu(i, j) = 0.0_wp;  d%hcv(i, j) = 0.0_wp
            do k = 1, nz
               d%hcu(i, j) = d%hcu(i, j) + d%dz3u(i, j, k)
               d%hcv(i, j) = d%hcv(i, j) + d%dz3v(i, j, k)
            end do
            d%inv_hcu(i, j) = merge(1.0_wp / max(d%hcu(i, j), tiny(1.0_wp)), 0.0_wp, d%hcu(i, j) > 0.0_wp)
            d%inv_hcv(i, j) = merge(1.0_wp / max(d%hcv(i, j), tiny(1.0_wp)), 0.0_wp, d%hcv(i, j) > 0.0_wp)
         end do
      end do
   end subroutine domain5_build

end module mod_domain5
