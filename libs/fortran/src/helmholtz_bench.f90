!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: helmholtz_bench                                              !
!  Description: Standalone benchmark of the VARIABLE-COEFFICIENT free-  !
!               surface Helmholtz solve of spec S10.5,                  !
!                   eta - coef * D[ K G[eta] ] = rhs                    !
!               on a real bathymetry read from disk. Isolating the      !
!               elliptic solve is the point: it is the only part of the !
!               core whose cost changes with topography, and docs/21    !
!               showed it is also the part that decides whether a GPU   !
!               wins at all.                                            !
!                                                                       !
!  The same source builds three backends (RULES.md R5): gfortran/      !
!  nvfortran OpenMP CPU, and nvfortran OpenACC GPU. The OpenACC         !
!  directives sit beside the OpenMP ones so the arithmetic is           !
!  byte-identical across them.                                          !
!  Pipeline: tools/write_helmholtz_case.py -> helmholtz_bench -> JSON   !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_helm
   implicit none
   public

   ! Working precision of the SOLVER, selectable at compile time (axis G).
   ! The problem itself is always read from disk in fp64, so a reduced
   ! precision build solves the identical discrete system with a coarser
   ! arithmetic - which is the only way the accuracy loss is attributable.
#ifdef SINGLE_PRECISION
   integer, parameter :: wp = selected_real_kind(6, 37)
   character(len=*), parameter :: PRECISION_NAME = 'fp32'
#else
   integer, parameter :: wp = selected_real_kind(15, 307)
   character(len=*), parameter :: PRECISION_NAME = 'fp64'
#endif
   ! The files on disk are always fp64.
   integer, parameter :: dp = selected_real_kind(15, 307)

   type :: level_t
      integer :: nx = 0, ny = 0
      real(wp) :: dx = 0.0_wp, dy = 0.0_wp
      real(wp), allocatable :: ku(:, :), kv(:, :), msk(:, :)
      real(wp), allocatable :: dinv(:, :)
      real(wp), allocatable :: x(:, :), b(:, :), r(:, :)
   end type level_t

contains

   subroutine level_setup(lv, coef)
      type(level_t), intent(inout) :: lv
      real(wp),      intent(in)    :: coef
      integer  :: i, j, im, jm
      real(wp) :: cx, cy, d
      cx = coef / lv%dx**2
      cy = coef / lv%dy**2
      allocate(lv%dinv(lv%nx, lv%ny))
      do j = 1, lv%ny
         jm = merge(lv%ny, j - 1, j == 1)
         do i = 1, lv%nx
            im = merge(lv%nx, i - 1, i == 1)
            d = 1.0_wp + cx * (lv%ku(i, j) + lv%ku(im, j)) &
                       + cy * (lv%kv(i, j) + lv%kv(i, jm))
            if (lv%msk(i, j) <= 0.0_wp) d = 1.0_wp
            lv%dinv(i, j) = 1.0_wp / d
         end do
      end do
   end subroutine level_setup

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

   ! One red-black Gauss-Seidel sweep. colour = 0 (red) or 1 (black).
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

end module mod_helm


program helmholtz_bench
   use mod_helm
   implicit none

   integer, parameter :: MAXLVL = 12
   integer  :: nx = 128, ny = 128, max_iter = 20000, n_repeat = 5, n_warmup = 1
   real(dp) :: dx = 1.0_dp, dy = 1.0_dp, coef = 1.0_dp, rtol = 1.0e-10_dp
   character(len=256) :: datadir = '.', solver = 'pcg_jacobi', outfile = ''
   namelist /helmholtz/ nx, ny, dx, dy, coef, rtol, max_iter, datadir, &
                        solver, n_repeat, n_warmup, outfile

   type(level_t) :: lvl(MAXLVL)
   integer  :: nlvl, iters, rep, argc
   real(wp) :: residual
   real(dp) :: wall, t0, t1, err
   real(dp) :: wall_min = 0.0_dp, wall_mad = 0.0_dp
   real(dp), allocatable :: samples(:)
   real(wp) :: wcoef, wrtol, wdx, wdy
   real(wp), allocatable :: rhs(:, :), x(:, :), eta_ref(:, :)
   character(len=256) :: nmlfile
   logical :: have_ref

   argc = command_argument_count()
   if (argc < 1) then
      write(*, '(a)') 'usage: helmholtz_bench <case.nml> [solver]'
      stop 1
   end if
   call get_command_argument(1, nmlfile)
   open(unit=10, file=trim(nmlfile), status='old', action='read')
   read(10, nml=helmholtz)
   close(10)
   if (argc >= 2) call get_command_argument(2, solver)

   wcoef = real(coef, wp); wrtol = real(rtol, wp)
   wdx = real(dx, wp); wdy = real(dy, wp)
   call build_levels()
   ! Make every level resident on the device for the whole run. Without this
   ! the multigrid path has no data region at all and leans entirely on
   ! managed memory, whose page migration cost is platform-dependent: the
   ! same directives ran 4.7x slower than CUDA on one GPU and 20.8x on
   ! another (docs/25 S3.2). The PCG path keeps its own inner data region;
   ! this one covers the coarse levels the V-cycle walks.
   call levels_to_device()
   allocate(rhs(nx, ny), x(nx, ny))
   call read_field(trim(datadir)//'/rhs.bin', rhs, nx, ny)
   call read_optional(trim(datadir)//'/eta_ref.bin', eta_ref, nx, ny, have_ref)

   do rep = 1, n_warmup
      call run_solver(iters, residual)
   end do
   ! R7-2: median + MAD over the repeats (the minimum is kept as a secondary
   ! column; docs/25 was produced with min-of-3 before this change).
   allocate(samples(n_repeat))
   do rep = 1, n_repeat
      call wall_time(t0)
      call run_solver(iters, residual)
      call wall_time(t1)
      samples(rep) = t1 - t0
   end do
   wall = median_of(samples)
   wall_min = minval(samples)
   wall_mad = median_of(abs(samples - wall))

   err = -1.0_dp
   if (have_ref) err = l2_rel(x, eta_ref)

   call write_json()

contains

   function median_of(a) result(m)
      real(dp), intent(in) :: a(:)
      real(dp) :: m, b(size(a)), tmp
      integer  :: i, j, n
      b = a;  n = size(b)
      do i = 2, n
         tmp = b(i);  j = i - 1
         do while (j >= 1)
            if (b(j) <= tmp) exit
            b(j + 1) = b(j);  j = j - 1
         end do
         b(j + 1) = tmp
      end do
      if (modulo(n, 2) == 1) then
         m = b((n + 1) / 2)
      else
         m = 0.5_dp * (b(n / 2) + b(n / 2 + 1))
      end if
   end function median_of

   subroutine wall_time(t)
      real(dp), intent(out) :: t
      integer(kind=8) :: c, r
      call system_clock(count=c, count_rate=r)
      t = real(c, dp) / real(r, dp)
   end subroutine wall_time

   subroutine read_field(path, a, n1, n2)
      character(len=*), intent(in)  :: path
      integer,          intent(in)  :: n1, n2
      real(wp),         intent(out) :: a(n1, n2)
      real(dp), allocatable :: buf(:, :)
      integer :: u
      allocate(buf(n1, n2))
      open(newunit=u, file=path, form='unformatted', access='stream', &
           status='old', action='read')
      read(u) buf
      close(u)
      a = real(buf, wp)
   end subroutine read_field

   subroutine read_optional(path, a, n1, n2, ok)
      character(len=*),      intent(in)  :: path
      integer,               intent(in)  :: n1, n2
      real(wp), allocatable, intent(out) :: a(:, :)
      logical,               intent(out) :: ok
      inquire(file=path, exist=ok)
      if (.not. ok) return
      allocate(a(n1, n2))
      call read_field(path, a, n1, n2)
   end subroutine read_optional

   ! Coarse faces are the average of the two fine faces they replace and a
   ! coarse cell is wet if any child is (spec S10.5).
   subroutine build_levels()
      integer :: l, i, j, cx, cy
      lvl(1)%nx = nx; lvl(1)%ny = ny; lvl(1)%dx = wdx; lvl(1)%dy = wdy
      allocate(lvl(1)%ku(nx, ny), lvl(1)%kv(nx, ny), lvl(1)%msk(nx, ny))
      call read_field(trim(datadir)//'/ku.bin', lvl(1)%ku, nx, ny)
      call read_field(trim(datadir)//'/kv.bin', lvl(1)%kv, nx, ny)
      call read_field(trim(datadir)//'/mask.bin', lvl(1)%msk, nx, ny)
      call level_setup(lvl(1), wcoef)
      allocate(lvl(1)%x(nx, ny), lvl(1)%b(nx, ny), lvl(1)%r(nx, ny))
      nlvl = 1
      do l = 2, MAXLVL
         if (modulo(lvl(l-1)%nx, 2) /= 0 .or. modulo(lvl(l-1)%ny, 2) /= 0) exit
         if (lvl(l-1)%nx / 2 < 8 .or. lvl(l-1)%ny / 2 < 8) exit
         cx = lvl(l-1)%nx / 2; cy = lvl(l-1)%ny / 2
         lvl(l)%nx = cx; lvl(l)%ny = cy
         lvl(l)%dx = 2.0_wp * lvl(l-1)%dx; lvl(l)%dy = 2.0_wp * lvl(l-1)%dy
         allocate(lvl(l)%ku(cx, cy), lvl(l)%kv(cx, cy), lvl(l)%msk(cx, cy))
         do j = 1, cy
            do i = 1, cx
               lvl(l)%ku(i, j) = 0.5_wp * (lvl(l-1)%ku(2*i, 2*j-1) &
                                         + lvl(l-1)%ku(2*i, 2*j))
               lvl(l)%kv(i, j) = 0.5_wp * (lvl(l-1)%kv(2*i-1, 2*j) &
                                         + lvl(l-1)%kv(2*i, 2*j))
               lvl(l)%msk(i, j) = max(max(lvl(l-1)%msk(2*i-1, 2*j-1),  &
                                          lvl(l-1)%msk(2*i, 2*j-1)),   &
                                      max(lvl(l-1)%msk(2*i-1, 2*j),    &
                                          lvl(l-1)%msk(2*i, 2*j)))
            end do
         end do
         call level_setup(lvl(l), wcoef)
         allocate(lvl(l)%x(cx, cy), lvl(l)%b(cx, cy), lvl(l)%r(cx, cy))
         nlvl = l
      end do
   end subroutine build_levels

   real(dp) function l2_rel(a, b) result(e)
      real(wp), intent(in) :: a(:, :), b(:, :)
      real(dp) :: num, den
      num = sqrt(sum(real(a - b, dp)**2)); den = sqrt(sum(real(b, dp)**2))
      e = merge(num / den, num, den > 0.0_dp)
   end function l2_rel

   real(wp) function dot(a, b) result(s)
      real(wp), intent(in) :: a(:, :), b(:, :)
      integer  :: i, j
      ! A combined construct, not an orphaned one: dot is called from serial
      ! context between the hoisted parallel regions, and OpenMP forbids
      ! reducing into a variable that is already private in an enclosing
      ! region - which is exactly what an orphaned !$omp do here would be.
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

   subroutine run_solver(iters, residual)
      integer,  intent(out) :: iters
      real(wp), intent(out) :: residual
      select case (trim(solver))
      case ('pcg_jacobi');  call pcg(.false., iters, residual)
      case ('pcg_rbgs');    call pcg(.true., iters, residual)
      case ('rbgs');        call smoother_solve(.false., iters, residual)
      case ('multigrid');   call smoother_solve(.true., iters, residual)
      case default
         write(*, '(a)') 'FATAL: unknown solver '//trim(solver)
         stop 1
      end select
   end subroutine run_solver

   subroutine precondition(use_rbgs, r, z)
      logical,  intent(in)    :: use_rbgs
      real(wp), intent(in)    :: r(:, :)
      real(wp), intent(inout) :: z(:, :)
      integer :: i, j
      if (.not. use_rbgs) then
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               z(i, j) = lvl(1)%dinv(i, j) * r(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      else
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               z(i, j) = 0.0_wp
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         ! Symmetric sweep: forward then backward keeps M symmetric, which
         ! is what CG requires of its preconditioner.
         call rbgs_colour(lvl(1), wcoef, z, r, 0)
         call rbgs_colour(lvl(1), wcoef, z, r, 1)
         call rbgs_colour(lvl(1), wcoef, z, r, 1)
         call rbgs_colour(lvl(1), wcoef, z, r, 0)
      end if
   end subroutine precondition

   subroutine pcg(use_rbgs, iters, residual)
      logical,  intent(in)  :: use_rbgs
      integer,  intent(out) :: iters
      real(wp), intent(out) :: residual
      real(wp), allocatable, save :: r(:, :), z(:, :), p(:, :), ap(:, :)
      real(wp) :: norm_b, rz, rz_new, alpha, beta, tol, best_res
      integer  :: it, i, j, stall
      if (.not. allocated(r)) allocate(r(nx, ny), z(nx, ny), p(nx, ny), ap(nx, ny))
      !$acc data copyin(lvl(1)%ku, lvl(1)%kv, lvl(1)%msk, lvl(1)%dinv, rhs) &
      !$acc      create(r, z, p, ap) copyout(x)
      !$omp parallel default(shared)
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector
      do j = 1, ny
         do i = 1, nx
            x(i, j) = 0.0_wp
            r(i, j) = rhs(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel
      norm_b = sqrt(dot(rhs, rhs))
      tol = wrtol * norm_b
      iters = 0; residual = norm_b
      best_res = huge(1.0_wp); stall = 0
      if (norm_b > 0.0_wp) then
         !$omp parallel default(shared)
         call precondition(use_rbgs, r, z)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, ny
            do i = 1, nx
               p(i, j) = z(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         !$omp end parallel
         rz = dot(r, z)
         do it = 1, max_iter
            !$omp parallel default(shared)
            call helm_apply(lvl(1), wcoef, p, ap)
            !$omp end parallel
            alpha = rz / dot(p, ap)
            !$omp parallel default(shared)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  x(i, j) = x(i, j) + alpha * p(i, j)
                  r(i, j) = r(i, j) - alpha * ap(i, j)
               end do
            end do
            !$acc end parallel loop
            !$omp end do
            !$omp end parallel
            residual = sqrt(dot(r, r))
            iters = it
            if (residual <= tol) exit
            ! Reduced precision cannot reach an arbitrary tolerance: the
            ! residual settles where the matrix-vector product stops being
            ! accurate. Detect that instead of spinning to max_iter.
            !
            ! ONLY in the reduced-precision build. CG's residual is not
            ! monotone, and a converging fp64 solve can spend forty
            ! iterations without improving on its best - one did, and was
            ! recorded as 445 iterations with residual 1.8e-05 where the
            ! true answer is 869 iterations and 9.9e-11 (docs/90 N8).
            if (wp /= dp) then
               if (residual < best_res * 0.99_wp) then
                  best_res = residual; stall = 0
               else
                  stall = stall + 1
                  if (stall >= 40) exit
               end if
            end if
            !$omp parallel default(shared)
            call precondition(use_rbgs, r, z)
            !$omp end parallel
            rz_new = dot(r, z)
            beta = rz_new / rz
            !$omp parallel default(shared)
            !$omp do collapse(2) schedule(static)
            !$acc parallel loop collapse(2) gang vector
            do j = 1, ny
               do i = 1, nx
                  p(i, j) = z(i, j) + beta * p(i, j)
               end do
            end do
            !$acc end parallel loop
            !$omp end do
            !$omp end parallel
            rz = rz_new
         end do
         residual = residual / norm_b
      end if
      !$acc end data
   end subroutine pcg

   recursive subroutine vcycle(l)
      integer, intent(in) :: l
      integer :: i, j, s
      do s = 1, 2
         call rbgs_colour(lvl(l), wcoef, lvl(l)%x, lvl(l)%b, 0)
         call rbgs_colour(lvl(l), wcoef, lvl(l)%x, lvl(l)%b, 1)
      end do
      if (l < nlvl) then
         call helm_apply(lvl(l), wcoef, lvl(l)%x, lvl(l)%r)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, lvl(l)%ny
            do i = 1, lvl(l)%nx
               lvl(l)%r(i, j) = (lvl(l)%b(i, j) - lvl(l)%r(i, j)) * lvl(l)%msk(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, lvl(l+1)%ny
            do i = 1, lvl(l+1)%nx
               lvl(l+1)%b(i, j) = 0.25_wp * (lvl(l)%r(2*i-1, 2*j-1)  &
                                           + lvl(l)%r(2*i,   2*j-1)  &
                                           + lvl(l)%r(2*i-1, 2*j)    &
                                           + lvl(l)%r(2*i,   2*j))
               lvl(l+1)%x(i, j) = 0.0_wp
            end do
         end do
         !$acc end parallel loop
         !$omp end do
         call vcycle(l + 1)
         !$omp do collapse(2) schedule(static)
         !$acc parallel loop collapse(2) gang vector
         do j = 1, lvl(l)%ny
            do i = 1, lvl(l)%nx
               lvl(l)%x(i, j) = lvl(l)%x(i, j)                              &
                  + lvl(l+1)%x((i + 1) / 2, (j + 1) / 2) * lvl(l)%msk(i, j)
            end do
         end do
         !$acc end parallel loop
         !$omp end do
      end if
      do s = 1, 2
         call rbgs_colour(lvl(l), wcoef, lvl(l)%x, lvl(l)%b, 1)
         call rbgs_colour(lvl(l), wcoef, lvl(l)%x, lvl(l)%b, 0)
      end do
   end subroutine vcycle

   subroutine smoother_solve(use_mg, iters, residual)
      logical,  intent(in)  :: use_mg
      integer,  intent(out) :: iters
      real(wp), intent(out) :: residual
      real(wp), allocatable, save :: res(:, :)
      real(wp) :: norm_b, tol, best_res
      integer  :: it, i, j, stall
      if (.not. allocated(res)) allocate(res(nx, ny))
      norm_b = sqrt(sum(rhs * rhs))
      tol = wrtol * norm_b
      !$omp parallel default(shared)
      !$omp do collapse(2) schedule(static)
      !$acc parallel loop collapse(2) gang vector copyin(rhs)
      do j = 1, ny
         do i = 1, nx
            lvl(1)%x(i, j) = 0.0_wp
            lvl(1)%b(i, j) = rhs(i, j)
         end do
      end do
      !$acc end parallel loop
      !$omp end do
      !$omp end parallel
      iters = 0; residual = norm_b
      best_res = huge(1.0_wp); stall = 0
      do it = 1, max_iter
         !$omp parallel default(shared)
         if (use_mg) then
            call vcycle(1)
         else
            call rbgs_colour(lvl(1), wcoef, lvl(1)%x, lvl(1)%b, 0)
            call rbgs_colour(lvl(1), wcoef, lvl(1)%x, lvl(1)%b, 1)
         end if
         call helm_apply(lvl(1), wcoef, lvl(1)%x, res)
         !$omp end parallel
         residual = 0.0_wp
         !$omp parallel default(shared)
         !$omp do collapse(2) schedule(static) reduction(+:residual)
         do j = 1, ny
            do i = 1, nx
               residual = residual + (rhs(i, j) - res(i, j))**2
            end do
         end do
         !$omp end do
         !$omp end parallel
         residual = sqrt(residual)
         iters = it
         if (residual <= tol) exit
         if (wp /= dp) then
            if (residual < best_res * 0.99_wp) then
               best_res = residual; stall = 0
            else
               stall = stall + 1
               if (stall >= 40) exit
            end if
         end if
      end do
      !$acc update self(lvl(1)%x)
      x = lvl(1)%x
      residual = residual / merge(norm_b, 1.0_wp, norm_b > 0.0_wp)
   end subroutine smoother_solve

   subroutine levels_to_device()
      integer :: l
      do l = 1, nlvl
         !$acc enter data copyin(lvl(l)%ku, lvl(l)%kv, lvl(l)%msk, lvl(l)%dinv)
         !$acc enter data create(lvl(l)%x, lvl(l)%b, lvl(l)%r)
      end do
   end subroutine levels_to_device

   subroutine levels_from_device()
      integer :: l
      do l = nlvl, 1, -1
         !$acc exit data delete(lvl(l)%x, lvl(l)%b, lvl(l)%r)
         !$acc exit data delete(lvl(l)%ku, lvl(l)%kv, lvl(l)%msk, lvl(l)%dinv)
      end do
   end subroutine levels_from_device

   subroutine write_json()
      integer :: u
      character(len=512) :: path
      path = trim(outfile)
      if (len_trim(path) == 0) path = trim(datadir)//'/metrics_'//trim(solver)//'.json'
      open(newunit=u, file=trim(path), status='replace', action='write')
      write(u, '(a)') '{'
      write(u, '(a)')     '  "backend": "fortran",'
      write(u, '(a)')     '  "precision": "'//PRECISION_NAME//'",'
      write(u, '(a)')     '  "solver": "'//trim(solver)//'",'
      write(u, '(a,i0,a)')'  "nx": ', nx, ','
      write(u, '(a,i0,a)')'  "ny": ', ny, ','
      write(u, '(a,i0,a)')'  "iterations": ', iters, ','
      write(u, '(a,es24.16,a)') '  "wall_s": ', wall, ','
      write(u, '(a,es24.16,a)') '  "wall_mad_s": ', wall_mad, ','
      write(u, '(a,es24.16,a)') '  "wall_min_s": ', wall_min, ','
      write(u, '(a,es24.16,a)') '  "residual": ', residual, ','
      write(u, '(a,es24.16,a)') '  "l2_rel_vs_reference": ', err, ','
      write(u, '(a,i0)')  '  "n_repeat": ', n_repeat
      write(u, '(a)') '}'
      close(u)
      write(*, '(a,a12,a,i8,a,es12.5,a,es12.5)') 'solver=', trim(solver), &
         ' iters=', iters, ' wall=', wall, ' l2_vs_ref=', err
   end subroutine write_json

end program helmholtz_bench
