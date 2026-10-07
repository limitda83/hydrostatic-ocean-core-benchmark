!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Module: mod_diag                                                     !
!  Description: Accuracy and conservation diagnostics matching          !
!               libs/core/diagnostics.py, plus the raw state dump and   !
!               metrics writer consumed by tools/compare_backends.py.   !
!  Pipeline: mod_schemes -> mod_diag -> the run directory (bin, json)      !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module mod_diag
   use mod_kinds, only: wp
   use mod_grid,  only: grid_t
   implicit none
   private
   public :: l2_rel, linf_rel, total_mass, total_energy, rms, jnum, &
             write_state_bin, write_metrics_json

contains

   function jnum(x) result(s)
      ! JSON has no NaN or Infinity literals. A diverged run must still write
      ! a parseable metrics file, so non-finite values become null and the
      ! divergence stays visible instead of corrupting the whole record.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      real(wp), intent(in) :: x
      character(len=32) :: s
      if (ieee_is_finite(x)) then
         write(s, '(es24.16)') x
      else
         s = '                    null'
      end if
   end function jnum

   pure function l2_rel(a, b) result(err)
      real(wp), intent(in) :: a(:, :), b(:, :)
      real(wp) :: err, num, den
      num = sqrt(sum((a - b)**2))
      den = sqrt(sum(b**2))
      if (den > 0.0_wp) then
         err = num / den
      else
         err = num
      end if
   end function l2_rel

   pure function linf_rel(a, b) result(err)
      real(wp), intent(in) :: a(:, :), b(:, :)
      real(wp) :: err, num, den
      num = maxval(abs(a - b))
      den = maxval(abs(b))
      if (den > 0.0_wp) then
         err = num / den
      else
         err = num
      end if
   end function linf_rel

   pure function rms(a) result(r)
      real(wp), intent(in) :: a(:, :)
      real(wp) :: r
      r = sqrt(sum(a**2) / real(size(a), wp))
   end function rms

   pure function total_mass(eta, g) result(m)
      real(wp),     intent(in) :: eta(:, :)
      type(grid_t), intent(in) :: g
      real(wp) :: m
      m = sum(eta) * g%cell_area
   end function total_mass

   function total_energy(u, v, eta, g, h0, gg) result(e)
      ! E = 1/2 sum[ H (u^2+v^2)|centre + g eta^2 ] dA  (spec S2.2)
      real(wp),     intent(in) :: u(:, :), v(:, :), eta(:, :)
      type(grid_t), intent(in) :: g
      real(wp),     intent(in) :: h0, gg
      real(wp) :: e, u2, v2
      integer  :: i, j
      e = 0.0_wp
      do j = 1, g%ny
         do i = 1, g%nx
            u2 = 0.5_wp * (u(i, j)**2 + u(g%im(i), j)**2)
            v2 = 0.5_wp * (v(i, j)**2 + v(i, g%jm(j))**2)
            e = e + h0 * (u2 + v2) + gg * eta(i, j)**2
         end do
      end do
      e = 0.5_wp * e * g%cell_area
   end function total_energy

   subroutine write_state_bin(path, u, v, eta)
      ! Stream fp64, order eta,u,v. Fortran (nx,ny) column-major has the same
      ! byte order as the reference's NumPy [ny,nx] row-major (spec S0), so
      ! numpy.fromfile().reshape(ny,nx) reads it directly.
      character(len=*), intent(in) :: path
      real(wp),         intent(in) :: u(:, :), v(:, :), eta(:, :)
      integer :: unit
      open(newunit=unit, file=trim(path), access='stream', form='unformatted', &
           status='replace', action='write')
      write(unit) eta
      write(unit) u
      write(unit) v
      close(unit)
   end subroutine write_state_bin

   subroutine write_metrics_json(path, nx, ny, dx, dt, n_steps, scheme, theta, &
                                 n_picard, solver, iters, failures, &
                                 l2_eta, l2_u, l2_v, linf_eta, mass, energy, &
                                 mass_drift, energy_drift, t_med, t_mad, &
                                 t_min, t_max, n_repeat, n_warmup, nthreads, host)
      character(len=*), intent(in) :: path, scheme, solver, host
      integer,  intent(in) :: nx, ny, n_steps, n_picard, iters, failures
      integer,  intent(in) :: n_repeat, n_warmup, nthreads
      real(wp), intent(in) :: dx, dt, theta, l2_eta, l2_u, l2_v, linf_eta
      real(wp), intent(in) :: mass, energy, mass_drift, energy_drift
      real(wp), intent(in) :: t_med, t_mad, t_min, t_max
      integer :: unit
      open(newunit=unit, file=trim(path), status='replace', action='write')
      write(unit, '(a)')            '{'
      write(unit, '(a)')            '  "backend": "fortran_cpu",'
      write(unit, '(a,a,a)')        '  "host": "', trim(host), '",'
      write(unit, '(a,i0,a)')       '  "nx": ', nx, ','
      write(unit, '(a,i0,a)')       '  "ny": ', ny, ','
      write(unit, '(a,a,a)')        '  "dx": ', trim(adjustl(jnum(dx))), ','
      write(unit, '(a,a,a)')        '  "dt": ', trim(adjustl(jnum(dt))), ','
      write(unit, '(a,i0,a)')       '  "n_steps": ', n_steps, ','
      write(unit, '(a,a,a)')        '  "scheme": "', trim(scheme), '",'
      write(unit, '(a,a,a)')        '  "theta": ', trim(adjustl(jnum(theta))), ','
      write(unit, '(a,i0,a)')       '  "n_picard": ', n_picard, ','
      write(unit, '(a,a,a)')        '  "solver": "', trim(solver), '",'
      write(unit, '(a,i0,a)')       '  "solver_iterations": ', iters, ','
      write(unit, '(a,i0,a)')       '  "solver_failures": ', failures, ','
      write(unit, '(a,a,a)')        '  "l2_rel_eta": ', trim(adjustl(jnum(l2_eta))), ','
      write(unit, '(a,a,a)')        '  "l2_rel_u": ', trim(adjustl(jnum(l2_u))), ','
      write(unit, '(a,a,a)')        '  "l2_rel_v": ', trim(adjustl(jnum(l2_v))), ','
      write(unit, '(a,a,a)')        '  "linf_rel_eta": ', trim(adjustl(jnum(linf_eta))), ','
      write(unit, '(a,a,a)')        '  "mass": ', trim(adjustl(jnum(mass))), ','
      write(unit, '(a,a,a)')        '  "energy": ', trim(adjustl(jnum(energy))), ','
      write(unit, '(a,a,a)')        '  "mass_drift": ', trim(adjustl(jnum(mass_drift))), ','
      write(unit, '(a,a,a)')        '  "energy_drift": ', trim(adjustl(jnum(energy_drift))), ','
      write(unit, '(a,a,a)')        '  "wall_s": ', trim(adjustl(jnum(t_med))), ','
      write(unit, '(a,a,a)')        '  "wall_mad_s": ', trim(adjustl(jnum(t_mad))), ','
      write(unit, '(a,a,a)')        '  "wall_min_s": ', trim(adjustl(jnum(t_min))), ','
      write(unit, '(a,a,a)')        '  "wall_max_s": ', trim(adjustl(jnum(t_max))), ','
      write(unit, '(a,i0,a)')       '  "n_repeat": ', n_repeat, ','
      write(unit, '(a,i0,a)')       '  "n_warmup": ', n_warmup, ','
      write(unit, '(a,i0)')         '  "omp_num_threads": ', nthreads
      write(unit, '(a)')            '}'
      close(unit)
   end subroutine write_metrics_json

end module mod_diag
