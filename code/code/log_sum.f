       SUBROUTINE log_sum(lx,nx,ls)

cf2py intent(out) ls
cf2py intent(hide) nx
    
       DOUBLE PRECISION lx(nx), ls, ed
       INTEGER nx, i
       
       ls = lx(1)       
       
       do i=2,nx
           ed=dexp(ls-lx(i))
!          Don't add this element if it's waaaaay smaller than the sum so far.
           if (ed<1D308) then
               ls = lx(i) + dlog(1.0D0 + ed)
           end if
       end do
       
       RETURN
       END