c SUBROUTINE ptsinpoly for multiple test points

      subroutine ptsinpoly(npts,x,y,xinf,yinf,n,xy,inpoly)
cf2py intent(hide) npts, n
cf2py intent(out) inpoly
      integer npts            
      real x(npts),y(npts)    
      real xinf,yinf        
      integer n             
      real xy(2,n)                    
      integer inpoly(npts)
      
      do 20 i=1,npts
      call ptinpoly(x(i),y(i),xinf,yinf,n,xy,inpoly(i))
  20  continue

      end

c SUBROUTINE ptinpoly for a single test point

      subroutine ptinpoly(x,y,xinf,yinf,n,xy,inpoly)
cf2py intent(hide) n
cf2py intent(out) inpoly
c    returns 1 if x,y is within polygon xy(1:2,1:n)
c    returns 0 if x,y is outside polygon xy(1:2,1:n)
c    (points are counterclockwise, with no repeated endpoint)
c    xinf,yinf is a point guaranteed to be outside the polygon
  
      integer n
      real x,y
      real xinf,yinf
      real xy(2,n)
      integer inpoly
  
c    local variables
      integer i,nint
      integer xi,yi
      real f1,f2    


      nint=0
      do 10 i=1,n
      i1=i
      if (i.eq.n) then
          i2=1
      else
          i2=i1+1
      endif
c      returns 0<f1<1 and 0<f2<1 if (x-xinf,y-yinf) intersects segment i
      call xsect(x,y,xy(1,i1),xy(2,i1),xy(1,i2),xy(2,i2),xinf,yinf
     . ,xi,yi,f1,f2)
      if (0.lt.f1.and.f1.lt.1.and.0.lt.f2.and.f2.lt.1) nint=nint+1

c    write(iotmsg,*) 'n,i,i1,i2,x,y=',n,i,i1,i2,x,y
c         write(iotmsg,*) 'xy=',xy(1,i1),xy(2,i1),xy(1,i2),xy(2,i2)
c         write(iotmsg,*) 'xinf,yinf,xi,yi,f1,f2,nint=',
c     x        xinf,yinf,xi,yi,f1,f2,nint

   10 continue
c    if nint even, point is outside

      if (mod(nint,2).eq.0) then
          inpoly=0
      else
          inpoly=1
      endif
  
      return
      end

      subroutine xsect(xa,ya,xb,yb,xc,yc,xd,yd,xe,ye,f1,f2)
c        given a segment a-d which crosses a segment b-c,
c        find the intersection e.
c
c    WARNING: if f1<0 or f1>1 or  f2<0 or f2>1, segments do not intersect
c    and the intersection (xe,ye) is not within the segments
c    
c    example:
c    xa=0,ya=1, xd=0,yd=-1
c    xb=-1,yb=0, xc=1,yc=0
c    xe=0,ye=0, f1=0.5,f2=0.5

c    f1 is the fractional distance (e-a)/(d-a)
c    f2 is the fractional distance (e-b)/(c-b)
c
c    xe=xa+f1*(xd-xa)
c    ye=ya+f1*(yd-ya)
c    xe=xb+f2*(xc-xb)
c    ye=yb+f2*(yc-yb)
c    example:
c    0=0+f1*(0-0)
c    0=1+f1*(-1-1)
c    0=-1+f2*(1+1)
c    0=0+f2*(0-0)

c    xa+f1*(xd-xa)=xb+f2*(xc-xb)
c    ya+f1*(yd-ya)=yb+f2*(yc-yb)
c    example:
c    0+f1*(0-0) = -1+f2*(1+1)
c    1+f1*(-1-1) = 0+f2*(0-0)
c    
c    (xd-xa)*f1 + (xb-xc)*f2 = (xb-xa)
c    (yd-ya)*f1 + (yb-yc)*f2 = (yb-ya)
c    example:
c    (0-0)*f1   + (-1-1)*f2  = (-1-0)
c    (-1-1)*f1  + (0-0)*f2   = (0-1)
c    
c    define:
c    A=(xd-xa)    B=(xb-xc)
c    C=(yd-ya)    D=(yb-yc)
c    det=AD-BC
c    example:
c    A=(0-0)=0    B=(-1-1)=-2
c    C=(-1-1)=-2    D=(0-0)=0
c    det=0*0-(-2)(-2)=-4
c
c    inverse matrix
c    A'=D/det    B'=-B/det
c    C'=-C/det    D'=A/det
c    example:
c    A'=0/(-4)=0    B'=2/(-4)=-0.5
c    C'=2/(-4)=-0.5    D'=0/(-4)=0
c
c    f1 = A'*(xb-xa) + B'*(yb-ya)
c    f2 = C'*(xb-xa) + D'*(yb-ya)
c    example:
c    f1 = 0*(-1-0)    + (-0.5)*(0-1) = 0.5
c    f2 = -0.5*(-1-0) + 0*(0-0)      = 0.5
c    

      real A,B,C,D,det,AI,BI,CI,DI,f1,f2
      real xa,ya,xb,yb,xc,yc,xd,yd,xe,ye
  
      xe=-999.
      ye=-999.
  
c    A=(xd-xa)    B=(xb-xc)
c    C=(yd-ya)    D=(yb-yc)
c    det=AD-BC

      A=xd-xa
      B=xb-xc
      C=yd-ya
      D=yb-yc
      det=A*D-B*C
  
      if (det.ne.0) then
  
          AI=D/det
          BI=-B/det
          CI=-C/det
          DI=A/det
  
          f1 = AI*(xb-xa) + BI*(yb-ya)
          f2 = CI*(xb-xa) + DI*(yb-ya)
  
          xe=xa+f1*(xd-xa)
          ye=ya+f1*(yd-ya)
c    the following should give the same result
          xe=xb+f2*(xc-xb)
          ye=yb+f2*(yc-yb)
  
      endif

c    write(iotmsg,'(a,4f8.4)') 'xsect: xa..xd=',xa,xb,xc,xd
c    write(iotmsg,'(a,4f8.4)') 'ysect: ya..yd=',ya,yb,yc,yd
c    write(iotmsg,'(a,5e12.4)') 'xsect: a,b,c,d,det=',a,b,c,d,det
c    write(iotmsg,'(a,4e12.4)') 'xsect: ai,bi,ci,di=',ai,bi,ci,di
c    write(iotmsg,'(a,4f8.4)') 'xsect: f1,f2,xe,ye=',f1,f2,xe,ye

      return
      end