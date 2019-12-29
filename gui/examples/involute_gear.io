;; pitch radius, pressure angle (radians), diametral pitch
(define (gear r pa p)
    ;; Round to the nearest number of teeth
    (let* ((n (round (* r p 2)))

    ;; Updated pitch radius based on number of teeth
    (r (/ n p 2))

    (rb (* r (cos pa))) ;; base circle radius
    (a (/ p)) ;; addendum
    (d (/ 1.157 p)) ;; dedendum
    (ro (+ r a)) ;; outer radius
    (rr (- r d)) ;; inner radius

    (t (sqrt (- (/ (square r) (square rb)) 1)))
    (rot (+ (atan (- (sin t) (* t (cos t)))
                         (+ (cos t) (* t (sin t)))) (/ pi 2 n)))

    (tooth (lambda-shape (x y z)
      (let* ((r2 (+ (square x) (square y)))
             (r (sqrt r2)))
      (- (sqrt (max 0 (- r2 (square rb))))
         (* rb (+ (atan (/ y x)) (acos (min 1 (max -1 (/ rb r)))) rot))))))

    (tooth (intersection tooth
        (reflect-y tooth)
        (lambda-shape (x y z) (- x))))
    (teeth
        (apply union
            (map (lambda (i) (rotate-z tooth (* 2 pi (/ i n))))
                 (iota n)))))
    (scale-xyz
      (intersection (circle ro) (union teeth (circle rr)))
      [0.3 0.3 0.1])
))

(define out (union
  (sequence
    (gear 1.25 0.3 8)
    (difference(circle 0.1))
    (extrude-z -0.08 0.08))
  (sequence
    (gear 0.6 0.3 8)
    (difference (circle 0.11))
    (move [0 0.57])
    (extrude-z -0.04 0.04))
  (sequence
    (gear 0.6 0.3 8)
    (difference (circle 0.09))
    (reflect-xy)
    (move [0.57 0])
    (extrude-z -0.12 0.12))
  ))
out
#|
Notes:

(1)
We want to find the angle such that the involute curve
intersects a circle of radius R, where the involute is being
unwound from a circle of radius RB (and RB < R)

The involute has coordinates
    x, y = RB*(cos(t)+t*sin(t)), RB*(sin(t)-t*cos(t))
where t is a parameter.

Solving x**2 + y**2 = R**2 gives
    t = sqrt(R**2/RB**2 -1)
which we can plug back in to find x(t) and y(t).

We take atan2(y(t), x(t)) to find the angle, then add another
(2*pi)/(4*N) so that we can mirror the tooth about the x axis.


(2)
[Clever math & explanation by Peter Fedak, HMC 2013]

Assuming that restricting to x and y greater than
R is not interesting/challenging, an expression
separating one side of a portion of a circle involute
from another is

R(atan(y/x) + acos( R/sqrt(x^2+y^2) )) - sqrt(x^2+y^2-R^2)

which is 0 on the curve and negative for points to the northeast.
This assumes the involute starts at (1,0), if you want to rotate it,
or deal with a different turn of the involute, subtract R*rotation
angle.

For points P=(x,y) in the first quadrant, atan(y/x) accurately gives
the angle between the x-axis and the ray from O=(0,0) to P. Assuming
we are "unwinding" the involute counterclockwise, the place where the
"string" meets the circle, Q, will be the more-counterclockwise
tangent from P to the circle. Then O,P,Q is a right triangle, and the
length of the string, PQ, is sqrt(x^2+y^2-R^2) (right angle between
the tangent and the radius OQ). 

The angle of OQ from the x-axis is, again restricted to P in the first
quadrant, the sum of the angle of OP from the x-axis and one of the
angles in teh triangle. the cosine of the relevant angle is
R/sqrt(x^2+y^2), and as this angle will always be smaller than a right
angle, can by given by the inverse cosine of this ratio directly. The
condition for a point to lie on the involute is for the length of the
"string" to be equal to the amount unwound. The amount unwound is the
angle from OQ to the x-axis times R, which is the first term in the
expression. The length of the string is the expression after the minus
sign. Thus the involute is where this F=0, whereas if the point is too
close to the origin, the expression will be positive.
|#
