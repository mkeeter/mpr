;; http://paulbourke.net/geometry/decocube/
(define c 1)
(define d 0.8)
(lambda-shape (x y z)
  (-
  (* (+ (square (+ (square x) (square y) (- (square c))))
        (square (- (square z) 1)))
     (+ (square (+ (square y) (square z) (- (square c))))
        (square (- (square x) 1)))
     (+ (square (+ (square z) (square x) (- (square c))))
        (square (- (square y) 1))))
  d))
