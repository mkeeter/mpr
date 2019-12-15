(set-bounds! [-50 -50 -20] [50 50 20])
(set-quality! 8)
(set-resolution! 15)

;(box [-5 -5 -0.5] [5 5 0])
(define arch
  (let((h 2.5) (r 1.5))
    (union (rectangle [(- r) 0] [r h])
      (circle r [0 h]))))
      
(define subpillar
  (union
      (box [5.7 -0.2 6] [6.8 0.9 8.3])
      (scale-z (sphere 0.5 [6.25 0.35 8.1])
        2 8.3)
      (cylinder-z 0.1 0.8 [6.25 0.35 8.3]) 
      (sphere 0.1 [6.25 0.35 9.1])
      ))
      
(define railing-cut
    (box [0.1 0 6.9] [0.9 1 7.6]))
(define railing-multicut
  (union
    (move railing-cut [2.9 0 0])
    (move railing-cut [3.85 0 0])
    (move railing-cut [4.8 0 0])))

(define arches-tight
  (sequence
    (difference
      (rectangle [0 0] [12.5 6])
      (move arch [2 0])
      (move arch [5.5 0])
      (move arch [9 0])
      (move arch [12.5 0])
    )
    (extrude-z 0 0.5)
  (reflect-yz)
  (reflect-xy)
  (move [6 0])))

(define stair-wall-cutouts
  (union
    (difference
      (box [14 -2.6 5] [14.8 -0.6 20])
      (lambda-shape (x y z)
        (- (- z 5) (* (- x 14) 0.8))
      ))
    (box [14 1 6.6] [14.8 9.7 9.5])
    (box [14 1.2 6.8] [15 9.5 9.3])))

(define stairs-angle
  (difference
    (lambda-shape (x y z)
      (- (* 3 (- z 4)) (- y -2.8)))
    (lambda-shape (x y z) (- x 14.5))
    (lambda-shape (x y z) (- 18 x))
))
    
(define stairs
  (sequence
    (move stairs-angle [0 0 0.5])
    (difference
      (move stairs-angle [0 0 -0.3])
      (lambda-shape (x y z) (- 3.7 z))
      (lambda-shape (x y z) (+ y 5))
          (lambda-shape (x y z) (+ y 5))
      )
    (union
      (box [16 -3.7 3.4] [18 0.3 3.7])
      (box [0 8.9 6.3] [18 12.5 6.6])
      (difference
        (move
          (difference (move stairs-angle [0 0 0.5])
            (move stairs-angle [0 0 -0.3]))
            [0 5.25 0.45])
          (lambda-shape (x y z) (- z 3.4))
          (lambda-shape (x y z) (- 6.6 z)))
      )
    (difference    
      (move 
        (apply union (map
          (lambda (i)
           (lambda-shape (x y z)
              (max (- (* i 0.171) z)
                   (- y (/ i 2))))) 
          (iota 20)))
          [0 0.5 3.7])
      (move 
        (apply union (map
          (lambda (i)
           (lambda-shape (x y z)
              (max (- (* i 0.171) z)
                   (- y (/ i 2))))) 
          (iota 5)))
          [0 -5.7 2.85])
          ))
)

(define stair-wall
  (sequence
    (box [14.5 -5 0] [16 12.5 10])
    (difference
      (box [14 -10 7] [18 -0.6 20])
      (box [14 -5 4] [18 -4 20])
      (box [12 10 6.6] [18 14 10])
      stair-wall-cutouts
      (reflect-x stair-wall-cutouts 15.25)
    )
    (union
      (box [14.7 -3 6.8] [15.8 -1.9 7])
      (box [14.3 -0.8 10] [16.2 10 10.2])
      (difference
        (pyramid-z [14.5 -4] [16 -2.6] 6.8 4)
        (lambda-shape (x y z) (- 7.4 z)))
      (sequence
        (circle 1.5)
        (extrude-z -0.45 0.45)
        (reflect-xz)
        (move [15.25 -0.6 6.8]))
      (sequence
        (difference (circle 1.5) (circle 1.3))
        (extrude-z -0.55 0.55)
        (reflect-xz)
        (difference (lambda-shape (x y z) z))
        (move [15.25 -0.6 6.8]))

    )))
(define arches
  (sequence
    (difference
      (rectangle [0 0] [15 7.9])
      (move arch [0 0])
      (move arch [3.5 0])
      (move arch [9 0])
      (move arch [12.5 0])
    )
    (extrude-z 0 0.5)
  (reflect-yz)
  (difference
    (box [0 -0.1 6.8]
         [14.5 0.1 7.7])
    railing-cut
    (move railing-cut [0.95 0 0])
    railing-multicut
    (move railing-multicut [3.8 0 0])
    (move railing-multicut [7.6 0 0])
  )
  (union
    ;; Floor
    (box [0 -5 -0.5] [16 12.5 0])

    ;; Balcony
    (box [0 -0.4 5.5] [16 12.5 5.8])
    (box [0 -0.3 5.8] [16 12.5 6.6])
    (box [0 -0.4 6.4] [16 0 6.7])

    ;; Pillar
    (box [5.7 -0.2 0] [6.8 0.9 10])
    (scale-z (sphere 0.5 [6.25 0.35 9.8])
      2 10)
    (cylinder-z 0.1 0.9 [6.25 0.35 10]) 
    (sphere 0.1 [6.25 0.35 10.9])

    ;; Subpillars
    (move subpillar [-3.8 0 0])
    (move subpillar [3.8 0 0])
    (move subpillar [7.6 0 0])

    ;; Stair wall
    stair-wall

    ;; Internal arches
    arches-tight
    (move arches-tight [8.2 0 0])

    stairs
  )

))
(define s 0.05)
(sequence
  (union arches
    (reflect-x arches)
    ;(reflect-y arches 12.5)
    )
  (rotate-z -0.2)
  (rotate-x -1)
  (scale-xyz [s s s])
)
  
