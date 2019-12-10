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
      (cylinder-z 0.1 1 [6.25 0.35 8.3]) 
      (sphere 0.1 [6.25 0.35 9.3])
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
      (rectangle [0 0] [10 6])
      (move arch [2 0])
      (move arch [5.5 0])
      (move arch [9 0])
    )
    (extrude-z 0 0.5)
  (reflect-yz)
  (reflect-xy)
  (move [6 0])))

(define stair-wall
  (sequence
    (box [14.5 -5 0] [16 10 10])
    (difference
      (box [14 1 6.6] [14.8 9.7 9.5])
      (box [14 1.2 6.8] [15 9.5 9.3])
      (box [14 -10 6.8] [18 -0.6 20])
      (box [14 -5 4] [18 -4 20])
      (box [14 -2.6 4.5] [14.8 -0.6 20])
    )
    (union
      (box [14.3 -0.8 10] [16.2 9.6 10.2])
      (difference
        (pyramid-z [14.5 -4] [16 -2.6] 6.8 4)
        (lambda-shape (x y z) (- 7.4 z)))
    
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
    (box [0 -5 -0.5] [16 10 0])

    ;; Balcony
    (box [0 -0.4 5.5] [16 10 5.8])
    (box [0 -0.3 5.8] [16 10 6.4])
    (box [0 -0.4 6.4] [16 0 6.7])
    
    ;; Pillar
    (box [5.7 -0.2 0] [6.8 0.9 10])
    (scale-z (sphere 0.5 [6.25 0.35 9.8])
      2 10)
    (cylinder-z 0.1 1 [6.25 0.35 10]) 
    (sphere 0.1 [6.25 0.35 11])
    
    ;; Subpillars
    (move subpillar [-3.8 0 0])
    (move subpillar [3.8 0 0])
        (move subpillar [7.6 0 0])

    ;; Stair wall

    
    ;; Internal arches
    arches-tight
    (move arches-tight [8.2 0 0])
      stair-wall
  )

))
arches




