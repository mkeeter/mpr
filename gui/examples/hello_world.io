(sequence
  (text "hello,\nworld")
  (move [-1.6 0.1])
  (scale-xyz [0.5 0.5 1])
  (extrude-z -0.5 0.5)
  (intersection
    (scale-z (sphere 2) 0.5))
  (rotate-y -0.3)
  (rotate-x -0.1)
  )
