(set-bounds! [-1 -1 -1] [1 1 1])
(set-quality! 8)
(set-resolution! 200)
(sequence
   (text (string-append
    "I'll break my staff\n"
    "Bury it certain fathoms\n"
    "   in the earth\n"
    "And deeper than did\n"
    "   ever plummet sound\n"
    "I'll drown my book!"))
   (extrude-z -0.1 0.1)
   (move [-6.5 2.5])
   (scale-x 0.125)
   (scale-y 0.125)
)