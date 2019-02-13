# overview_ai

Two programs. ``generate_variations.py`` will randomly hue shift, scale, rotate, affine warp, and perspective warp an input image. 
``rectify_variations.py`` will detect the corners of a colorful, planar, rectangular object that has undergone those transforms and revert it to a neutral, directly overhead view. It also displays detected edges (Canny) and a subtraction comparison with the previous detected edges. Ideally, this would be near zero, but it appears to not be a good approach here. 


First run ``generate_variations``, which will default to creating 25 variations of ``pcb.jpg`` in ``variations``. Then run ``rectify_variations`` to see them corrected. Press ``esc`` to quit the program, or press ``f`` to see the next image. 



Known limitations:
* There's no input UI in either program, to change from defaults edit the python.
* The object is detected with a saturation mask. It must be much more colorful than the background. It'd be easy to change this to a brightness/value mask.
* A 'perfect storm' of random paramaters could result in transforms that exceed the padding. If this happens, the corner detection will of course fail.
* The corner sorting isn't truly robust right now. It's possible (though unlikely with defaults) to create a projective transform that will cause it to double up on one corner point.
* The color changes are limited to hue to avoid interfering with the detection.
