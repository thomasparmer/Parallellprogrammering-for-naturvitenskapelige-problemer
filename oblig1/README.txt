Image denoising refers to the removal of noises from a noise-contaminated image, 
such that the “smoothed” image more closely resembles the original noise-free image.

The purpose of this assignment is to get familiarized with the following important tasks:
1.  Translation of mathematical formulas to a working code.
2.  Compilation of existing C source codes into an external library.
3.  Implementimg a denoising algorithm.
4.  Parallelization of the denoising algorithm via MPI programming.


When you are in the serial/ or parallel/ directories you may
compile the programs by typing "make" in the terminal.
The two programs accept the following parameters (in order)

- Serial
$ ./program number_of_iterations kappa_value infile outfile
- Parallel
$ mpirun -np [cores] ./program number_of_iterations kappa_value infile outfile


if you want to create a compressed file containing all the files type 
"make delivery" in the terminal. A tarball with the files will then be created.