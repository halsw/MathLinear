# [Linear Algebra Template Library](https://github.com/halsw/MathLinear) for Arduino and Teensy
A library with template classes and functions for vectors and matrices

The library provides mixed basic operations and mixed functions with support for vectors, matrices, scalars, complex numbers and quaternions as well as limited support (only for symmetric matrices) for eigenvalues and eigenvectors (and therefore for dependent on eigenvalue functions). The functions are overloaded versions of the [MathFixed](https://github.com/halsw/MathFixed) library so it can be used with double, float or any fixed point number type.

## Example file
The included .ino file provides a limited example of the full capabilities and uses fixed point numbers with 7 integer bits and 8 fractional. You may change the type of numbers used by changing the statement *#define TFixed*. If *TFixed* definition is ommited it defaults to float. Note that 32-bit fixed point types don't offer a speed advantage over builtin floats unless usage is mainly for additions/subtractons

To compile the .ino file you must first install the [MathFixed](https://github.com/halsw/MathFixed),  **FixedPoints** and [Quaternions](https://github.com/halsw/Quaternions)  libraries

## Warning
The use of high dimension matrices is prohibitive for Arduino because the allocation of temporary  storage may exhaust available RAM leading to unexpected behavior

 
