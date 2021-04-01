/*
 * This file is part of the MathLinear library
 * Usage: Provide an example use of the library
 * 
 * Version 1.0.0
 * Developed by Evan https://github.com/halsw
 *
 * Dependencies: MathFixed library (https://github.com/halsw/MathFixed)
 *               FixedPoints library (https://github.com/Pharap/FixedPointsArduino)
 *               Quaternions library (https://github.com/halsw/Quaternions)
 *               
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <FixedPoints.h>
#include <FixedPointsCommon.h>

//Define the type used for calls to MathFixed functions is not explicitly defined (defaults to float if ommited)
#define TFixed SQ7x8

#include "MathFixed.h"
#include "Quaternions.h"
#include "MathLinear.h"

//Define Fixed Point matrix in program memory using fx() macro
const TFixed  pgm3x3[9] PROGMEM = {
  fx(1.0), fx(2.0), fx(3.0),
  fx(2.0), fx(4.0), fx(5.0),
  fx(3.0), fx(5.0), fx(6.0)
};

#define PERIOD_MS 1500

#define PROFILE(X,V) static float x##X = 0.0;\
  static uint16_t n##X = 1;\
  unsigned long t##X = micros();\
  V;\
  t##X = micros() - t##X;\
  x##X += (t##X - x##X)/ n##X++;\
  Serial.print(x##X/1000.0,3);\
  Serial.println("ms " #X ": " #V)

void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(0)); //assuming A0 is not connected
  fxalmost<TFixed>(64); 
}

void loop() {
  TFixed c;
  Vector<3> v, w; //Default initialized vectors initially hold a NaN value
  Matrix<3> x=Matrix<3>(pgm3x3,true); //true in initialization means read from program memory
  Matrix<3> y(1,  2, -1,
               3,  1,  2,
               4, -2,  1); //Direct initialization
  Matrix<4> a; //Default initialized matrices initially hold a NaN value
  Complex<TFixed> z(1.1,1.1);
  
  x = pgm3x3; //Initialization with array assignment works only for PROGMEM arrays

  w << 2.0 << 1.0 << 3.0; // Stream like initialization (uses push_back method)

  w = {2, 1, 3}; //Literal array initialization
  
  v = y[2]; //Initialization from matrix column

  v = y(1); //Initialization from matrix row

  y = a.sub(1,1); //Initialization from sub matrix

  //Be warned that the use stream extensions eats up program memory
  Serial << "Progmem arrays are implicitly considered vectors or matrices depending on usage, so be careful" << endl;
  x += pgm3x3;
  Serial << setw(6,2) << v << eol(2) << setw(6,2) << x << eol(2);

  c = fxrandom((TFixed)10.0);
  Serial << "Vector basic operations ie multiplication "  << c << " * " << setw(6,2) << v << " = " << c*v << endl;
  Serial << "Matrix basic operations ie multiplication:" << endl;
  Serial << setw(6,2) << v << " * " << indent(8) << x << " = " << v*x << eol(2);

  fxrandom(x);
  fxrandom(v);
  Serial << "Linear system  A" << indent(6,2) << x << " * X = " << v << endl;
  Serial << setw(6,2) << "<=> X = " << x.solve(v) << eol(2);

  x = pgm3x3;
  x /= (TFixed)12;
  Serial << "X=" << indent(6,2) << x << endl;
  Serial << "Symmetric matrix eigenvalues  "  << setw(6,2) << x.eigenvals() << endl;
  x.eigenvecs();
  Serial << "Symmetric matrix eigenvectors "  << indent(6,2) << x << endl;
  
  x = pgm3x3;
  x /= (TFixed)12;
  Serial << "X=" << indent(6,2) << x << " matrix functions ie "<< endl;
  Serial << "exp(X)=" << indent(14,6) << fxexp(x) << eol(2);

  Serial << "QR decomposition of x=" << indent(6,2) << x << endl;
  y = x.QRgivens();
  Serial << "is: " << indent(6,2) << x << " *" << endl << y << endl;
  
  fxrandom(x);
  fxrandom(v);
  fxrandom(w);
  fxrandom(y, w); //random y with known eigenvalues(w)

  Serial << "\nExecution times for some operations:\n";
  PROFILE(Vector_Inner_Product,v * w);
  PROFILE(Vector_Outer_Product,v ^ w);
  PROFILE(Vector_Complex_Product, v * z );
  PROFILE(Matrix_Transpose,!x);
  PROFILE(Matrix_Add, x + y );
  PROFILE(Matrix_Subtract, x - y );
  PROFILE(Matrix_Multiply, x * y );
  PROFILE(Matrix_Vector_Multiply, x * v );
  PROFILE(Matrix_Divide, x / y );
  PROFILE(Matrix_Negate, -x );
  PROFILE(Matrix_Inverse, x.inv());
  PROFILE(Matrix_Determinant, x.det());
  PROFILE(Linear_Solve, v.solve(x));
  PROFILE(Matrix_QR_Decomposition, x.QRgivens() );
  PROFILE(Matrix_Exponential, fxexp(y) );
  PROFILE(Eigenvalues, y.eigenvals());
  PROFILE(Eigenvectors, y.eigenvecs());

  delay(PERIOD_MS);
}
