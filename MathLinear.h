/*
 * This file is part of the MathLinear library
 * Usage: A template library for the implementation of linear algebra
 *         math functions with vectors and square matrices for use 
 *         with fixed & floating point types
 * Version 1.0.0
 * Developed by Evan https://github.com/halsw
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
 * 
 * Classes:
 *   Vector a vector container
 *   CVector a pointer to vector coefficients
 *   Matrix square matrix container
 *   GMatrix general matrix container
 *   
 * Functions:
 *   fxsize() gets number of real numbers needed for representation
 *   fxnan() gets the representation of NaN
 *   fxisnan() tests if argument is not a vector/matrix
 *   fxisinf() tests if argument is infinity, but here just a copy of fxisnan()
 *   fxabs() the absolute value of individual coefficients
 *   fxequ() almost equal comparison (set by fxalmost tollerance)
 *   fxsign() the sign (-1, 1)
 *   fxcopysign() copies the sign of second argument to the first one
 *   fxmax() the greater in norm of two vectors/matrices
 *   fxmin() the lesser in norm of two vectors/matrices
 *   fxsq() the square of a matrix
 *   fxsqrt() the square root of a matrix
 *   fxcbrt() the cubic root of a matrix
 *   fxrandom() get a random vector/matrix
 *   fxsin() the sine of a matrix
 *   fxcos() the cosine of a matrix
 *   fxtan() the tangent of a matrix
 *   fxcot() the cotangent of a matrix
 *   fxatan() the inverse tangent of a matrix
 *   fxasin() the inverse sine of a matrix
 *   fxacos() the inverse cosine of a matrix
 *   fxexp() the natural exponential of a matrix
 *   fxlog() the natural logarithm of a matrix
 *   fxpow() raise a matrix to given exponent
 *   fxsinh() the hyperbolic sine of a matrix
 *   fxsinh() the hyperbolic cosine of a matrix
 *   fxtanh() the hyperbolic tangent of a matrix
 *   fxconj() the conjugate of a vector/matrix
 *   
 * fxm namespace:  
 *   fxm::func0() helper function to evaluate matrix functions one matrix parameter
 *   fxm::func1() helper function to evaluate matrix functions one matrix and one scalar parameter
 *   fxm::isqrt() constexpr helper function to get the integer square root of a dimension
 *   
 * Defines:
 *   mfx() macro converts Row, Column pair to element index for square matrices of size N
 *   FXM_DIMENSION the default dimension used by the library
 *   
 * Warning:  
 *   The use of high dimension matrices is prohibitive for Arduino because the allocation of temporary
 *   storage may exhaust available RAM leading to unexpected behavior
 */

#ifndef MATHLINEAR_H
#define MATHLINEAR_H

#include "MathFixed.h"

#ifndef FXM_DIMENSION
#define FXM_DIMENSION 3
#endif

#define mfx(R,C) (N*(R)+(C))

template <int N, class T> class Matrix;
template <int M, int N, class T> class CVector;
template <int M, int N, class T> class GMatrix;

template <int N=FXM_DIMENSION, class T=TFixed>
class Vector {
  friend class CVector<N,N,T>;
  friend class CVector<1,N,T>;
  friend class Matrix<N,T>;
protected:
  T m[N];
public:
  Vector() { fxnan(m[0]); memset(m+1,0,(N-1)*sizeof(T)); }
  template <int M> Vector(CVector<M,N,T>& x) { for (uint8_t i=0, j=0; i<N; i++, j+=M) m[i] = x.m[j]; }
  Vector(T x): Vector() {m[0] = x; }
  Vector(T x, bool fill) : Vector(x) {if (fill) for (uint8_t i=1; i<N; i++) m[i] = (T)x;}
  Vector(const T x[N], T a) : Vector() {if (a==1) memcpy(m,x,N*sizeof(T)); else for (uint8_t i=0; i<N; i++) m[i] = a*x[i];}
  Vector(const T x[N], bool progmem) : Vector() { if (progmem) memcpy_P(m,x,N*sizeof(T)); else memcpy(m,x,N*sizeof(T)); }
  Vector(uint8_t i, T x) : Vector() {if (i<N) m[i] = x; else fxnan( m );}
  template <class C>   Vector(C x, C y, ...):Vector() { va_list vl; m[0] = (T)x; m[1] = (T)y; va_start(vl,y); for (uint8_t i=2; i<N; i++) m[i] = (T)va_arg(vl,C); va_end(vl); }
  
  void push_back(const T& x) { memmove(m, m+1,(N-1)*sizeof(T)); m[N-1] = x; }
  
  void pop_back(const T& x) { memmove(m+1, m,(N-1)*sizeof(T)); fxnan(m[0]); }

  bool isnan() const { for (uint8_t i=0; i<N; i++) if ( fxisnan(m[i]) ) return true; return false;}

  bool iszero() const { for (uint8_t i=0; i<N; i++) if ( !fxequ( m[i] ) ) return false; return true;}

  //Euclidean norm
  T norm() const { return fxnorm(m,N); }

  virtual T norm1() { T r=(T)0.0; for (uint8_t i=0; i<N; i++) if ( fxisnan(m[i]) ) return m[i]; else r+=fxabs(m[i]); return r; }

  Vector& rand( T range = 1.0, int16_t dec=100 ) {range /= (T)dec; for (uint8_t i=0; i<N; i++) m[i] = range*((T)random(-dec, dec)); return *this;}  

  Vector& transpose() { if (fxsize(m[0]) > 1) for (uint8_t i=0; i<N; i++) m[i] = fxconj(m[i]); return *this;}

  Vector& project(const Vector &x) { return *this = (*this * x) / (x * x); }
  
  Vector& normalize() { return *this /= norm(); }

  Vector& swap( Vector &x ) { T v; for (uint8_t i=0; i<N; i++) { v=m[i]; m[i] = x.m[i]; x.m[i] = v; }  return *this; }

  template <int M> Vector& swap( CVector<M,N,T> &x ) { x.swap(*this);  return *this; }

  Vector<N-1,T> sub(uint8_t n) { uint8_t i, j; Vector<N-1,T> r; for (i=0, j=0; i<N; i++) { if ( i == n) continue; r[j++] = m[i]; } return r; }

  Vector& solve(const Matrix<N,T>& m) { return *this = m.solve(*this); }

  Vector& solve(const T a[N*N]) { Matrix<N,T>& m(a); return *this = m.solve(this); }

  Vector& operator = (const Vector &x) { if (this != &x) memcpy(m, x.m,N*sizeof(T)); return *this; }

  template <int M, int K> Vector& operator = (const CVector<M,K,T> &x) { uint8_t i, j; for (uint8_t i=0, j=0; i<N && j<K*M; i++, j+=M) m[i]=x[j]; for (; i<N; i++) m[i]=0.0; return *this; }

  Vector& operator = (T x) { m[0] = x; memset(m+1,0,(N-1)*sizeof(T)); return *this; }

  Vector& operator = (const T x[]) { for (uint8_t i=0; i<N; i++) m[i] = fxpgmread<T>(x++); return *this; }

  template <int M> Vector& operator = (const CVector<M,N,T> &x) { for (uint8_t i=0; i<N; i++) m[i] = x[i]; return *this; }

  explicit operator T* () const {return m;}

  T& operator[](int8_t i) {if (i<N) return m[i]; fxnan(m[0]); return m[0];}

  Vector& operator << (const T &x) { push_back(x); return *this; }  

  inline bool operator == (const Vector<N,T> &x) { if (isnan() || x.isnan()) return false; for (uint8_t i=0; i<N; i++) if (m[i]!=x.m[i]) return false; return true;}

  template <int M> inline bool operator == (const CVector<M,N,T> &x) { return *this == Vector(x);}

  inline bool operator != (const Vector<N,T> &x) { return !(*this == x); }

  template <int M> inline bool operator != (const CVector<M,N,T> &x) { return !(this == Vector(x)); }

  inline bool operator > (const Vector<N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me > xx; }

  template <int M> inline bool operator > (const CVector<M,N,T> &x) { return *this > Vector(x); }

  inline bool operator < (const Vector<N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me < xx; }

  template <int M> inline bool operator < (const CVector<M,N,T> &x) { return *this < Vector(x); }

  inline bool operator >= (const Vector<N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me >= xx; }

  template <int M> inline bool operator >= (const CVector<M,N,T> &x) { return *this >= Vector(x); }

  inline bool operator <= (const Vector<N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me <= xx; }

  template <int M> inline bool operator <= (const CVector<M,N,T> &x) { return *this <= Vector(x); }

  inline bool operator == (const T &x) { if (isnan() || fxisnan(x)) return false; return norm() == x;}

  inline bool operator != (const T &x) { return !(*this == x);}

  inline bool operator > (const T &x) { T me = norm(), xx=fxabs(x); return !fxisnan(me) && !fxisnan(xx) && me > xx; }

  inline bool operator < (const T &x) { T me = norm(), xx=fxabs(x); return !fxisnan(me) && !fxisnan(xx) && me < xx; }

  inline bool operator >= (const T &x) { T me = norm(), xx=fxabs(x); return !fxisnan(me) && !fxisnan(xx) && me >= xx; }

  inline bool operator <= (const T &x) { T me = norm(), xx=fxabs(x); return !fxisnan(me) && !fxisnan(xx) && me <= xx; }

  Vector& operator+() const {return *this;}

  Vector operator-() const { return Vector(m,(T)-1.0);}

  Vector operator!() const { return Vector(m,false).transpose();}

  Vector& operator+=(const Vector &x) { for (uint8_t i=0; i<N; i++) if (fxisnan(m[i]) || fxisnan(x.m[i])) fxnan(m[0]); else m[i]+=x.m[i]; return*this; }

  template <int M> Vector& operator+=(const CVector<M,N,T> &x) {  return *this += Vector(x); }

  Vector& operator+=(const T x[]) { for (uint8_t i=0; i<N; i++) { T v=fxpgmread<T>(x+i); if (fxisnan(m[i])) fxnan(m[0]); else m[i]+=v; } return*this; }

  Vector inline operator+(const Vector &x) { Vector<N,T> r(m,false); return r += x ; }

  template <int M> Vector inline operator+(const CVector<M,N,T> &x) { return *this + Vector(x) ; }

  Vector inline operator+(const T x[]) { Vector<N,T> r(m,false); return r += x ; }

  Vector& operator-=(const Vector &x) { for (uint8_t i=0; i<N; i++) if (fxisnan(m[i]) || fxisnan(x.m[i])) fxnan(m[0]); else m[i]-=x.m[i]; return*this; }

  template <int M> Vector& operator-=(const CVector<M,N,T> &x) {  return *this -= Vector(x); }

  Vector& operator-=(const T x[]) { for (uint8_t i=0; i<N; i++) { T v=fxpgmread<T>(x+i); if (fxisnan(m[i])) fxnan(m[0]); else m[i]-=v; } return*this; }

  Vector inline operator-(const Vector &x) { Vector<N,T> r(m,false); return r -= x;}

  template <int M> Vector inline operator-(const CVector<M,N,T> &x) { return *this - Vector(x) ; }

  Vector inline operator-(const T x[]) { Vector<N,T> r(m); return r -= x;}

  T operator*(const Vector &x) { T r=0; for (uint8_t i=0; i<N; i++) if (fxisnan(m[i]) || fxisnan(x.m[i])) return fxnan(m[0]); else r+=m[i]*fxconj(x.m[i]); return r; }

  template <int M> T inline operator*(const CVector<M,N,T> &x) { return *this * Vector(x) ; }

  Vector operator*(const Matrix<N,T> &x) { Vector r(0); for (uint8_t i=0; i<N; i++) if (fxisnan(m[i])) {fxnan(r.m[0]); break;} else for (uint8_t j=0; j<N; j++) if (fxisnan(x.m[j*N+i])) {fxnan(m[0]); break;} else r.m[i]+=m[i]*fxconj(x.m[j*N+i]); return r; }

  template <int M> Vector<M,T> operator*(const GMatrix<N,M,T> &x) { Vector r(0); for (uint8_t i=0; i<M; i++) if (fxisnan(m[i])) {fxnan(r.m[0]); break;} else for (uint8_t j=0; j<N; j++) if (fxisnan(x.m[j*M+i])) {fxnan(m[0]); break;} else r.m[i]+=m[i]*fxconj(x.m[j*M+i]); return r; }

  Vector operator*(const T x[N*N]) { Vector r(0); for (uint8_t i=0; i<N; i++) if (fxisnan(m[i])) {fxnan(r.m[0]); break;} else for (uint8_t j=0; j<N; j++) {T v=fxpgmread<T>(x+j*N+i); r.m[i]+=m[i]*fxconj(v);} return r; }

  Vector& operator*=(const Matrix<N,T> &x) {  return *this = *this * x; }

  Vector& operator*=(const T x[N*N]) { return *this = *this * x; }

  Matrix<N,T> operator^(const Vector &x) { Matrix<N,T> r; for (uint8_t i=0; i<N; i++) if (fxisnan(m[i])) {fxnan(r.m[0]); break;} else for (uint8_t j=0; j<N; j++) if( fxisnan(x.m[j])) {fxnan(m[0]); break;} else r.m[i*N+j]=m[i]*fxconj(x.m[j]); return r; }

  template <int M> Matrix<N,T> inline operator^(const CVector<M,N,T> &x) { return *this ^ Vector(x); }

  Vector inline operator*(T x) const { if ( isnan() || fxisnan(x) ) return Vector(); return Vector(m,x);}

  template <class C> Vector<N,C> inline operator*(const C& x) const { Vector<N,C> r; if ( !isnan() && !fxisnan(x) ) for (uint8_t i=0; i<N; i++) r[i] = m[i] * x ; return r;}

  Vector& operator*=(T x) { if ( isnan() ) return *this; for (uint8_t i=0; i<N; i++) if (fxisnan(x)) fxnan(m[0]); else m[i]*=x; return *this;}

  Vector inline operator/(T x) const { if ( isnan() || fxisnan(x) ) return Vector(); return Vector(m,1/x);}

  Vector& operator/=(T x) { if ( isnan() ) return *this; for (uint8_t i=0; i<N; i++) if (fxisnan(x)) fxnan(m[0]); else m[i]/=x; return *this;}  
};

template <int N, class T> inline Vector<N,T> operator*(T x, const Vector<N,T> &y) { return y*x;}
template <int N, class T> inline Vector<N,T> operator*(const T &x, const T* y) {  return Vector<N,T>(y,true) * x;}
template <int N, class T> inline bool operator==(const T &x, const Vector<N,T> &y) { return y==x;}
template <int N, class T> inline bool operator!=(const T &x, const Vector<N,T> &y) { return y!=x;}
template <int N, class T> inline bool operator>(const T &x, const Vector<N,T> &y) { return y<x;}
template <int N, class T> inline bool operator<(const T &x, const Vector<N,T> &y) { return y>x;}
template <int N, class T> inline bool operator>=(const T &x, const Vector<N,T> &y) { return y<=x;}
template <int N, class T> inline bool operator<=(const T &x, const Vector<N,T> &y) { return y>=x;}

template <int M=1,int N=FXM_DIMENSION, class T=TFixed>
class CVector {
  friend class Vector<N,T>;
  friend class Matrix<N,T>;
  friend class GMatrix<M,N,T>;
  friend class GMatrix<N,M,T>;
protected:
  T* m;
public:
  CVector(T &x): m(&x) {}

  CVector(Matrix<N,T> &x, uint8_t column): m(x.m + column) {if (M!=N) fxnan(m[0]);}

  template <int K> CVector(GMatrix<N,K,T> &x, uint8_t column): m(x.m + column) {}

  CVector(uint8_t row, Matrix<N,T> &x): m(x.m + N*row) {if (M!=1) fxnan(m[0]);}

  template <int K> CVector(uint8_t row, GMatrix<K,N,T> &x): m(x.m + N*row) {if (M!=1) fxnan(m[0]);}

  void push_back(const T& x) { uint8_t i, j; for (i=0, j=M; j<M*N; i+=M, j+=M); m[i] = m[j]; m[i] = x; }
  
  void pop_back(const T& x) { uint8_t i, j=N*M-1; for (i=j-M; i>0; i-=M, j-=M); m[i] = m[i-M]; fxnan(m[0]); }

  bool isnan() const { for (uint8_t i=0; i<M*N; i+=M) if ( fxisnan(m[i]) ) return true; return false;}

  bool iszero() const { for (uint8_t i=0; i<M*N; i+=M) if ( !fxequ( m[i] ) ) return false; return true;}

  //Euclidean norm
  T norm() const { return Vector<N,T>(*this).norm(); }

  virtual T norm1() { return Vector<N,T>(*this).norm1(); }

  CVector& rand( T range = 1.0, int16_t dec=100 ) {range /= (T)dec; for (uint8_t i=0; i<M*N; i+=M) m[i] = range*random(-dec, dec); return *this;}  

  CVector& project( const Vector<N,T> &x) { return *this = (*this * x) / (x * x); }
  
  CVector& normalize() { return *this /= *this * *this; }

  CVector& swap( Vector<N,T> &x ) { T v; for (uint8_t i=0, j=0; i<N; i++, j+=M) { v=m[j]; m[j] = x.m[i]; x.m[i] = v; }  return *this; }

  template <int K> CVector& swap( CVector<K,N,T> &x ) { T v; for (uint8_t i=0, j=0; i<M*N; i+=M, j+=K) { v=m[i]; m[i] = x.m[j]; x.m[j] = v; }  return *this; }

  Vector<N-1,T> sub(uint8_t n) { uint8_t i, j; Vector<N-1,T> r; for (i=0, j=0; i<M*N; i+=M) { if ( i == n) continue; r[j++] = m[i]; } return r; }

  template <int K, int L> CVector& operator = (const CVector<K,L,T> &x) { uint8_t i, j;  if (this != &x) for (i=0, j=0; i<M*N && j<L*N ; i+=M, j+=K) m[i] = x.m[j]; for ( ; i<M*N; i+=M) m[i] = 0.0; return *this; }

  CVector& operator = (const Vector<N,T> &x) { uint8_t i, j; for (i=0, j=0; j<N; i+=M, j++) m[i] = x.m[j]; return *this; }

  CVector& operator = (const T x[]) { for (uint8_t i=0; i<M*N; i+=M) m[i] = fxpgmread<T>(x++); return *this; }

  T& operator[](int8_t i) {if (i<N) return m[i*M]; fxnan(m[0]); return m[0];}

  CVector& operator << (const T &x) { push_back(x); return *this; }  

  inline bool operator == (const Vector<N,T> &x) { if (isnan() || x.isnan()) return false; for ( uint8_t i=0, j=0; i<N; i++, j+=M) if (m[j]!=x.m[i]) return false; return true;}

  template <int K> inline bool operator == (const CVector<K,N,T> &x) { if (isnan() || x.isnan()) return false; for ( uint8_t i=0, j=0; i<N; i+=M, j+=K) if (m[i]!=x.m[j]) return false; return true;}

  inline bool operator != (const Vector<N,T> &x) { return !(*this == x); }

  template <int K> inline bool operator != (const CVector<K,N,T> &x) { return !(*this == x); }

  inline bool operator > (const Vector<N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me > xx; }

  template <int K> inline bool operator > (const CVector<K,N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me > xx; }

  inline bool operator < (const Vector<N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me < xx; }

  template <int K> inline bool operator < (const CVector<K,N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me < xx; }

  inline bool operator >= (const Vector<N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me >= xx; }

  template <int K> inline bool operator >= (const CVector<K,N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me >= xx; }

  inline bool operator <= (const Vector<N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me <= xx; }

  template <int K> inline bool operator <= (const CVector<K,N,T> &x) { T me = norm(), xx=x.norm(); return !fxisnan(me) && !fxisnan(xx) && me <= xx; }

  inline bool operator == (const T &x) { if (isnan() || fxisnan(x)) return false; return norm() == x;}

  inline bool operator != (const T &x) { return !(*this == x);}

  inline bool operator > (const T &x) { T me = norm(), xx=fxabs(x); return !fxisnan(me) && !fxisnan(xx) && me > xx; }

  inline bool operator < (const T &x) { T me = norm(), xx=fxabs(x); return !fxisnan(me) && !fxisnan(xx) && me < xx; }

  inline bool operator >= (const T &x) { T me = norm(), xx=fxabs(x); return !fxisnan(me) && !fxisnan(xx) && me >= xx; }

  inline bool operator <= (const T &x) { T me = norm(), xx=fxabs(x); return !fxisnan(me) && !fxisnan(xx) && me <= xx; }

  CVector& operator+() const {return *this;}

  CVector& operator-() const { for (uint8_t i=0; i<M*N; i+=M) m[i] = -m[i]; return *this;}

  Vector<N,T> inline operator!() const { return Vector<N,T>(*this).transpose();}

  template <int K> CVector& operator+=(const CVector<K,N,T> &x) { for ( uint8_t i=0, j=0; i<M*N; i+=M, j+=K) if (fxisnan(m[i]) || fxisnan(x.m[j])) fxnan(m[0]); else m[i]+=x.m[j]; return*this; }

  CVector& operator+=(const Vector<N,T> &x) { for ( uint8_t i=0, j=0; i<N; i++, j+=M) if (fxisnan(m[j]) || fxisnan(x.m[i])) fxnan(m[0]); else m[j]+=x.m[i]; return*this; }

  CVector& operator+=(const T* x) { for ( uint8_t i=0, j=0; i<N; i++, j+=M) { T v=fxpgmread<T>(x+i); if (fxisnan(m[j])) fxnan(m[0]); else m[j]+=v; } return*this; }

  template <int K> Vector<N,T> inline operator+(const CVector<K,N,T> &x) { return Vector<N,T>(*this) += x ; }

  Vector<N,T> inline operator+(const Vector<N,T> &x) { return Vector<N,T>(*this) += x ; }

  Vector<N,T> inline operator+(const T* x) { return Vector<N,T>(*this) += x ; }

  template <int K> CVector& operator-=(const CVector<K,N,T> &x) { for ( uint8_t i=0, j=0; i<M*N; i+=M, j+=K) if (fxisnan(m[i]) || fxisnan(x.m[j])) fxnan(m[0]); else m[i]-=x.m[j]; return*this; }

  CVector& operator-=(const Vector<N,T> &x) { for ( uint8_t i=0, j=0; i<N; i++, j+=M) if (fxisnan(m[j]) || fxisnan(x.m[i])) fxnan(m[0]); else m[j]-=x.m[i]; return*this; }

  CVector& operator-=(const T* x) { for ( uint8_t i=0, j=0; i<N; i++, j+=M) { T v=fxpgmread<T>(x+i); if (fxisnan(m[j])) fxnan(m[0]); else m[j]-=v; } return*this; }

  template <int K> Vector<N,T> inline operator-(const CVector<K,N,T> &x) { return Vector<N,T>(*this) -= x ; }

  Vector<N,T> inline operator-(const Vector<N,T> &x) { return Vector<N,T>(*this) -= x ; }

  Vector<N,T> inline operator-(const T* x) { return Vector<N,T>(*this) -= x ; }

  template <int K> T operator*(const CVector<K,N,T> &x) { T r=0; for ( uint8_t i=0, j=0; i<M*N; i+=M, j+=K) if (fxisnan(m[i]) || fxisnan(x.m[j])) return fxnan(m[0]); else r+=m[i]*fxconj(x.m[j]); return r; }

  T inline operator*(const Vector<N,T> &x) { return Vector<N,T>(*this) * x; }

  Vector<N,T> inline operator*(const Matrix<N,T> &x) { return Vector<N,T>(*this) * x; }

  template <int K> Vector<K,T> operator*(const GMatrix<N,K,T> &x) { return Vector<N,T>(*this) * x; }

  Vector<N,T> inline operator*(const T x[N*N]) { return Vector<N,T>(*this) * x; }

  CVector&  operator*=(const Matrix<N,T> &x) {  return *this = *this * x; }

  CVector&  operator*=(const T x[N*N]) { return *this = *this * x; }

  template <int K> Matrix<N,T> operator^(const CVector<K,N,T> &x) { Matrix<N,T> r; for (uint8_t i=0, k=0; i<N; i++, k+=M) if (fxisnan(m[k])) {fxnan(r.m[0]); break;} else for (uint8_t j=0, n=0; j<N; j++, n+=K) if( fxisnan(x.m[n])) {fxnan(m[0]); break;} else r.m[i*N+j]=m[k]*fxconj(x.m[n]); return r; }

  Matrix<N,T> inline operator^(const Vector<N,T> &x) { return Vector<N,T>(*this) ^ x ; }

  Vector<N,T> inline operator*(T x) { return Vector<N,T>(*this) * x;}

  CVector& operator*=(T x) { if ( isnan() ) return *this; for (uint8_t i=0; i<M*N; i+=M) if (fxisnan(x)) fxnan(m[0]); else m[i]*=x; return *this;}

  Vector<N,T> inline operator/(T x) { return Vector<N,T>(*this) / x;}

  CVector& operator/=(T x) { if ( isnan() ) return *this; for (uint8_t i=0; i<M*N; i+=M) if (fxisnan(x)) fxnan(m[0]); else m[i]/=x; return *this;}  
};

template <int M, int N, class T> inline Vector<N,T> operator*(const T &x, const CVector<M,N,T> &y) { return y*x;}
template <int M, int N, class T> inline bool operator==(const T &x, const CVector<M,N,T> &y) { return y==x;}
template <int M, int N, class T> inline bool operator!=(const T &x, const CVector<M,N,T> &y) { return y!=x;}
template <int M, int N, class T> inline bool operator>(const T &x, const CVector<M,N,T> &y) { return y<x;}
template <int M, int N, class T> inline bool operator<(const T &x, const CVector<M,N,T> &y) { return y>x;}
template <int M, int N, class T> inline bool operator>=(const T &x, const CVector<M,N,T> &y) { return y<=x;}
template <int M, int N, class T> inline bool operator<=(const T &x, const CVector<M,N,T> &y) { return y>=x;}

template <int N=FXM_DIMENSION, class T=TFixed>
class Matrix: public Vector<N*N,T> {
  friend class Vector<N,T>;
  friend class CVector<N,N,T>;
  friend class CVector<1,N,T>;
public:
  Matrix() : Vector<N*N,T>() {}
  Matrix(T x): Vector<N*N,T>() {for (uint8_t i=0; i<N*N; i+=N+1) this->m[i] = x; }
  Matrix(T x, bool fill) : Vector<N*N,T>() { for (uint8_t i=0; i<N*N; i += 1 + N*fill) this->m[i] = x;}
  Matrix(const T x[N*N], T a) : Vector<N*N,T>(x,a) {}
  Matrix(const T x[N*N], bool progmem) : Vector<N*N,T>(x, progmem)  {}
  Matrix(const T diagn[N], const T* sdiag, bool progmem=false) : Vector<N*N,T>() {for (uint8_t i=0; i<N; i++)  {this->m[(N+1)*i] = progmem ? fxpgmread<T>(diagn+i) : diagn[i]; if (i && sdiag) this->m[(N+1)*i-1] = this->m[(N+1)*i-N] = progmem ? fxpgmread<T>(sdiag+i-1) : sdiag[i-1];}}
  Matrix(const Vector<N,T>& x, bool diag) : Vector<N*N,T>() {uint8_t i,k; for (i=0, k=0; i<N*N; i+=N+diag, k++) this->m[i] = x.m[k];}
  Matrix(uint8_t c, const Vector<N,T>& x) : Vector<N*N,T>() {for (uint8_t i=c, k=0; i<N*N; i+=N, k++) this->m[i] = x.m[k];}
  Matrix(const Vector<N,T>& x, uint8_t l) : Vector<N*N,T>() {for (uint8_t i=0; i<N; i++) this->m[l*N+i] = x.m[i];}
  Matrix(uint8_t c, const T x[N]) : Vector<N*N,T>() {for (uint8_t i=c, k=0; i<N*N; i+=N, k++) this->m[i] = x[k];}
  Matrix(const T x[N], uint8_t l) : Vector<N*N,T>() {for (uint8_t i=0; i<N; i++) this->m[l*N+i] = x.m[i];}
  template <class C> Matrix(C x, C y, ...)  : Vector<N*N,T>() { va_list vl;  this->m[0] = (T)x; this->m[1] = (T)y; va_start(vl,y); for (uint8_t i=2; i<N*N; i++) this->m[i] = (T)va_arg(vl,C); va_end(vl); }

  bool isSymmetric() const { uint8_t i, j; for (j=N-2; j<N; j--) for (i=j+1; i<N; i--)if ( !fxequ(this->m[N*j+i], this->m[N*i+j]) ) return false; return !this->isnan();}

  bool isDiagonal() const { uint8_t i, j; for (j=0; j<N; j++) for (i=0; i<N; i++) if ( i!=j && !fxequ(this->m[N*j+i]) ) return false; return !this->isnan(); }

  bool isTridiagonal() const { uint8_t i, j; for (j=0; j<N; j++) for (i=0; i<N; i++) if ( (i<j ? j-i : i-j)>1 && !fxequ(this->m[N*j+i]) ) return false; return !this->isnan(); }

  bool isLowHessenberg() const { uint8_t i, j; for (j=0; j<N-2; j++) for (i=N-1; i>j+1; i--) if (!fxequ(this->m[N*i+j]) ) return false; return !this->isnan(); }

  bool isUppHessenberg() const { uint8_t i, j; for (i=0; i<N-2; i++) for (j=i+2; j<N-1; j++) if (!fxequ(this->m[N*i+j]) ) return false; return !this->isnan(); }

  bool isOrthogonal() const { 
    uint8_t i, j;
    Matrix x(this->m, false);
    if ( x.isnan() ) return false;
    x = x.transpose() * *this;
    for ( j=0; j<N; j++ ) {
      if ( !fxequ(x.m[j*(N+1)], (T)1.0) ) return false;
      for ( i=0; i<N; i++ )
        if ( i!=j && !fxequ(x[j*N+i]) ) return false;
    }
    return (true);
  };

  Matrix& fill(T v=0) {uint8_t i; for (i=0; i<N; i++) (*this)(i) = v; return *this;}

  Matrix& transpose() {T v; uint8_t i, j; for (j=0; j<N; j++) for (i=j+1; i<N; i++) {v=fxconj(this->m[N*j+i]); this->m[N*j+i]=fxconj(this->m[N*i+j]);this->m[N*i+j]=v;} return *this;}

  Matrix& project(Vector<N,T> &x) { for (uint8_t i=0; i<N; i++) (*this)[i].project(x); return *this; }
  
  Matrix& normalize() { for (uint8_t i=0; i<N; i++) (*this)[i].normalize(); return *this; }

  Matrix<N-1,T> sub(uint8_t r, uint8_t c) {
    uint8_t i, j;
    Matrix<N-1,T> x;
    for (i=0, j=0; i<N; i++) {
      if ( i == r) continue;
      x[j++] = (*this)[i].sub(c);
    }
    return x;
  }
    
  //1 norm
  virtual T norm1() { T r= (T)0.0; for (uint8_t i=0; i<N; i++) r=fxmax( col(i).norm1(), r ); return r; }

  //infinity norm
  T normi() {  T r= (T)0.0; for (uint8_t i=0; i<N; i++) r=fxmax( row(i).norm1(), r ); return r; }

  //Max norm
  inline T normax() { return Vector<N*N,T>::norm1(); }

  Matrix& rand( T range = 1.0, int16_t dec=100 ) {  Vector<N*N,T>::rand( range, dec ); return *this; }  

  T trace() { T r=0;  for (uint8_t i=0; i<N*N; i+=N+1) r+=this->m[i]; return r;  }

  CVector<N,N,T> col(uint8_t c) { return CVector<N,N,T>(*this, c); }
  
  Vector<N,T> getcol(int8_t c) { Vector<N,T> x; if (c>=N) return x; for (uint8_t i=0, j=c; i<N; i++, j+=N) x.m[i] = this->m[j]; return x;}

  void swapcol(uint8_t c1, uint8_t c2) {uint8_t i, j; T v; for (  i = c1, j = c2; i<N*N; i+=N, j+=N )  {v = this->m[i]; this->m[i] = this->m[j]; this->m[j] = v;}}

  CVector<1,N,T> row(uint8_t c) { return CVector<1,N,T>(c, *this);}

  Vector<N,T> getrow(int8_t c) { Vector<N,T> x; if (c>=N) return x; for (uint8_t i=0, j=c*N; i<N; i++, j++) x.m[i] = this->m[j]; return x;}

  void swaprow(uint8_t r1, uint8_t r2) {T v[N]; memcpy(v,this->m+r1*N,N*sizeof(T)); memcpy(this->m+r1*N,this->m+r2*N,N*sizeof(T)); memcpy(this->m+r2*N,v,N*sizeof(T));}

  //Diagonal K>0 right of the main diagonal & L<0 left
  template<int K=0> CVector<N+1,N-(K>=0?K:-K),T> diag() { return CVector<N+1,N-(K>=0?K:-K),T>(this->m[K>=0?K:-N*K]);}

  Vector<N,T> getdiag(int8_t c=0) { Vector<N,T> x; if (c>=N || c<=-N) return x; for (uint8_t i=0, j=(c>=0?c:-c*N); j<N*N; i++, j+=N+1) x.m[i] = this->m[j]; return x;}
  
  T det() { //Determinant using Barreiss algo
    uint8_t i, j, k;
    T piv = 1, dvd, mul, mat[N*N];
    if (this->isnan()) return fxm::info<T>.nan.val;
    memcpy(mat, this->m, N*N*sizeof(T));
    for ( i=0; i<N-1; i++ ) {
      dvd = piv;
      piv = mat[i*(N+1)];
      for ( j=i+1; j<N; j++ ) {
        mul = mat[j*N+i];
        for ( k=i+1; k<N; k++ )
          mat[j*N+k] = (piv*mat[j*N+k] - mul*mat[i*N+k]) / dvd;   
      }
    }  
    return mat[N*N-1]; 
  }

  T tridet() { //Determinant for tridiagonal matrix
    uint8_t i, j;
    T r, r1=1, r2;
    if ( !isTridiagonal() ) return this->nan;
    for (i=0; i<N*N; i+=N+1) {
      r = this->m[i]*r1;
      if (i) r-= this->m[i-1]*this->m[i-N]*r2;
      r2 = r1;
      r1 = r;
    }
    return r; 
  }

  Matrix& inverse() {
    int8_t i,j,k;
    T pivot, ratio;
    Matrix inv((T)1);
    for( i=0; i<N; i++) {
      k = i;
      pivot = fxabs(this->m[i*(N+1)]);
      for (j=i+1; j<N; j++) {
        ratio = fxabs(this->m[j*N+i]);  
        if ( ratio<=pivot ) continue;
        pivot = ratio;
        k = j;
      }
      if (pivot==0.0) {
        fxnan(inv.m[0]);
        break;
      }
      if (k != i) {
        for (j=i; j<N; j++) fxswap(this->m[i*N+j],this->m[k*N+j]);
        inv.swaprow(i,j);
      }
      for (k=0; k<N; k++) {
        if (k==i) continue;
        ratio = this->m[k*N+i]/pivot;
        for (j=i+1; j<N; j++) this->m[k*N+j] -= ratio*this->m[i*N+j];
        for (j=0;   j<N; j++) inv.m[k*N+j] -= ratio*inv.m[i*N+j];
      }
      for (j=i+1; j<N; j++) this->m[i*N+j] /= pivot;
      for (j=0; j<N; j++) inv.m[i*N+j] /= pivot;
    }
    memcpy(this->m, inv.m, N*N*sizeof(T));  
    return(*this);
  }

  Matrix inv() {
    Matrix mat(this->m, false);
    return mat.inverse();
  }

  Vector<N,T> solve(const Vector<N,T>& v) {
    bool z = v.iszero();
    uint8_t i,j,k;
    T pivot, ratio, mat[N*N];
    Vector<N,T> vec(v.m, false);
    if ( this->isnan() ) return Vector<N>();
    if ( isTridiagonal() ) return solveTridiag(v);
    memcpy(mat, this->m, N*N*sizeof(T));
    for( i=0; i<N; i++) {
      if (z && i == N-1 /*&& mfxequ(mat[N*N-1])*/ ) {
        vec[N-1] = (T)1.0; 
        k = 255; //eigenvector problem
      break;
      }
      k = i;
      pivot = fxabs(mat[i*N+i]);
      for (j=i+1; j<N; j++) {
        ratio = fxabs(mat[j*N+i]);  
        if ( ratio<=pivot ) continue;
        pivot = ratio;
        k = j;
      }
      if ( fxequ(pivot) ) {
        for (j=i; j<N; j++) if ( !fxequ(vec[j]) ) return Vector<N>();
        for (j=i; j<N; j++) mat[j*(N+1)] = 1;
        break; //multiple solutions found
      }   
      if (pivot==0.0) return Vector<N>();
      pivot = mat[k*N+i];
      if (k != i) {
        for (j=i; j<N; j++) fxswap(mat[i*N+j],mat[k*N+j]);
        if (!z) fxswap(vec.m[i],vec.m[k]);
      }
      for (k=i+1; k<N; k++) {
        ratio = mat[k*N+i]/pivot;
        for (j=i+1; j<N; j++) mat[k*N+j] -= ratio*mat[i*N+j];
        vec[k] -= ratio*vec[i];
      }
      for (j=i+1; j<N; j++) mat[i*N+j] /= pivot;
      if (!z) vec.m[i] /= pivot;
    }
    for( i=N-2; i<N; i--)
      for (j=i+1; j<N; j++)
        vec[i] -= mat[i*N+j]*vec[j];
    if (k==255) vec.normalize();     
    return vec;
  }

  //Solve mat*x = v when mat is tridiagonal
  Vector<N,T> solveTridiag(const Vector<N,T>& v) {
    uint8_t i;
    Vector<N,T> d, e;
    if ( !isTridiagonal() ) return Vector<N>();
    d = getdiag(1);
    e = v;
    d.m[0] /= this->m[0];
    for (i=1; i<N-1; i++) d[i] /= this->m[i*(N+1)] - this->m[i*(N+1)-1]*d.m[i-1];
    e.m[0] /= this->m[0];
    for (i=1; i<N; i++) e.m[i] = (e.m[i] - this->m[i*(N+1)-1]*e[i-1]) / (this->m[i*(N+1)] - this->m[i*(N+1)-1]*d.m[i-1]);
    if ( e == 0 ) {
      e.m[N-1] = 1.0;
      i = 254; //multiple solutions
    }
    for (i=N-2; i<N; i--) e.m[i] -= d.m[i]*e.m[i+1];
    if (i==254) e.normalize();
    return e;
  }

  // Gram Schmidt orthogonalization
  Matrix& orthogonalize() {
    uint8_t i, j;
    transpose();
    for (i=1; i<N; i++)
      for (j=0; j<i; j++)
        (*this)(i) -= (*this)(i).project( (*this)(j) );
    for (i=0; i<N; i++) (*this)(i).normalize();
    return transpose();
  }

  //Householder reflection mat Q = I - 2.0*( v.vT )/(vT.v)
  Matrix& householder(const Vector<N,T> vec) {
    uint8_t i, j;
    Vector<N,T> v = vec;
    v.normalize();
    *this = 1;
    *this -= (v ^ v) * (T)2.0;
    return *this;
  }

  //Do householder reflection at submatrix starting at (i,i) and return reflection matrix
  Matrix householder(uint8_t i) {
    uint8_t c, j, k;
    Matrix hh;
    Vector<N,T> v(0);
    T r;
    for (j=i; j<N; j++) v.m[j] = this->m[j*N+i];
    r = fxcopysign(fxnorm(v.m+i,N-i), v.m[i]);
    v.m[i] += r;
    hh.householder(v);
    this->m[i*(N+1)] = -r;
    for (c=i+1; c<N; c++) {
      v = col(c);
      this->m[c*N+i] = 0;
      for (j=i; j<N; j++)  this->m[j*N+c] = hh(j) * v;
    }
    return hh;
  }

  //QR decompose matrix and return Q using Householder reflections
  Matrix QRhouseholder() {
    Matrix q(1);
    for (uint8_t i=0; i<N-1; i++)
      q *= householder(i);
    return q;
  }

  //Performs Givens rotation and returns the matrix that zeroes (i,j) element (right lower / left upper)
  Matrix givens( uint8_t i, uint8_t j, bool right=true ) {
    uint8_t p, k;
    T c, s, r, a, b;
    Matrix rot(1);
    p = right ? i-1 : i+1;
    if (p>N) return;
    a = this->m[p*N+j];
    b = this->m[i*N+j];
    if ( b == 0 ) {
      c = fxsign(a);
      s = 0;
      r = fxabs(a);
    } else if ( a == 0 ) {
      c = 0;
      s = fxsign(b);
      r = fxabs(b);
    } else if ( fxabs(a) > fxabs(b) ) {
      s = b/a;
      r = fxcopysign( fxsqrt(1 + s*s), a );
      c = 1.0/r;
      s *= c;
      r *= a;
    } else {
      c = a/b;
      r = fxcopysign( fxsqrt(1 + s*s), b );
      s = 1.0/r;
      c *= s;
      r *= b;
    }
    rot.m[p*(N+1)] = rot.m[i*(N+1)] = c;
    rot.m[p*N+i] = s;
    rot.m[i*N+p] = -s;
    //assuming elements to the left (or right) are zeroed out
    for (k = (right ? j+1 : 0); k < (right ? N : j); k++) { 
      a = c*this->m[p*N+k]+s*this->m[i*N+k];  
      b = c*this->m[i*N+k]-s*this->m[p*N+k];
      this->m[p*N+k] = a;  
      this->m[i*N+k] = b;
    }    
    this->m[p*N+j] = r;
    this->m[i*N+j] = 0.0;
    return rot; 
  }

  //QR decompose matrix and return Q using Givens rotations
  Matrix QRgivens() {
    uint8_t i, j;
    Matrix q(1);
    for (j=0; j<N-1; j++)
      for (i=N-1; i>j; i--)
        q *= givens(i, j).transpose();
    return q;
  }

  //Transform to Hessenberg form using Givens rorations
  Matrix& hessenberg() {
    uint8_t i, j;
    Matrix q(1);
    for (j=0; j<N-2; j++)
      for (i=N-1; i>j+1; i--)
        givens(i, j);
    return *this;    
  }

  Matrix& tridiag() {
  //Adopted from the tred functions of the Eispack library and the Handbook of Automatic Computation by J.H.Wilkinson and C. Reincsh
  //Reduce a symmetric matrix to tridiagonal form using Householder reflections
  uint8_t i, j, k;  
  T s, norm, p, r, sdiag[N];
  for (i=N-1; i>0; i--) {
    k = i - 1;
    if ( !k ) {
      sdiag[1] = this->m[N];
      continue;
    }
    s = (*this)(i).norm1();  
    if ( fxequ(s) ) {
     //If s is too small the transformation is skipped for orthogonality to be guaranteed
     //Be sure to set desired tolerance using the fxapprox() function the source recommends
     //setting it to the relative machine precision in Algol which as far as I understand is
     //for floating numbers is 2^(-mantissa digits) for Arduino which only supports IEEE754 
     //single precision (1.1921e-07), for fixed point numbers it depends on the application
     //but something like 2^(2-integer digits) that won't overflow on inversion sounds good 
      sdiag[i] = this->m[N*i+k];
      continue;
    }
    norm=0.0;
    for (j=0; j<i; j++) {
      this->m[N*i+j] /= s;
      norm += fxsq( this->m[N*i+j] );
    }
    p = this->m[N*i+k];
    r = fxcopysign(fxsqrt(norm), p);
    sdiag[i] = s*r;
    norm -= p*r;
    this->m[N*i+k] = p - r;
    p=0.0;
    for ( k=0; k<i; k++ ) {
      this->m[N*k+i] = this->m[N*i+k] / norm;
      r = 0.0;
      for ( j=0; j<=k; j++ ) r += this->m[N*k+j]*this->m[N*i+j];
      for ( j=k+1; j<i; j++) r += this->m[N*j+k]*this->m[N*i+j];
      sdiag[k] = r/norm;
      p += sdiag[k] * this->m[N*i+k];
    }
    s = (0.5*p) / norm;
    for ( k=0; k<i; k++ ) {
      p = this->m[N*i+k];
      r = sdiag[k] -= s*p;
      for (j=0; j<=k; j++) this->m[N*k+j] -= p*sdiag[j]+r*this->m[N*i+j];
    }
  }
  for ( k=N+1, j=1; k<N*N; j++, k+=N+1) {
    this->m[k-1] = this->m[k-N] = sdiag[j];
    if (j<2) continue;
    for (i=0; i<j-2; i++) this->m[i*N+j] = this->m[j*N+i] = 0;
  }  
  return *this;
}

//Get eigenvalues of symmetric matrix
  Vector<N,T> eigenvals() {
  //Adopted from the tql functions of the Eispack library and the Handbook of Automatic Computation by J.H.Wilkinson and C. Reincsh
  //Calculation of eigenvalues by QL transformation os symmetric matrix
    uint8_t i, j, k, n;
    T d, r, s, c, h;
    Matrix<N> mat = *this;
    Vector<N> lambda, sdiag;
    if ( isTridiagonal() ) {
      if ( !isSymmetric() )
        for (i=1; i<N; i++) //If non symmetric tridiagonal matrix make it one
          mat.m[i*(N+1)-1] = mat.m[i*(N+1)-N] = fxcopysign( fxsqrt( mat.m[i*(N+1)-1]*mat.m[i*(N+1)-N] ), mat.m[i*(N+1)-N] );
    } else if ( isSymmetric() )
      mat.tridiag();
    else
      return lambda;
    lambda = mat.getdiag();
    sdiag = mat.getdiag(1);
    for ( i=0; i<N; i++ ) {
      n=0;
      do {
        for (j=i; j<N-1; j++) if ( fxequ(sdiag.m[j]/lambda.m[j] ) ) break;
        if (j == i) continue;
        if ( n++>30 ) return Vector<N>();
        d = lambda[i+1] - lambda[i];
        d *= 0.5 / sdiag.m[i];
        r = fxhypot(d);
        d= lambda[j] - lambda[i] + sdiag.m[i]/( d + fxcopysign(r, d) );
        c = h = 1.0;
        s = 0.0;
        k = j-1;
        do {
          mat.m[0] = c*sdiag.m[k];
          mat.m[1] = h*sdiag.m[k];
          sdiag.m[k+1] = r = fxhypot(mat.m[0],d);
          if (r == 0.0) {
            lambda[k+1] -= s;
            sdiag.m[j] = 0.0;
            break;
          }
          c = mat.m[0]/r;
          h = d/r;
          d = lambda[k+1] - s;
          r= (lambda[k]-d)*c + 2.0*h*mat.m[1];
          s= c*r;
          lambda[k+1] = d + s;
          d = h*r - mat.m[1];
        } while (k-- != i);
        if (r == 0.0 && k) continue;
        lambda[i] -= s;
        sdiag.m[i] = d;
        sdiag.m[j] = 0.0;
      } while (j != i);
    }
    do {
      j = 0;
      for (i=1; i<N; i++)
        if ( fxabs(lambda[i]) > fxabs(lambda[i-1]) ) {
          fxswap(lambda[i], lambda[i-1]);
          j = 1; 
        }      
    } while (j);
  return lambda;
}

//Get eigenvector of symmetric matrix from eigenvalue
  Vector<N,T> eigenvec( T lambda ) {
    uint8_t i;
    Vector<N> vec(0);
    Matrix<N> mat = *this;
    mat -= Matrix<N>(lambda);
    if ( isTridiagonal() ) {
      vec = getdiag(1);
      vec.m[0] /= mat.m[0];
      for (i=1; i<N-1; i++) vec.m[i] /= mat.m[i*(N+1)] - mat.m[i*(N+1)-1]*vec.m[i-1];    
      vec.m[N-1] = 1.0;
      for (i=N-2; i<N; i--) vec.m[i] = -vec.m[i]*vec.m[i+1];
      return vec.normalize();
    }
    vec = mat.solve(vec);
    return vec.normalize();
  }

  //Get eigenvalues and eigenvectors(as columns of the matrix) of symmetric matrix
  Vector<N,T> eigenvecs() {
    uint8_t i;
    Matrix<N> mat = *this;
    Vector<N> lambda = eigenvals();
    if ( lambda.isnan() )
      fxnan(this->m[0]);
    else for ( i=0; i<N; i++)
      (*this)(i) = mat.eigenvec( lambda.m[i] );
    return lambda;  
  }
  
  Matrix& operator = (T x) { memset(this->m,0,N*N*sizeof(T)); for (uint8_t i=0; i<N*N; i+=N+1) this->m[i] = x; return *this; }

  Matrix& operator = (const T x[]) { Vector<N*N,T>::operator = (x); return *this; }

  Matrix& operator = (Vector<N*N,T> x) { memcpy(this->m,x.m,N*N*sizeof(T)); return *this; }

  Matrix operator!() const { return Matrix(this->m, false).transpose();}

  CVector<1,N,T> operator[](uint8_t r) {if (r<N) return CVector<1,N,T>(this->m[r*N]); fxnan(this->m[0]); return CVector<1,N,T>(this->m[0]);}

  CVector<N,N,T> operator()(uint8_t c) {if (c<N) return CVector<N,N,T>(this->m[c]); fxnan(this->m[0]); return CVector<N,N,T>(this->m[0]);}

  T& operator()(uint8_t r, uint8_t c) {if (r<N && c<N) return this->m[r*N+c]; fxnan(this->m[0]); return this->m[0];}

  Matrix operator*(const Matrix<N,T> &x) {
    uint8_t i, j, k;
    T v;
    Matrix<N,T> s;
    if ( this->isnan() || x.isnan())
      fxnan(s.m[0]);
    else  for( i=0; i<N; i++ )
      for( j=0; j<N; j++ ) {
        v = 0;     
        for( k=0; k<N; k++ ) v += this->m[i*N+k]*fxconj(x.m[k*N+j]);
        s.m[i*N+j] = v;
      }  
    return s;
  }

  Matrix operator*(const T x[N*N]) {
    uint8_t i, j, k;
    T v;
    Matrix<N,T> s;
    if ( this->isnan() )
      fxnan(s.m[0]);
    else  for( i=0; i<N; i++ )
      for( j=0; j<N; j++ ) {
        v = 0;
        for( k=0; k<N; k++ )
          v += this->m[i*N+k]*fxconj(fxpgmread<T>(x+k*N+j));
        s.m[i*N+j] = v;
      }  
    return s;
  }

  template <class C> Matrix<N,C> operator*(const C& x) const { Matrix<N,C> r; if ( !this->isnan() && !fxisnan(x) ) for (uint8_t i=0; i<N*N; i++) r[i] = this->m[i] * x ; return r;}

  Vector<N,T> operator*(const Vector<N,T> &x) {
    uint8_t i, j;
    T v;
    Vector<N,T> s;
    if ( this->isnan() || x.isnan())
      fxnan(s.m[0]);
    else  for( i=0; i<N; i++ ) {
      v = 0;     
      for( j=0; j<N; j++ ) v += this->m[i*N+j]*fxconj(x.m[j]);
      s.m[i] = v;
    }  
    return s;
  }

  template <int M> inline Vector<N,T> operator*(const CVector<M,N,T> &x) {return (*this) * Vector<N,T>(x);}

  template <int M> GMatrix<N,M,T> operator*(const GMatrix<N,M,T> &x) { GMatrix<N,M,T> r; for (uint8_t i=0; i<M; i++) r[i] = (*this) * x[i];}

  Matrix operator*=(const Matrix<N,T> &x) { return *this = *this * x; }

  Matrix operator*=(const T x[N*N]) { return *this = *this * x; }

  Matrix operator / (const Matrix<N,T> &x) {
    return *this * x.inv();
  }

  Matrix operator / (const T x[N*N]) {
    Matrix y(x,fxm::readpgm);
    return *this * y.inv();
  }

  Matrix operator /=(const Matrix<N,T> &x) { return *this = *this / x; }

  Matrix operator /=(const T x[N*N]) { return *this = *this / x; }

//protected:
  Matrix<N,T>& balance() {
  //Adopted from the balanc function of the Eispack library and the Handbook of Automatic Computation by J.H.Wilkinson and C. Reincsh
  //Reduce the norm of matrix by exact diagonal similarity transdormations
    uint8_t i;
    bool conv=true;
    T c, f, g, r, s;

    while ( conv ) {
      conv = false;
      for ( i=0; i<N; i++) {
        f = fxabs(this->m[i*(N+1)]);
        c = (*this)(i).norm1() - f;
        r = (*this)[i].norm1() - f;
        if (c != 0.0 && r != 0.0) {
          g = r / 2;
          f = (T)1.0;
          s = c + r;
          while ( c < g ) {
            f *= 2;
            c *= 4;
          }
          g = r * (T)2.0;
          while ( c > g ) {
            f /= 2;
            c /= 4;
          }
          if ( (c+r)/f < (T)0.95*s) {
            conv = true;
            g = (T)1.0 / f;
            (*this)[i] *= g;
            (*this)(i) *= f;
          }
        }
      }
    }
    return *this;
  }
  
};


template <int M, int N=FXM_DIMENSION, class T=TFixed>
class GMatrix: public Vector<M*N,T> {
  friend class Vector<N,T>;
  friend class CVector<M,N,T>;
  friend class CVector<1,N,T>;
  friend class GMatrix<N,M,T>;
public:
  GMatrix() : Vector<M*N,T>() {}
  GMatrix(T x): Vector<M*N,T>(x) {}
  GMatrix(T x, bool fill) : Vector<M*N,T>(x) { this->m[0] = x; if (fill) for (uint8_t i=1; i<M*N; i++) this->m[i] = x;}
  GMatrix(const T x[N*N], T a) : Vector<M*N,T>(x,a) {}
  GMatrix(const T x[N*N], bool progmem) : Vector<M*N,T>(x, progmem)  {}
  template <class C> GMatrix(C x, C y, ...)  : Vector<M*N,T>() { va_list vl;  this->m[0] = (T)x; this->m[1] = (T)y; va_start(vl,y); for (uint8_t i=2; i<M*N; i++) this->m[i] = (T)va_arg(vl,C); va_end(vl); }

  GMatrix& fill(T v=0) { for (uint8_t i=0; i<M*N; i++) this->m[i] = v; return *this;}

  GMatrix<N,M,T> transpose() {GMatrix<N,M,T> r; T v; uint8_t i, j; for (j=0; j<M; j++) for (i=0; i<N; i++) r.m[i*N+j] = fxconj(this->m[j*N+i]); return r;}

  GMatrix<M,N,T>& project(Vector<M,T> &x) { for (uint8_t i=0; i<N; i++) (*this)[i].project(x); return *this; }
  
  GMatrix<M,N,T>& normalize() { for (uint8_t i=0; i<N; i++) (*this)[i].normalize(); return *this; }

  GMatrix<M-1,N-1,T> sub(uint8_t r, uint8_t c) {
    uint8_t i, j;
    GMatrix<M-1,N-1,T> x;
    for (i=0, j=0; i<N; i++) {
      if ( i == r) continue;
      x(j++) = (*this)(i).sub(c);
    }
    return x;
  }
    
  virtual T norm1() { T r= (T)0.0; for (uint8_t i=0; i<N; i++) r=fxmax( col(i).norm1(), r ); return r; }

  T normi() {  T r= (T)0.0; for (uint8_t i=0; i<M; i++) r=fxmax( row(i).norm1(), r ); return r; }

  inline T normax() { return Vector<M*N,T>::norm1(); }

  CVector<M,N,T> col(uint8_t c) { return CVector<M,N,T>(*this, c); }
  
  Vector<M,T> getcol(int8_t c) { Vector<M,T> x; if (c>=M) return x; for (uint8_t i=0, j=c; i<M; i++, j+=N) x.m[i] = this->m[j]; return x;}

  void swapcol(uint8_t c1, uint8_t c2) {uint8_t i, j; T v; for (  i = c1, j = c2; i<M*N; i+=N, j+=N )  {v = this->m[i]; this->m[i] = this->m[j]; this->m[j] = v;}}

  CVector<1,N,T> row(uint8_t c) { return CVector<1,N,T>(c, *this);}

  Vector<N,T> getrow(int8_t c) { Vector<N,T> x; if (c>=M) return x; for (uint8_t i=0, j=c*N; i<N; i++, j++) x.m[i] = this->m[j]; return x;}

  void swaprow(uint8_t r1, uint8_t r2) {T v[N]; memcpy(v,this->m+r1*N,N*sizeof(T)); memcpy(this->m+r1*N,this->m+r2*N,N*sizeof(T)); memcpy(this->m+r2*N,v,N*sizeof(T));}

  GMatrix<M,N,T>& operator = (const T x[N*N]) { for (uint8_t i=0; i<M*N; i++) this->m[i] = fxpgmread<T>(x+i); return *this; }

  GMatrix<N,M,T> operator!() const { return transpose();}

  CVector<1,N,T> operator[](uint8_t r) {if (r<M) return CVector<1,N,T>(this->m[r*N]); fxnan(this->m[0]); return CVector<1,N,T>(this->m[0]);}

  CVector<M,N,T> operator()(uint8_t c) {if (c<N) return CVector<M,N,T>(this->m[c]); fxnan(this->m[0]); return CVector<N,N,T>(this->m[0]);}

  T& operator()(uint8_t r, uint8_t c) {if (r<M && c<N) return this->m[r*N+c]; fxnan(this->m[0]); return this->m[0];}

  Matrix<M,T> operator*(const GMatrix<N,M,T> &x) { Matrix<M,T> s; for( uint8_t i=0; i<M; i++ ) s[i] = *this * x[i]; return s; }

  GMatrix<M,N,T> operator*(const Matrix<M,T> &x) { GMatrix<M,N,T> s; for( uint8_t i=0; i<M; i++ ) s[i] = *this * x[i]; return s; }

  template <int K> GMatrix<M,K,T> operator*(const GMatrix<N,K,T> &x) { GMatrix<M,K,T> s; for( uint8_t i=0; i<M; i++ ) s[i] = *this * x[i]; return s; }
};

template <int N, class T>
Stream& operator << (Stream& s, const Vector<N,T>& x) {
  union {
    const Vector<N,T>* p;
    Vector<N,T>* me;
  };
  uint8_t i;
  bool z=!fxWidth;
  T v;
  for (i=fxPos; i<fxIndent; i++) s.write(' ');
  fxPos=i;
  p = &x;
  if (z) fxWidth = 4;
  s.write('|');
  fxPos++;
  for (i=0; i<N; i++) s << (v = (*me)[i]);
  s.write('|');
  s.write('T');
  fxPos+=2;
  if (z) fxWidth = 0;
  return s;
}

template <int M, int N, class T>
Stream& operator << (Stream& s, const CVector<M,N,T>& x) {
  union {
    const CVector<M,N,T>* p;
    CVector<M,N,T>* me;
  };
  uint8_t i;
  bool z=!fxWidth;
  T v;
  for (i=fxPos; i<fxIndent; i++) s.write(' ');
  fxPos=i;
  p = &x;
  if (z) fxWidth = 4;
  s.write('|');
  fxPos++;  
  for (i=0; i<N; i++) s << (v = (*me)[i]);
  s.write('|');
  fxPos++;
  if (M!=1)   fxPos+=s.write('T');
  if (z) fxWidth = 0;
  return s;
}

template <int N, class T>
Stream& operator << (Stream& s, const Matrix<N,T>& x) {
  union {
    const Matrix<N,T>* p;
    Matrix<N,T>* me;
  };
  uint8_t i, j;
  bool z=!fxWidth;
  T v;
  for (i=fxPos; i<fxIndent; i++) s.write(' ');
  fxPos=i;
  p = &x;
  if (z) fxWidth = 6;
  for (i =0; i<N; i++) {
    if (i) {
      s.write('\n');
      if (fxIndent) for (j=0; j<fxIndent; j++) s.write(' ');
      fxPos = fxIndent;
    }
    s.write('|');
    fxPos++;
    for (j =0; j<N; j++) s << (v=(*me)(i,j));
    s.write('|');
    fxPos++;
  }
  return s;   
}

template <int M, int N, class T>
Stream& operator << (Stream& s, const GMatrix<M,N,T>& x) {
  union {
    const GMatrix<M,N,T>* p;
    GMatrix<M,N,T>* me;
  };
  uint8_t i, j;
  bool z=!fxWidth;
  T v;
  for (i=fxPos; i<fxIndent; i++) s.write(' ');
  fxPos=i;
  p = &x;
  if (z) fxWidth = 6;
  for (i =0; i<M; i++) {
    if (i) {
      s.write('\n');
      if (fxIndent) for (j=0; j<fxIndent; j++) s.write(' ');
      fxPos = fxIndent;
    }
    s.write('|');
    fxPos++;
    for (j =0; j<N; j++) s << (v=(*me)(i,j));
    s.write('|');
    fxPos++;
  }
  return s;   
}

template <int M, int N, class T=TFixed> constexpr size_t fxsize(const CVector<M,N,T>& x) { return N;}

template <int N=FXM_DIMENSION, class T=TFixed> constexpr size_t fxsize(const Vector<N,T>& x) { return N;}

template <int N=FXM_DIMENSION, class T=TFixed> inline Vector<N,T> fxnan() {return Vector<N,T>();}

template <int N=FXM_DIMENSION, class T=TFixed> inline Vector<N,T>& fxnan( Vector<N,T>& x ) {x[0] = fxm::info<T>.nan.val; return x;}

template <int M, int N, class T=TFixed> inline CVector<M,N,T>& fxnan( CVector<M,N,T>& x ) {x[0] = fxm::info<T>.nan.val; return x;}

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxisnan(const Vector<N,T>& x) { return x.isnan(); }

template <int M, int N, class T=TFixed> inline bool fxisnan(const CVector<M,N,T>& x) { return x.isnan(); }

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxisinf(const Vector<N,T>& x) { return x.isnan(); }

template <int M, int N, class T=TFixed>inline bool fxisinf(const CVector<M,N,T>& x) { return x.isnan(); }

template <int N=FXM_DIMENSION, class T=TFixed> inline T fxabs(const Vector<N,T>& x) { return x.norm(); }

template <int M, int N, class T=TFixed> inline T fxabs(const CVector<M,N,T>& x) { return x.norm(); }

template <int M, int N, class T=TFixed>
  bool fxequ(const CVector<M,N,T> &x) { 
    if ( x.isnan() ) return false;
    for (uint8_t i=0; i<N; i++) if ( !fxequ(x[i]) ) return false;
    return true;
  }

template <int N=FXM_DIMENSION, class T=TFixed>  inline bool fxequ(const Vector<N,T> &x) {  return fxequ(CVector<1,N,T>(x.m)); }

template <int M, int K, int N, class T=TFixed> inline bool fxequ(const CVector<M,N,T> &x, const CVector<K,N,T> &y) {  return fxequ(x-y); }

template <int M, int N, class T=TFixed>  inline bool fxequ(const Vector<N,T> &x, const CVector<M,N,T> &y) {  return fxequ(x-y); }

template <int M, int N, class T=TFixed>  inline bool fxequ(const CVector<M,N,T> &x, const Vector<N,T> &y) {  return fxequ(x-y); }

template <int M, int N, class T=TFixed> 
  Vector<N,T> fxsign(const CVector<M,N,T> &x) {
    if (x.isnan()) return Vector<N,T>();
    if (fxequ(x)) return Vector<N,T>((T)1.0);
    return x / x.norm();
  }

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxsign(const Vector<N,T> &x) {  return fxsign(CVector<1,N,T>(x.m)); }

template <int M, int K, int N, class T=TFixed> 
  Vector<N,T> fxcopysign(const CVector<M,N,T>& x, const CVector<K,N,T>& y) {
    if (x.isnan() || y.isnan()) return Vector<N,T>();
    return x.norm() * fxsign(y);
  }

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxcopysign(const Vector<N,T> &x, const Vector<N,T> &y) {  return fxcopysign(CVector<1,N,T>(x.m), CVector<1,N,T>(y.m)); }

template <int M, int N, class T=TFixed> inline bool fxcopysign(const Vector<N,T> &x, const CVector<M,N,T> &y) {  return fxcopysign(CVector<1,N,T>(x.m), y); }

template <int M, int N, class T=TFixed> inline bool fxcopysign(const CVector<M,N,T> &x, const Vector<N,T> &y) {  return fxcopysign(x, CVector<1,N,T>(y.m)); }

template <int M, int K, int N, class T=TFixed> 
Vector<N,T> fxmax(const CVector<M,N,T>& x, const CVector<K,N,T>& y) {
  if ( x.isnan() ) return x;
  if ( y.isnan() ) return y;
  return x.norm()< y.norm() ? y : x;
}

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxmax(const Vector<N,T> &x, const Vector<N,T> &y) {  return fxmax(CVector<1,N,T>(x.m), CVector<1,N,T>(y.m)); }

template <int M, int N, class T=TFixed>  inline bool fxmax(const Vector<N,T> &x, const CVector<M,N,T> &y) {  return fxmax(CVector<1,N,T>(x.m), y); }

template <int M, int N, class T=TFixed>  inline bool fxmax(const CVector<M,N,T> &x, const Vector<N,T> &y) {  return fxmax(x, CVector<1,N,T>(y.m)); }

template <int M, int N, class T=TFixed>
T fxmax(T x, const CVector<M,N,T>& y) {
  if ( y.isnan() || fxisnan(x) ) return fxm::info<T>.nan.val;
  T ay = y.norm();
  return x < ay ? ay : x;
  
}

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxmax(T x, const Vector<N,T> &y) {  return fxmax(x, CVector<1,N,T>(y.m)); }

template <int M, int N, class T=TFixed>  inline T fxmax(const CVector<M,N,T>& x, T y) { return fxmax(y, x); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline T fxmax(const Vector<N,T>& x, T y) { return fxmax(y, CVector<1,N,T>(x.m)); }

template <int M, int K, int N, class T=TFixed> 
Vector<N,T> fxmin(const CVector<M,N,T>& x, const CVector<K,N,T>& y) {
  if ( x.isnan() ) return x;
  if ( y.isnan() ) return y;
  return x.norm() > y.norm() ? y : x;
}

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxmin(const Vector<N,T> &x, const Vector<N,T> &y) {  return fxmin(CVector<1,N,T>(x.m), CVector<1,N,T>(y.m)); }

template <int M, int N, class T=TFixed> inline bool fxmin(const Vector<N,T> &x, const CVector<M,N,T> &y) {  return fxmin(CVector<1,N,T>(x.m), y); }

template <int M, int N, class T=TFixed>inline bool fxmin(const CVector<M,N,T> &x, const Vector<N,T> &y) {  return fxmin(x, CVector<1,N,T>(y.m)); }

template <int M, int N, class T=TFixed>
T fxmin(T x, const CVector<M,N,T>& y) {
  if ( y.isnan() || fxisnan(x) ) return fxm::info<T>.nan.val;
  T ay = y.norm();
  return x > ay ? ay : x;
  
}

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxmin(T x, const Vector<N,T> &y) {  return fxmin(x, CVector<1,N,T>(y.m)); }

template <int M, int N, class T=TFixed>  inline T fxmin(const CVector<M,N,T>& x, T y) { return fxmin(y, x); }

template <int M, int N, class T=TFixed>  inline T fxmin(const Vector<N,T>& x, T y) { return fxmin(y, CVector<M,N,T>(x.m)); }

template <int M, int N, class T=TFixed>
Vector<N,T> fxconj(const CVector<M,N,T>& x) {
  Vector<N,T> r=x;
  if ( x.isnan()) return r;
  for (uint8_t i=0; i<N; i++) r[i] = fxconj(r[i]);
  return r;  
}

template <int N=FXM_DIMENSION, class T=TFixed>  inline T fxconj(const Vector<N,T>& x) { return fxconj(CVector<1,N,T>(x.m)); }

template <int M, int N, class T=TFixed> 
CVector<M,N,T>& fxrandom(CVector<M,N,T>& x, T nrm = (T)1 ) { x = nrm * x.rand(nrm).normalize(); return x; }

template <int N=FXM_DIMENSION, class T=TFixed> inline Vector<N,T>& fxrandom(Vector<N,T>& x, T nrm = (T)1 ) { x = nrm * x.rand(nrm).normalize(); return x; }


namespace fxm { 

  constexpr uint8_t isqrt(uint8_t x) {
    return x == 1 ? 1 : x == 4 ? 2 : x == 9 ? 3 : x == 16 ? 4 : x == 25 ? 5 : x == 36 ? 6 : x == 49 ? 7 : x == 64 ? 8 : x == 81 ? 9 : x == 100 ? 10 : x == 121 ? 11 : x == 144 ? 12 : x == 169 ? 13 : x == 196 ? 14 : x == 225 ? 15 : 0;
  }

template <int N=FXM_DIMENSION, class T=TFixed> 
  Matrix<N,T> func0(const Matrix<N,T>& x, T (*f)(T)  ) {
    Matrix<N,T> r = x;
    if ( r.isnan() ) return r;
    Vector<N,T> lambda=r.eigenvecs();
    if ( lambda.isnan() ) return r;
    for (int8_t i=0; i< N; i++) lambda[i] = f(lambda[i]);
    if ( lambda.isnan() ){
      fxnan(r(0,0));
      return r;
    }
    return r * Matrix<N,T>(lambda,true) * !r;
  }

template <int N=FXM_DIMENSION, class T=TFixed> 
  Matrix<N,T> func1(const Matrix<N,T>& x, T y, T (*f)(T, T)  ) {
    Matrix<N,T> r = x;
    if ( r.isnan() ) return r;
    Vector<N,T> lambda=r.eigenvecs();
    if ( lambda.isnan() ) return r;
    for (int8_t i=0; i< N; i++) lambda[i] = f(lambda[i], y);
    if ( lambda.isnan() ){
      fxnan(r(0,0));
      return r;
    }
    return r * Matrix<N,T>(lambda,true) * !r;
  }
}


template <int N=FXM_DIMENSION, class T=TFixed> constexpr size_t fxsize(Matrix<N,T> x) { return N*N;}

template <int M, int N, class T=TFixed> constexpr size_t fxsize(const GMatrix<M,N,T>& x) { return M*N;}

template <int N=FXM_DIMENSION, class T=TFixed> inline Matrix<N,T> fxnan() {return Matrix<N,T>();}

template <int M, int N, class T=TFixed> inline GMatrix<M,N,T> fxnan() {return GMatrix<M,N,T>();}

template <int N=FXM_DIMENSION, class T=TFixed> inline Matrix<N,T>& fxnan( Matrix<N,T>& x ) {x[0] = fxm::info<T>.nan.val; return x;}

template <int M, int N, class T=TFixed> inline GMatrix<M,N,T>& fxnan( GMatrix<M,N,T>& x ) {x[0] = fxm::info<T>.nan.val; return x;}

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxisnan(const Matrix<N,T>& x) { return x.isnan(); }

template <int M, int N, class T=TFixed> inline bool fxisnan(const GMatrix<M,N,T>& x) { return x.isnan(); }

template <int N=FXM_DIMENSION, class T=TFixed> inline bool fxisinf(const Matrix<N,T>& x) { return x.isnan(); }

template <int M, int N, class T=TFixed> inline bool fxisinf(const GMatrix<M,N,T>& x) { return x.isnan(); }

template <int N=FXM_DIMENSION, class T=TFixed> inline Matrix<N,T> fxabs(const Matrix<N,T>& x) { Matrix<N,T> r=x; for(uint8_t i=0; i<N*N; i++) r[i] = fxabs(r[i]); return r;}

template <int N=FXM_DIMENSION, class T=TFixed> 
Matrix<N,T> fxmax(const Matrix<N,T>& x, const Matrix<N,T>& y) {
  if ( x.isnan() ) return x;
  if ( y.isnan() ) return y;
  return x.norm1()< y.norm1() ? y : x;
}


template <int N=FXM_DIMENSION, class T=TFixed> 
Matrix<N,T> fxmin(const Matrix<N,T>& x, const Matrix<N,T>& y) {
  if ( x.isnan() ) return x;
  if ( y.isnan() ) return y;
  return x.norm1() > y.norm1() ? y : x;
}


template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T>& fxrandom(Matrix<N,T>& x, T rng = (T)1 ) {return x.rand(rng);}

// Construct random matrix from random eigen values in x and random eigen vecs
template <int N=FXM_DIMENSION, class T=TFixed> 
Matrix<N,T> fxrandom(Matrix<N,T>& v, const Vector<N,T>& x ) {
  for (uint8_t i = 0; i < N; i++) v(i).rand(1.0).normalize(); //random eigen vectors
  return v *= Matrix<N,T>(x,true) * !v;
}

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxsq(const Matrix<N,T>& x) {return !x * x; }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxsqrt(const Matrix<N,T>& x) {return fxm::func0(x, fxsqrt<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxcbrt(const Matrix<N,T>& x) {return fxm::func0(x, fxcbrt<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxexp(const Matrix<N,T>& x) {return fxm::func0(x, fxexp<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxlog(const Matrix<N,T>& x) {return fxm::func0(x, fxlog<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxpow(const Matrix<N,T>& x, T y) {return fxm::func1(x, y, fxpow<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxsin(const Matrix<N,T>& x) {return fxm::func0(x, fxsin<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxcos(const Matrix<N,T>& x) {return fxm::func0(x, fxcos<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxtan(const Matrix<N,T>& x) {return fxm::func0(x, fxtan<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxsinh(const Matrix<N,T>& x) {return fxm::func0(x, fxsinh<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxcosh(const Matrix<N,T>& x) {return fxm::func0(x, fxcosh<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxtanh(const Matrix<N,T>& x) {return fxm::func0(x, fxtanh<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxasin(const Matrix<N,T>& x) {return fxm::func0(x, fxasin<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxacos(const Matrix<N,T>& x) {return fxm::func0(x, fxacos<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxatan(const Matrix<N,T>& x) {return fxm::func0(x, fxatan<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxasinh(const Matrix<N,T>& x) {return fxm::func0(x, fxasinh<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxacosh(const Matrix<N,T>& x) {return fxm::func0(x, fxacosh<T>); }

template <int N=FXM_DIMENSION, class T=TFixed>  inline Matrix<N,T> fxatanh(const Matrix<N,T>& x) {return fxm::func0(x, fxatanh<T>); }
#endif
 
