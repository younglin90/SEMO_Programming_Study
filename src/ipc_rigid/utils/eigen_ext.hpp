#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../../ipc/utils/eigen_ext.hpp"

namespace ipc {

template <typename T>
using VectorX = Eigen::VectorX<T>;
template <typename T>
using Vector2 = Eigen::Vector2<T>;
template <typename T>
using Vector3 = Eigen::Vector3<T>;
template <typename T>
using Matrix2 = Eigen::Matrix2<T>;
template <typename T>
using Matrix3 = Eigen::Matrix3<T>;
template <typename T>
using MatrixX = Eigen::MatrixX<T>;




typedef Eigen::DiagonalMatrix<double, 3> DiagonalMatrix3d;

template <typename T>
Eigen::SparseMatrix<T> SparseDiagonal(const Eigen::VectorX<T>& x);

template <typename T> inline Eigen::Matrix2<T> Hat(T x);
template <typename T> inline Eigen::Matrix3<T> Hat(Eigen::Vector3<T> x);
template <typename T> inline MatrixMax3<T> Hat(VectorMax3<T> x);

} // namespace ipc

#include "eigen_ext.tpp"
