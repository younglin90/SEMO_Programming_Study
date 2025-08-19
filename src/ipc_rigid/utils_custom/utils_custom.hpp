#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace ipc::rigid {

    template <
        typename TX,
        typename TY,
        typename DerivedR,
        typename DerivedC>
    inline void slice(
        const Eigen::SparseMatrix<TX>& X,
        const Eigen::DenseBase<DerivedR>& R,
        const Eigen::DenseBase<DerivedC>& C,
        Eigen::SparseMatrix<TY>& Y)
    {
        int xm = X.rows();
        int xn = X.cols();
        int ym = R.size();
        int yn = C.size();

        // special case when R or C is empty
        if (ym == 0 || yn == 0)
        {
            Y.resize(ym, yn);
            return;
        }

        assert(R.minCoeff() >= 0);
        assert(R.maxCoeff() < xm);
        assert(C.minCoeff() >= 0);
        assert(C.maxCoeff() < xn);

        // Build reindexing maps for columns and rows
        std::vector<std::vector<typename DerivedR::Scalar>> RI;
        RI.resize(xm);
        for (int i = 0; i < ym; i++)
        {
            RI[R(i)].push_back(i);
        }
        std::vector<std::vector<typename DerivedC::Scalar>> CI;
        CI.resize(xn);
        for (int i = 0; i < yn; i++)
        {
            CI[C(i)].push_back(i);
        }

        // Take a guess at the number of nonzeros (this assumes uniform distribution
        // not banded or heavily diagonal)
        std::vector<Eigen::Triplet<TY>> entries;
        entries.reserve((X.nonZeros() / (X.rows() * X.cols())) * (ym * yn));

        // Iterate over outside
        for (int k = 0; k < X.outerSize(); ++k)
        {
            // Iterate over inside
            for (typename Eigen::SparseMatrix<TX>::InnerIterator it(X, k); it; ++it)
            {
                for (auto rit = RI[it.row()].begin(); rit != RI[it.row()].end(); rit++)
                {
                    for (auto cit = CI[it.col()].begin(); cit != CI[it.col()].end(); cit++)
                    {
                        entries.emplace_back(*rit, *cit, it.value());
                    }
                }
            }
        }
        Y.resize(ym, yn);
        Y.setFromTriplets(entries.begin(), entries.end());
    }


    template <
        typename DerivedX,
        typename DerivedR,
        typename DerivedC,
        typename DerivedY>
    inline void slice(
        const Eigen::DenseBase<DerivedX>& X,
        const Eigen::DenseBase<DerivedR>& R,
        const Eigen::DenseBase<DerivedC>& C,
        Eigen::PlainObjectBase<DerivedY>& Y)
    {
#ifndef NDEBUG
        int xm = X.rows();
        int xn = X.cols();
#endif
        int ym = R.size();
        int yn = C.size();

        // special case when R or C is empty
        if (ym == 0 || yn == 0)
        {
            Y.resize(ym, yn);
            return;
        }

        assert(R.minCoeff() >= 0);
        assert(R.maxCoeff() < xm);
        assert(C.minCoeff() >= 0);
        assert(C.maxCoeff() < xn);

        // Resize output
        Y.resize(ym, yn);
        // loop over output rows, then columns
        for (int i = 0; i < ym; i++)
        {
            for (int j = 0; j < yn; j++)
            {
                Y(i, j) = X(R(i), C(j));
            }
        }
    }

    template <typename DerivedX, typename DerivedY, typename DerivedR>
    inline void slice(
        const Eigen::DenseBase<DerivedX>& X,
        const Eigen::DenseBase<DerivedR>& R,
        Eigen::PlainObjectBase<DerivedY>& Y)
    {
        // phony column indices
        Eigen::Matrix<typename DerivedR::Scalar, Eigen::Dynamic, 1> C;
        C.resize(1);
        C(0) = 0;
        return slice(X, R, C, Y);
    }

    template <typename DerivedX, typename DerivedR>
    inline DerivedX slice(
        const Eigen::DenseBase<DerivedX>& X,
        const Eigen::DenseBase<DerivedR>& R)
    {
        // This is not safe. See PlainMatrix
        DerivedX Y;
        slice(X, R, Y);
        return Y;
    }

    template <typename DerivedX, typename DerivedR>
    inline DerivedX slice(
        const Eigen::DenseBase<DerivedX>& X,
        const Eigen::DenseBase<DerivedR>& R,
        const int dim)
    {
        // This is not safe. See PlainMatrix
        DerivedX Y;
        slice(X, R, dim, Y);
        return Y;
    }

    template< class T >
    inline void slice(
        const std::vector<T>& unordered,
        std::vector<size_t> const& index_map,
        std::vector<T>& ordered)
    {
        // copy for the slice according to index_map, because unordered may also be
        // ordered
        std::vector<T> copy = unordered;
        ordered.resize(index_map.size());
        for (int i = 0; i < (int)index_map.size(); i++)
        {
            ordered[i] = copy[index_map[i]];
        }
    }



    template <typename T, typename DerivedR, typename DerivedC>
    inline void slice_into(
        const Eigen::SparseMatrix<T>& X,
        const Eigen::MatrixBase<DerivedR>& R,
        const Eigen::MatrixBase<DerivedC>& C,
        Eigen::SparseMatrix<T>& Y)
    {

        const int xm = X.rows();
        const int xn = X.cols();
        assert(R.size() == xm);
        assert(C.size() == xn);
        const int ym = Y.rows();
        const int yn = Y.cols();
        assert(R.minCoeff() >= 0);
        assert(R.maxCoeff() < ym);
        assert(C.minCoeff() >= 0);
        assert(C.maxCoeff() < yn);

        std::vector<bool> in_R(Y.rows());
        for (int r = 0; r < R.size(); r++) { in_R[R(r)] = true; }

        // Rebuild each column in C
        for (int c = 0; c < C.size(); c++)
        {
            int k = C(c);
            Eigen::SparseVector<T> Yk = Y.col(k);
            // implicit zeros
            for (typename Eigen::SparseMatrix<T>::InnerIterator yit(Y, k); yit; ++yit)
            {
                if (in_R[yit.row()]) { Yk.coeffRef(yit.row()) = 0; }
            }
            // explicit values
            for (typename Eigen::SparseMatrix<T>::InnerIterator xit(X, c); xit; ++xit)
            {
                // row in X
                int r = xit.row();
                // row in Y
                int s = R(r);
                Yk.coeffRef(s) = xit.value();
            }
            Y.col(k) = Yk;
        }

    }

    template <typename DerivedX, typename DerivedY, typename DerivedR, typename DerivedC>
    inline void slice_into(
        const Eigen::MatrixBase<DerivedX>& X,
        const Eigen::MatrixBase<DerivedR>& R,
        const Eigen::MatrixBase<DerivedC>& C,
        Eigen::PlainObjectBase<DerivedY>& Y)
    {

        int xm = X.rows();
        int xn = X.cols();
        assert(R.size() == xm);
        assert(C.size() == xn);
        const int ym = Y.rows();
        const int yn = Y.cols();
        assert(R.minCoeff() >= 0);
        assert(R.maxCoeff() < ym);
        assert(C.minCoeff() >= 0);
        assert(C.maxCoeff() < yn);

        // Build reindexing maps for columns and rows, -1 means not in map
        Eigen::Matrix<typename DerivedR::Scalar, Eigen::Dynamic, 1> RI;
        RI.resize(xm);
        for (int i = 0; i < xm; i++)
        {
            for (int j = 0; j < xn; j++)
            {
                Y(int(R(i)), int(C(j))) = X(i, j);
            }
        }
    }


    template <typename DerivedX, typename DerivedR, typename DerivedY>
    inline void slice_into(
        const Eigen::MatrixBase<DerivedX>& X,
        const Eigen::MatrixBase<DerivedR>& R,
        Eigen::PlainObjectBase<DerivedY>& Y)
    {
        // phony column indices
        Eigen::Matrix<typename DerivedR::Scalar, Eigen::Dynamic, 1> C;
        C.resize(1);
        C(0) = 0;
        return slice_into(X, R, C, Y);
    }






    template <typename DerivedV, typename DerivedF, typename DerivedL>
    inline void squared_edge_lengths(
        const Eigen::MatrixBase<DerivedV>& V,
        const Eigen::MatrixBase<DerivedF>& F,
        Eigen::PlainObjectBase<DerivedL>& L)
    {
        using namespace std;
        const int m = F.rows();
        switch (F.cols())
        {
        case 2:
        {
            L.resize(F.rows(), 1);
            for (int i = 0; i < F.rows(); i++)
            {
                L(i, 0) = (V.row(F(i, 1)) - V.row(F(i, 0))).squaredNorm();
            }
            break;
        }
        case 3:
        {
            L.resize(m, 3);
            // loop over faces
            for (int i = 0; i < m; ++i) {
                L(i, 0) = (V.row(F(i, 1)) - V.row(F(i, 2))).squaredNorm();
                L(i, 1) = (V.row(F(i, 2)) - V.row(F(i, 0))).squaredNorm();
                L(i, 2) = (V.row(F(i, 0)) - V.row(F(i, 1))).squaredNorm();
            }
            //parallel_for(
            //    m,
            //    [&V, &F, &L](const int i)
            //    {
            //        L(i, 0) = (V.row(F(i, 1)) - V.row(F(i, 2))).squaredNorm();
            //        L(i, 1) = (V.row(F(i, 2)) - V.row(F(i, 0))).squaredNorm();
            //        L(i, 2) = (V.row(F(i, 0)) - V.row(F(i, 1))).squaredNorm();
            //    },
            //    1000);
            break;
        }
        case 4:
        {
            L.resize(m, 6);
            // loop over faces
            for (int i = 0; i < m; ++i) {
                L(i, 0) = (V.row(F(i, 3)) - V.row(F(i, 0))).squaredNorm();
                L(i, 1) = (V.row(F(i, 3)) - V.row(F(i, 1))).squaredNorm();
                L(i, 2) = (V.row(F(i, 3)) - V.row(F(i, 2))).squaredNorm();
                L(i, 3) = (V.row(F(i, 1)) - V.row(F(i, 2))).squaredNorm();
                L(i, 4) = (V.row(F(i, 2)) - V.row(F(i, 0))).squaredNorm();
                L(i, 5) = (V.row(F(i, 0)) - V.row(F(i, 1))).squaredNorm();
            }
            //parallel_for(
            //    m,
            //    [&V, &F, &L](const int i)
            //    {
            //        L(i, 0) = (V.row(F(i, 3)) - V.row(F(i, 0))).squaredNorm();
            //        L(i, 1) = (V.row(F(i, 3)) - V.row(F(i, 1))).squaredNorm();
            //        L(i, 2) = (V.row(F(i, 3)) - V.row(F(i, 2))).squaredNorm();
            //        L(i, 3) = (V.row(F(i, 1)) - V.row(F(i, 2))).squaredNorm();
            //        L(i, 4) = (V.row(F(i, 2)) - V.row(F(i, 0))).squaredNorm();
            //        L(i, 5) = (V.row(F(i, 0)) - V.row(F(i, 1))).squaredNorm();
            //    },
            //    1000);
            break;
        }
        default:
        {
            assert(false && "Simplex size not supported");
        }
        }
    }


    template <typename DerivedV, typename DerivedF, typename DerivedL>
    inline void edge_lengths(
        const Eigen::MatrixBase<DerivedV>& V,
        const Eigen::MatrixBase<DerivedF>& F,
        Eigen::PlainObjectBase<DerivedL>& L)
    {
        squared_edge_lengths(V, F, L);
        L = L.array().sqrt().eval();
    }



    template <typename DerivedA, typename DerivedB>
    inline void repmat(
        const Eigen::MatrixBase<DerivedA>& A,
        const int r,
        const int c,
        Eigen::PlainObjectBase<DerivedB>& B)
    {
        assert(r > 0);
        assert(c > 0);
        // Make room for output
        B.resize(r * A.rows(), c * A.cols());

        // copy tiled blocks
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                B.block(i * A.rows(), j * A.cols(), A.rows(), A.cols()) = A;
            }
        }
    }

    template <typename T, int majorType>
    inline void repmat(
        const Eigen::SparseMatrix<T, majorType>& A,
        const int r,
        const int c,
        Eigen::SparseMatrix<T, majorType>& B)
    {
        assert(r > 0);
        assert(c > 0);
        B.resize(r * A.rows(), c * A.cols());
        std::vector<Eigen::Triplet<T>> b;
        b.reserve(r * c * A.nonZeros());

        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                // loop outer level
                for (int k = 0; k < A.outerSize(); ++k)
                {
                    // loop inner level
                    for (typename Eigen::SparseMatrix<T, majorType>::InnerIterator
                        it(A, k); it; ++it)
                    {
                        Eigen::Triplet<T> triplet(i * A.rows() + it.row(), j * A.cols()
                            + it.col(), it.value());
                        b.push_back(triplet);
                    }
                }
            }
        }
        B.setFromTriplets(b.begin(), b.end());
    }




    template <
        typename DerivedV,
        typename DerivedT,
        typename Derivedvol>
    inline void volume(
        const Eigen::MatrixBase<DerivedV>& V,
        const Eigen::MatrixBase<DerivedT>& T,
        Eigen::PlainObjectBase<Derivedvol>& vol)
    {
        using namespace Eigen;
        const int m = T.rows();
        vol.resize(m, 1);
        for (int t = 0; t < m; t++)
        {
            typedef Eigen::Matrix<typename DerivedV::Scalar, 1, 3> RowVector3S;
            const RowVector3S& a = V.row(T(t, 0));
            const RowVector3S& b = V.row(T(t, 1));
            const RowVector3S& c = V.row(T(t, 2));
            const RowVector3S& d = V.row(T(t, 3));
            vol(t) = -(a - d).dot((b - d).cross(c - d)) / 6.;
        }
    }



    template <
        class IndexVectorI,
        class IndexVectorJ,
        class ValueVector,
        typename T>
    inline void sparse(
        const IndexVectorI& I,
        const IndexVectorJ& J,
        const ValueVector& V,
        const size_t m,
        const size_t n,
        Eigen::SparseMatrix<T>& X)
    {
        using namespace std;
        using namespace Eigen;
        assert((int)I.maxCoeff() < (int)m);
        assert((int)I.minCoeff() >= 0);
        assert((int)J.maxCoeff() < (int)n);
        assert((int)J.minCoeff() >= 0);
        assert(I.size() == J.size());
        assert(J.size() == V.size());
        // Really we just need .size() to be the same, but this is safer
        assert(I.rows() == J.rows());
        assert(J.rows() == V.rows());
        assert(I.cols() == J.cols());
        assert(J.cols() == V.cols());
        vector<Triplet<T> > IJV;
        IJV.reserve(I.size());
        for (int x = 0; x < I.size(); x++)
        {
            IJV.push_back(Triplet<T >(I(x), J(x), V(x)));
        }
        X.resize(m, n);
        X.setFromTriplets(IJV.begin(), IJV.end());
    }







    template <typename DerivedX, typename DerivedY, typename DerivedIX>
    inline void sort2(
        const Eigen::DenseBase<DerivedX>& X,
        const int dim,
        const bool ascending,
        Eigen::PlainObjectBase<DerivedY>& Y,
        Eigen::PlainObjectBase<DerivedIX>& IX)
    {
        using namespace Eigen;
        using namespace std;
        typedef typename DerivedY::Scalar YScalar;
        Y = X.derived().template cast<YScalar>();


        // get number of columns (or rows)
        int num_outer = (dim == 1 ? X.cols() : X.rows());
        // get number of rows (or columns)
        int num_inner = (dim == 1 ? X.rows() : X.cols());
        assert(num_inner == 2); (void)num_inner;
        typedef typename DerivedIX::Scalar Index;
        IX.resizeLike(X);
        if (dim == 1)
        {
            IX.row(0).setConstant(0);// = DerivedIX::Zero(1,IX.cols());
            IX.row(1).setConstant(1);// = DerivedIX::Ones (1,IX.cols());
        }
        else
        {
            IX.col(0).setConstant(0);// = DerivedIX::Zero(IX.rows(),1);
            IX.col(1).setConstant(1);// = DerivedIX::Ones (IX.rows(),1);
        }
        // loop over columns (or rows)
        for (int i = 0; i < num_outer; i++)
        {
            YScalar& a = (dim == 1 ? Y(0, i) : Y(i, 0));
            YScalar& b = (dim == 1 ? Y(1, i) : Y(i, 1));
            Index& ai = (dim == 1 ? IX(0, i) : IX(i, 0));
            Index& bi = (dim == 1 ? IX(1, i) : IX(i, 1));
            if ((ascending && a > b) || (!ascending && a < b))
            {
                std::swap(a, b);
                std::swap(ai, bi);
            }
        }
    }

    template <typename DerivedX, typename DerivedY, typename DerivedIX>
    inline void sort3(
        const Eigen::DenseBase<DerivedX>& X,
        const int dim,
        const bool ascending,
        Eigen::PlainObjectBase<DerivedY>& Y,
        Eigen::PlainObjectBase<DerivedIX>& IX)
    {
        using namespace Eigen;
        using namespace std;
        typedef typename DerivedY::Scalar YScalar;
        Y = X.derived().template cast<YScalar>();
        Y.resizeLike(X);
        for (int j = 0; j < X.cols(); j++)for (int i = 0; i < X.rows(); i++)Y(i, j) = (YScalar)X(i, j);

        // get number of columns (or rows)
        int num_outer = (dim == 1 ? X.cols() : X.rows());
        // get number of rows (or columns)
        int num_inner = (dim == 1 ? X.rows() : X.cols());
        assert(num_inner == 3); (void)num_inner;
        typedef typename DerivedIX::Scalar Index;
        IX.resizeLike(X);
        if (dim == 1)
        {
            IX.row(0).setConstant(0);// = DerivedIX::Zero(1,IX.cols());
            IX.row(1).setConstant(1);// = DerivedIX::Ones (1,IX.cols());
            IX.row(2).setConstant(2);// = DerivedIX::Ones (1,IX.cols());
        }
        else
        {
            IX.col(0).setConstant(0);// = DerivedIX::Zero(IX.rows(),1);
            IX.col(1).setConstant(1);// = DerivedIX::Ones (IX.rows(),1);
            IX.col(2).setConstant(2);// = DerivedIX::Ones (IX.rows(),1);
        }


        const auto& inner = [&IX, &Y, &dim, &ascending](const Index& i)
            {
                YScalar& a = (dim == 1 ? Y.coeffRef(0, i) : Y.coeffRef(i, 0));
                YScalar& b = (dim == 1 ? Y.coeffRef(1, i) : Y.coeffRef(i, 1));
                YScalar& c = (dim == 1 ? Y.coeffRef(2, i) : Y.coeffRef(i, 2));
                Index& ai = (dim == 1 ? IX.coeffRef(0, i) : IX.coeffRef(i, 0));
                Index& bi = (dim == 1 ? IX.coeffRef(1, i) : IX.coeffRef(i, 1));
                Index& ci = (dim == 1 ? IX.coeffRef(2, i) : IX.coeffRef(i, 2));
                if (ascending)
                {
                    // 123 132 213 231 312 321
                    if (a > b)
                    {
                        std::swap(a, b);
                        std::swap(ai, bi);
                    }
                    // 123 132 123 231 132 231
                    if (b > c)
                    {
                        std::swap(b, c);
                        std::swap(bi, ci);
                        // 123 123 123 213 123 213
                        if (a > b)
                        {
                            std::swap(a, b);
                            std::swap(ai, bi);
                        }
                        // 123 123 123 123 123 123
                    }
                }
                else
                {
                    // 123 132 213 231 312 321
                    if (a < b)
                    {
                        std::swap(a, b);
                        std::swap(ai, bi);
                    }
                    // 213 312 213 321 312 321
                    if (b < c)
                    {
                        std::swap(b, c);
                        std::swap(bi, ci);
                        // 231 321 231 321 321 321
                        if (a < b)
                        {
                            std::swap(a, b);
                            std::swap(ai, bi);
                        }
                        // 321 321 321 321 321 321
                    }
                }
            };

        for (Index i = 0; i < num_outer; ++i) {
            inner(i);
        }

        //parallel_for(num_outer, inner, 16000);
    }



    /// Comparison struct used by sort
    /// http://bytes.com/topic/c/answers/132045-sort-get-index
    template<class T> struct IndexLessThan
    {
        IndexLessThan(const T arr) : arr(arr) {}
        bool operator()(const size_t a, const size_t b) const
        {
            return arr[a] < arr[b];
        }
        const T arr;
    };

    template <class T>
    inline void sort(
        const std::vector<T>& unsorted,
        const bool ascending,
        std::vector<T>& sorted,
        std::vector<size_t>& index_map)
    {
        // Original unsorted index map
        index_map.resize(unsorted.size());
        for (size_t i = 0; i < unsorted.size(); i++)
        {
            index_map[i] = i;
        }
        // Sort the index map, using unsorted for comparison
        std::sort(
            index_map.begin(),
            index_map.end(),
            IndexLessThan<const std::vector<T>& >(unsorted));

        // if not ascending then reverse
        if (!ascending)
        {
            std::reverse(index_map.begin(), index_map.end());
        }
        // make space for output without clobbering
        sorted.resize(unsorted.size());
        // reorder unsorted into sorted using index map
        slice(unsorted, index_map, sorted);
    }





    template <typename DerivedX, typename DerivedY, typename DerivedIX>
    inline void sort(
        const Eigen::DenseBase<DerivedX>& X,
        const int dim,
        const bool ascending,
        Eigen::PlainObjectBase<DerivedY>& Y,
        Eigen::PlainObjectBase<DerivedIX>& IX)
    {
        typedef typename DerivedX::Scalar Scalar;
        // get number of rows (or columns)
        int num_inner = (dim == 1 ? X.rows() : X.cols());
        // Special case for swapping
        switch (num_inner)
        {
        default:
            break;
        case 2:
            return sort2(X, dim, ascending, Y, IX);
        case 3:
            return sort3(X, dim, ascending, Y, IX);
        }
        using namespace Eigen;
        // get number of columns (or rows)
        int num_outer = (dim == 1 ? X.cols() : X.rows());
        // dim must be 2 or 1
        assert(dim == 1 || dim == 2);
        // Resize output
        Y.resizeLike(X);
        IX.resizeLike(X);
        // idea is to process each column (or row) as a std vector
        // loop over columns (or rows)
        for (int i = 0; i < num_outer; i++)
        {
            // Unsorted index map for this column (or row)
            std::vector<size_t> index_map(num_inner);
            std::vector<Scalar> data(num_inner);
            for (int j = 0; j < num_inner; j++)
            {
                if (dim == 1)
                {
                    data[j] = (Scalar)X(j, i);
                }
                else
                {
                    data[j] = (Scalar)X(i, j);
                }
            }
            // sort this column (or row)
            sort(data, ascending, data, index_map);
            // Copy into Y and IX
            for (int j = 0; j < num_inner; j++)
            {
                if (dim == 1)
                {
                    Y(j, i) = data[j];
                    IX(j, i) = index_map[j];
                }
                else
                {
                    Y(i, j) = data[j];
                    IX(i, j) = index_map[j];
                }
            }
        }
    }

    template <typename DerivedX, typename DerivedY>
    inline void sort(
        const Eigen::DenseBase<DerivedX>& X,
        const int dim,
        const bool ascending,
        Eigen::PlainObjectBase<DerivedY>& Y)
    {
        Eigen::Matrix< int, DerivedX::RowsAtCompileTime, DerivedX::ColsAtCompileTime > IX;
        return sort(X, dim, ascending, Y, IX);
    }






    template <typename Derivedl, typename DeriveddblA>
    inline void doublearea(
        const Eigen::MatrixBase<Derivedl>& ul,
        const typename Derivedl::Scalar nan_replacement,
        Eigen::PlainObjectBase<DeriveddblA>& dblA)
    {
        using namespace Eigen;
        using namespace std;
        typedef typename Derivedl::Index Index;
        // Only support triangles
        assert(ul.cols() == 3);
        // Number of triangles
        const Index m = ul.rows();
        Eigen::Matrix<typename Derivedl::Scalar, Eigen::Dynamic, 3> l;
        MatrixXi _;
        //
        // "Miscalculating Area and Angles of a Needle-like Triangle"
        // https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf
        sort(ul, 2, false, l, _);
        // semiperimeters
        //Matrix<typename Derivedl::Scalar,Dynamic,1> s = l.rowwise().sum()*0.5;
        //assert((Index)s.rows() == m);
        // resize output
        dblA.resize(l.rows(), 1);
        for (Index i = 0; i < m; ++i) {

            // Kahan's Heron's formula
            typedef typename Derivedl::Scalar Scalar;
            const Scalar arg =
                (l(i, 0) + (l(i, 1) + l(i, 2))) *
                (l(i, 2) - (l(i, 0) - l(i, 1))) *
                (l(i, 2) + (l(i, 0) - l(i, 1))) *
                (l(i, 0) + (l(i, 1) - l(i, 2)));
            dblA(i) = 2.0 * 0.25 * sqrt(arg);
            // Alec: If the input edge lengths were computed from floating point
            // vertex positions then there's no guarantee that they fulfill the
            // triangle inequality (in their floating point approximations). For
            // nearly degenerate triangles the round-off error during side-length
            // computation may be larger than (or rather smaller) than the height of
            // the triangle. In "Lecture Notes on Geometric Robustness" Shewchuck 09,
            // Section 3.1 http://www.cs.berkeley.edu/~jrs/meshpapers/robnotes.pdf,
            // he recommends computing the triangle areas for 2D and 3D using 2D
            // signed areas computed with determinants.
            assert(
                (nan_replacement == nan_replacement ||
                    (l(i, 2) - (l(i, 0) - l(i, 1))) >= 0)
                && "Side lengths do not obey the triangle inequality.");
            if (dblA(i) != dblA(i))
            {
                dblA(i) = nan_replacement;
            }
            assert(dblA(i) == dblA(i) && "DOUBLEAREA() PRODUCED NaN");

        }
        //parallel_for(
        //    m,
        //    [&l, &dblA, &nan_replacement](const int i)
        //    {
        //        // Kahan's Heron's formula
        //        typedef typename Derivedl::Scalar Scalar;
        //        const Scalar arg =
        //            (l(i, 0) + (l(i, 1) + l(i, 2))) *
        //            (l(i, 2) - (l(i, 0) - l(i, 1))) *
        //            (l(i, 2) + (l(i, 0) - l(i, 1))) *
        //            (l(i, 0) + (l(i, 1) - l(i, 2)));
        //        dblA(i) = 2.0 * 0.25 * sqrt(arg);
        //        // Alec: If the input edge lengths were computed from floating point
        //        // vertex positions then there's no guarantee that they fulfill the
        //        // triangle inequality (in their floating point approximations). For
        //        // nearly degenerate triangles the round-off error during side-length
        //        // computation may be larger than (or rather smaller) than the height of
        //        // the triangle. In "Lecture Notes on Geometric Robustness" Shewchuck 09,
        //        // Section 3.1 http://www.cs.berkeley.edu/~jrs/meshpapers/robnotes.pdf,
        //        // he recommends computing the triangle areas for 2D and 3D using 2D
        //        // signed areas computed with determinants.
        //        assert(
        //            (nan_replacement == nan_replacement ||
        //                (l(i, 2) - (l(i, 0) - l(i, 1))) >= 0)
        //            && "Side lengths do not obey the triangle inequality.");
        //        if (dblA(i) != dblA(i))
        //        {
        //            dblA(i) = nan_replacement;
        //        }
        //        assert(dblA(i) == dblA(i) && "DOUBLEAREA() PRODUCED NaN");
        //    },
        //    1000l);
    }





    /// Type of mass matrix
    enum MassMatrixType
    {
        /// Lumping area of each element to corner vertices in equal parts
        MASSMATRIX_TYPE_BARYCENTRIC = 0,
        /// Lumping area by Voronoi dual area (clamped to be positive according to
        /// Meyer et al. 2003)
        MASSMATRIX_TYPE_VORONOI = 1,
        /// Full (non-diagonal mass matrix) for piecewise linear functions
        MASSMATRIX_TYPE_FULL = 2,
        /// Use MASSMATRIX_TYPE_VORONOI for triangles and MASSMATRIX_TYPE_BARYCENTRIC
        /// for tetrahedra
        MASSMATRIX_TYPE_DEFAULT = 3,
        /// Total number of mass matrix types
        NUM_MASSMATRIX_TYPES = 4
    };




    template <typename Derivedl, typename DerivedF, typename Scalar>
    inline void massmatrix_intrinsic(
        const Eigen::MatrixBase<Derivedl>& l,
        const Eigen::MatrixBase<DerivedF>& F,
        const MassMatrixType type,
        Eigen::SparseMatrix<Scalar>& M)
    {
        const int n = F.maxCoeff() + 1;
        return massmatrix_intrinsic(l, F, type, n, M);
    }

    template <typename Derivedl, typename DerivedF, typename Scalar>
    inline void massmatrix_intrinsic(
        const Eigen::MatrixBase<Derivedl>& l,
        const Eigen::MatrixBase<DerivedF>& F,
        const MassMatrixType type,
        const int n,
        Eigen::SparseMatrix<Scalar>& M)
    {
        using namespace Eigen;
        using namespace std;
        MassMatrixType eff_type = type;
        const int m = F.rows();
        const int simplex_size = F.cols();
        // Use voronoi of for triangles by default, otherwise barycentric
        if (type == MASSMATRIX_TYPE_DEFAULT)
        {
            eff_type = (simplex_size == 3 ? MASSMATRIX_TYPE_VORONOI : MASSMATRIX_TYPE_BARYCENTRIC);
        }
        assert(F.cols() == 3 && "only triangles supported");
        Matrix<Scalar, Dynamic, 1> dblA;
        doublearea(l, 0., dblA);
        Matrix<typename DerivedF::Scalar, Dynamic, 1> MI;
        Matrix<typename DerivedF::Scalar, Dynamic, 1> MJ;
        Matrix<Scalar, Dynamic, 1> MV;

        switch (eff_type)
        {
        case MASSMATRIX_TYPE_BARYCENTRIC:
            // diagonal entries for each face corner
            MI.resize(m * 3, 1); MJ.resize(m * 3, 1); MV.resize(m * 3, 1);
            MI.block(0 * m, 0, m, 1) = F.col(0);
            MI.block(1 * m, 0, m, 1) = F.col(1);
            MI.block(2 * m, 0, m, 1) = F.col(2);
            MJ = MI;
            repmat(dblA, 3, 1, MV);
            MV.array() /= 6.0;
            break;
        case MASSMATRIX_TYPE_VORONOI:
        {
            // diagonal entries for each face corner
            // http://www.alecjacobson.com/weblog/?p=874
            MI.resize(m * 3, 1); MJ.resize(m * 3, 1); MV.resize(m * 3, 1);
            MI.block(0 * m, 0, m, 1) = F.col(0);
            MI.block(1 * m, 0, m, 1) = F.col(1);
            MI.block(2 * m, 0, m, 1) = F.col(2);
            MJ = MI;

            // Holy shit this needs to be cleaned up and optimized
            Matrix<Scalar, Dynamic, 3> cosines(m, 3);
            cosines.col(0) =
                (l.col(2).array().pow(2) + l.col(1).array().pow(2) - l.col(0).array().pow(2)) / (l.col(1).array() * l.col(2).array() * 2.0);
            cosines.col(1) =
                (l.col(0).array().pow(2) + l.col(2).array().pow(2) - l.col(1).array().pow(2)) / (l.col(2).array() * l.col(0).array() * 2.0);
            cosines.col(2) =
                (l.col(1).array().pow(2) + l.col(0).array().pow(2) - l.col(2).array().pow(2)) / (l.col(0).array() * l.col(1).array() * 2.0);
            Matrix<Scalar, Dynamic, 3> barycentric = cosines.array() * l.array();
            // Replace this: normalize_row_sums(barycentric,barycentric);
            barycentric = (barycentric.array().colwise() / barycentric.array().rowwise().sum()).eval();

            Matrix<Scalar, Dynamic, 3> partial = barycentric;
            partial.col(0).array() *= dblA.array() * 0.5;
            partial.col(1).array() *= dblA.array() * 0.5;
            partial.col(2).array() *= dblA.array() * 0.5;
            Matrix<Scalar, Dynamic, 3> quads(partial.rows(), partial.cols());
            quads.col(0) = (partial.col(1) + partial.col(2)) * 0.5;
            quads.col(1) = (partial.col(2) + partial.col(0)) * 0.5;
            quads.col(2) = (partial.col(0) + partial.col(1)) * 0.5;

            quads.col(0) = (cosines.col(0).array() < 0).select(0.25 * dblA, quads.col(0));
            quads.col(1) = (cosines.col(0).array() < 0).select(0.125 * dblA, quads.col(1));
            quads.col(2) = (cosines.col(0).array() < 0).select(0.125 * dblA, quads.col(2));

            quads.col(0) = (cosines.col(1).array() < 0).select(0.125 * dblA, quads.col(0));
            quads.col(1) = (cosines.col(1).array() < 0).select(0.25 * dblA, quads.col(1));
            quads.col(2) = (cosines.col(1).array() < 0).select(0.125 * dblA, quads.col(2));

            quads.col(0) = (cosines.col(2).array() < 0).select(0.125 * dblA, quads.col(0));
            quads.col(1) = (cosines.col(2).array() < 0).select(0.125 * dblA, quads.col(1));
            quads.col(2) = (cosines.col(2).array() < 0).select(0.25 * dblA, quads.col(2));

            MV.block(0 * m, 0, m, 1) = quads.col(0);
            MV.block(1 * m, 0, m, 1) = quads.col(1);
            MV.block(2 * m, 0, m, 1) = quads.col(2);

            break;
        }
        case MASSMATRIX_TYPE_FULL:
            MI.resize(m * 9, 1); MJ.resize(m * 9, 1); MV.resize(m * 9, 1);
            // indicies and values of the element mass matrix entries in the order
            // (0,1),(1,0),(1,2),(2,1),(2,0),(0,2),(0,0),(1,1),(2,2);
            MI << F.col(0), F.col(1), F.col(1), F.col(2), F.col(2), F.col(0), F.col(0), F.col(1), F.col(2);
            MJ << F.col(1), F.col(0), F.col(2), F.col(1), F.col(0), F.col(2), F.col(0), F.col(1), F.col(2);
            repmat(dblA, 9, 1, MV);
            MV.block(0 * m, 0, 6 * m, 1) /= 24.0;
            MV.block(6 * m, 0, 3 * m, 1) /= 12.0;
            break;
        default:
            assert(false && "Unknown Mass matrix eff_type");
        }
        sparse(MI, MJ, MV, n, n, M);
    }













    template <typename DerivedV, typename DerivedF, typename Scalar, int simplex_size = DerivedF::ColsAtCompileTime>
    struct MassMatrixHelper;

    // This would be easier with C++17 if constexpr
    // Specialization for triangles
    template <typename DerivedV, typename DerivedF, typename Scalar>
    struct MassMatrixHelper<DerivedV, DerivedF, Scalar, 3> {
        static void compute(
            const Eigen::MatrixBase<DerivedV>& V,
            const Eigen::MatrixBase<DerivedF>& F,
            const MassMatrixType type,
            Eigen::SparseMatrix<Scalar>& M)
        {
            MassMatrixType eff_type =
                type == MASSMATRIX_TYPE_DEFAULT ? MASSMATRIX_TYPE_VORONOI : type;
            Eigen::Matrix<Scalar, Eigen::Dynamic, 3> l;
            edge_lengths(V, F, l);
            return massmatrix_intrinsic(l, F, eff_type, M);
        }
    };






    // Specialization for tetrahedra
    template <typename DerivedV, typename DerivedF, typename Scalar>
    struct MassMatrixHelper<DerivedV, DerivedF, Scalar, 4> {
        static void compute(
            const Eigen::MatrixBase<DerivedV>& V,
            const Eigen::MatrixBase<DerivedF>& F,
            const MassMatrixType type,
            Eigen::SparseMatrix<Scalar>& M)
        {
            const int n = V.rows();
            const int m = F.rows();
            using Eigen::Matrix;
            using Eigen::Dynamic;
            MassMatrixType eff_type =
                type == MASSMATRIX_TYPE_DEFAULT ? MASSMATRIX_TYPE_BARYCENTRIC : type;
            Eigen::Matrix<Scalar, Dynamic, 1> vol;
            volume(V, F, vol);
            vol = vol.array().abs();
            Matrix<typename DerivedF::Scalar, Dynamic, 1> MI;
            Matrix<typename DerivedF::Scalar, Dynamic, 1> MJ;
            Matrix<Scalar, Dynamic, 1> MV;

            switch (eff_type)
            {
            case MASSMATRIX_TYPE_BARYCENTRIC:
                MI.resize(m * 4, 1); MJ.resize(m * 4, 1); MV.resize(m * 4, 1);
                MI.block(0 * m, 0, m, 1) = F.col(0);
                MI.block(1 * m, 0, m, 1) = F.col(1);
                MI.block(2 * m, 0, m, 1) = F.col(2);
                MI.block(3 * m, 0, m, 1) = F.col(3);
                MJ = MI;
                repmat(vol, 4, 1, MV);
                assert(MV.rows() == m * 4 && MV.cols() == 1);
                MV.array() /= 4.;
                break;
            case MASSMATRIX_TYPE_VORONOI:
            {
                assert(false);
                //MI = decltype(MI)::LinSpaced(n, 0, n - 1);
                //MJ = MI;
                //voronoi_mass(V, F, MV);
                break;
            }
            case MASSMATRIX_TYPE_FULL:
                MI.resize(m * 16, 1); MJ.resize(m * 16, 1); MV.resize(m * 16, 1);
                // indicies and values of the element mass matrix entries in the order
                // (1,0),(2,0),(3,0),(2,1),(3,1),(0,1),(3,2),(0,2),(1,2),(0,3),(1,3),(2,3),(0,0),(1,1),(2,2),(3,3);
                MI << F.col(1), F.col(2), F.col(3), F.col(2), F.col(3), F.col(0), F.col(3), F.col(0), F.col(1), F.col(0), F.col(1), F.col(2), F.col(0), F.col(1), F.col(2), F.col(3);
                MJ << F.col(0), F.col(0), F.col(0), F.col(1), F.col(1), F.col(1), F.col(2), F.col(2), F.col(2), F.col(3), F.col(3), F.col(3), F.col(0), F.col(1), F.col(2), F.col(3);
                repmat(vol, 16, 1, MV);
                assert(MV.rows() == m * 16 && MV.cols() == 1);
                MV.block(0 * m, 0, 12 * m, 1) /= 20.;
                MV.block(12 * m, 0, 4 * m, 1) /= 10.;
                break;
            default:
                assert(false && "Unknown Mass matrix eff_type");
            }
            sparse(MI, MJ, MV, n, n, M);
        }
    };

    // General template for handling Eigen::Dynamic at runtime
    template <typename DerivedV, typename DerivedF, typename Scalar>
    struct MassMatrixHelper<DerivedV, DerivedF, Scalar, Eigen::Dynamic> {
        static void compute(
            const Eigen::MatrixBase<DerivedV>& V,
            const Eigen::MatrixBase<DerivedF>& F,
            const MassMatrixType type,
            Eigen::SparseMatrix<Scalar>& M)
        {
            if (F.cols() == 3) {
                MassMatrixHelper<DerivedV, DerivedF, Scalar, 3>::compute(V, F, type, M);
            }
            else if (F.cols() == 4) {
                MassMatrixHelper<DerivedV, DerivedF, Scalar, 4>::compute(V, F, type, M);
            }
            else {
                // Handle unsupported simplex size at runtime
                assert(false && "Unsupported simplex size");
            }
        }
    };

    template <typename DerivedV, typename DerivedF, typename Scalar>
    inline void massmatrix(
        const Eigen::MatrixBase<DerivedV>& V,
        const Eigen::MatrixBase<DerivedF>& F,
        const MassMatrixType type,
        Eigen::SparseMatrix<Scalar>& M)
    {
        MassMatrixHelper<DerivedV, DerivedF, Scalar, DerivedF::ColsAtCompileTime>::compute(V, F, type, M);
    }






} // namespace ipc::rigid
