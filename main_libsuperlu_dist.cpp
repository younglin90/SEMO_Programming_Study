// pddrive_ABdist_eigen_rowmajor.cpp
// 분산 입력(ABdist) + Eigen::SparseMatrix(RowMajor/CSR) → SuperLU_DIST pdgssvx

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Sparse>

extern "C" {
#include "superlu_ddefs.h"        // SuperLU_DIST (double)
#ifdef __has_include
#  if __has_include("superlu_dist_config.h")
#    include "superlu_dist_config.h"  // SUPERLU_DIST_* 버전 매크로 있을 수 있음
#  endif
#endif
}

// -----------------------------
// 글로벌 연속 행 블록 분배
// -----------------------------
static inline void block_partition(int_t n_glob, int nprocs, int iam,
    int_t& m_loc, int_t& fst_row) {
    int_t base = n_glob / nprocs;
    int_t rem = n_glob % nprocs;
    m_loc = base + (iam < rem ? 1 : 0);
    fst_row = (iam < rem)
        ? iam * (base + 1)
        : rem * (base + 1) + (iam - rem) * base;
}

// -----------------------------
// 로컬 CSR(Eigen RowMajor) 생성: 2D nx×ny 5-포인트 라플라시안
// (대각 = 이웃수 + shift, 오프대각 = -1)
// - 이 랭크의 연속 행 블록 [fst_row, fst_row + m_loc)
// - 열 인덱스는 **글로벌** 인덱스여야 함
// -----------------------------
template<class StorageIndex = int_t>
static Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>
build_local_5pt_laplacian_rowmajor(int_t nx, int_t ny, double shift,
    int_t m_loc, int_t fst_row) {
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>;
    using Triplet = Eigen::Triplet<double, StorageIndex>;
    const StorageIndex n_glob = static_cast<StorageIndex>(nx * ny);

    std::vector<Triplet> T;
    T.reserve(static_cast<size_t>(m_loc) * 5);

    auto ix_of = [&](StorageIndex i)->StorageIndex { return i % nx; };
    auto iy_of = [&](StorageIndex i)->StorageIndex { return i / nx; };

    for (StorageIndex r = 0; r < (StorageIndex)m_loc; ++r) {
        StorageIndex i = static_cast<StorageIndex>(fst_row) + r;  // 글로벌 행
        StorageIndex ix = ix_of(i), iy = iy_of(i);

        if (iy > 0)        T.emplace_back(i, i - nx, -1.0); // 하
        if (ix > 0)        T.emplace_back(i, i - 1, -1.0); // 좌
        {   // 대각
            int deg = 0;
            if (iy > 0)     ++deg;
            if (ix > 0)     ++deg;
            if (ix + 1 < nx)  ++deg;
            if (iy + 1 < ny)  ++deg;
            T.emplace_back(i, i, (double)deg + shift);
        }
        if (ix + 1 < nx)    T.emplace_back(i, i + 1, -1.0); // 우
        if (iy + 1 < ny)    T.emplace_back(i, i + nx, -1.0); // 상
    }

    SpMat A_loc((StorageIndex)m_loc, n_glob);
    A_loc.setFromTriplets(T.begin(), T.end());
    A_loc.makeCompressed(); // CSR 포인터 고정 (RowMajor일 때 CSR)
    return A_loc;
}

int main(int argc, char* argv[]) try
{
    MPI_Init(&argc, &argv);

    // ---- 옵션 ----
    int_t nprow = 1, npcol = 1;        // SuperLU 2D grid (입력은 1D 연속 행 블록)
    int_t nx = 50, ny = 50;
    int    nrhs = 1;
    double shift = 1.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            std::printf("Options:\n");
            std::printf("\t-r <int>     : process rows      (default " IFMT ")\n", nprow);
            std::printf("\t-c <int>     : process cols      (default " IFMT ")\n", npcol);
            std::printf("\t-nx <int>    : grid Nx           (default " IFMT ")\n", nx);
            std::printf("\t-ny <int>    : grid Ny           (default " IFMT ")\n", ny);
            std::printf("\t-shift <real>: diagonal shift    (default %.3f)\n", shift);
            std::printf("\t-nrhs <int>  : # of RHS cols     (default %d)\n", nrhs);
            MPI_Finalize(); return 0;
        }
        else if (a == "-r" && i + 1 < argc) nprow = (int_t)std::atoi(argv[++i]);
        else if (a == "-c" && i + 1 < argc) npcol = (int_t)std::atoi(argv[++i]);
        else if (a == "-nx" && i + 1 < argc) nx = (int_t)std::atoi(argv[++i]);
        else if (a == "-ny" && i + 1 < argc) ny = (int_t)std::atoi(argv[++i]);
        else if (a == "-shift" && i + 1 < argc) shift = std::atof(argv[++i]);
        else if (a == "-nrhs" && i + 1 < argc) nrhs = std::atoi(argv[++i]);
    }
    if (nrhs <= 0) nrhs = 1;

    // ---- SuperLU 프로세스 그리드 ----
    gridinfo_t grid;
    superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);
    const int iam = grid.iam;
    const int nprocs = grid.nprow * grid.npcol;
    if (iam >= nprocs) { superlu_gridexit(&grid); MPI_Finalize(); return 0; }

    const int_t m_glob = nx * ny;
    const int_t n_glob = m_glob;

    // ---- 로컬 연속 행 블록 ----
    int_t m_loc = 0, fst_row = 0;
    block_partition(m_glob, nprocs, iam, m_loc, fst_row);

    // ---- 로컬 Eigen CSR(RowMajor) 생성 ----
    using StorageIndex = int_t; // SuperLU의 int_t와 폭 일치
    static_assert(std::is_same<StorageIndex, int_t>::value, "Index type mismatch.");

    using SpMatRM = Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>;
    SpMatRM A_loc = build_local_5pt_laplacian_rowmajor<StorageIndex>(nx, ny, shift, m_loc, fst_row);

    // Eigen RowMajor(CSR)의 내부 포인터:
    //  - valuePtr()        : nzval (nnz_loc)
    //  - innerIndexPtr()   : colind (nnz_loc)  ← 글로벌 열 인덱스
    //  - outerIndexPtr()   : rowptr (m_loc+1)  ← 로컬 행 시작 포인터(0부터)
    const int_t nnz_loc = static_cast<int_t>(A_loc.nonZeros());
    double* nzval = const_cast<double*>(A_loc.valuePtr());
    int_t* colind = const_cast<int_t*>(reinterpret_cast<const int_t*>(A_loc.innerIndexPtr()));
    int_t* rowptr = const_cast<int_t*>(reinterpret_cast<const int_t*>(A_loc.outerIndexPtr()));

    // ---- SuperLU 분산 입력(CompRowLoc) 생성 ----
    SuperMatrix A;
    dCreate_CompRowLoc_Matrix_dist(&A,
        m_glob, n_glob, nnz_loc, m_loc, fst_row,
        nzval, colind, rowptr,
        SLU_NR_loc, SLU_D, SLU_GE);

    // ---- RHS: x_true=1 ⇒ b=shift (각 로컬 행 동일) ----
    const int ldb = (int)m_loc;
    std::vector<double> B((size_t)m_loc * nrhs, shift);
    std::vector<double> berr((size_t)nrhs, 0.0);

    // ---- 옵션/구조체 ----
    superlu_dist_options_t options; set_default_options_dist(&options);
    dScalePermstruct_t ScalePermstruct; dLUstruct_t LUstruct;
    dScalePermstructInit(m_glob, n_glob, &ScalePermstruct);
    dLUstructInit(n_glob, &LUstruct);
    SuperLUStat_t stat; PStatInit(&stat);

    int info = 0;

    // ---- 버전 차이: SOLVEstruct_t 유무 처리 ----
#if defined(SUPERLU_DIST_MAJOR_VERSION) && (SUPERLU_DIST_MAJOR_VERSION >= 6)
    dSOLVEstruct_t SOLVEstruct;
    pdgssvx(&options, &A,
        &ScalePermstruct,
        B.data(), ldb, nrhs,
        &grid,
        &LUstruct, &SOLVEstruct,
        berr.data(),
        &stat, &info);
#else
    pdgssvx(&options, &A,
        &ScalePermstruct,
        B.data(), ldb, nrhs,
        &grid,
        &LUstruct,
        berr.data(),
        &stat, &info);
#endif

    if (info && iam == 0) std::fprintf(stderr, "pdgssvx failed: info=%d\n", info);

    // ---- 분산 ∞-norm 상대 오차: ||X-1||_∞ ----
    double loc_max = 0.0;
    for (int rhs = 0; rhs < nrhs; ++rhs) {
        for (int_t i = 0; i < m_loc; ++i) {
            double xi = B[(size_t)rhs * ldb + i];
            double err = std::abs(xi - 1.0);
            if (err > loc_max) loc_max = err;
        }
    }
    double glob_max = 0.0;
    MPI_Allreduce(&loc_max, &glob_max, 1, MPI_DOUBLE, MPI_MAX, grid.comm);
    if (iam == 0) std::printf("Relative inf-norm error vs 1: %e\n", glob_max);

    PStatPrint(&options, &stat, &grid);

    // ---- 정리 ----
    PStatFree(&stat);
#if defined(SUPERLU_DIST_MAJOR_VERSION) && (SUPERLU_DIST_MAJOR_VERSION >= 6)
    dSolveFinalize(&options, &SOLVEstruct);
#endif
    dDestroy_LU(n_glob, &grid, &LUstruct);
    dScalePermstructFree(&ScalePermstruct);
    dLUstructFree(&LUstruct);

    // 입력 래퍼만 파괴(Eigen이 버퍼 소유)
    Destroy_SuperMatrix_Store_dist(&A);

    superlu_gridexit(&grid);
    MPI_Finalize();
    return (info == 0) ? 0 : 1;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
}
catch (...) {
    std::fprintf(stderr, "Fatal: unknown\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
}
