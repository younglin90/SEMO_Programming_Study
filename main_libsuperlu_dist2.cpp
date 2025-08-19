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

#include <windows.h>
#include <psapi.h>

extern "C" {
#include "superlu_ddefs.h"        // SuperLU_DIST (double)
#ifdef __has_include
#  if __has_include("superlu_dist_config.h")
#    include "superlu_dist_config.h"  // SUPERLU_DIST_* 버전 매크로 있을 수 있음
#  endif
#endif
}

size_t getMemoryUsageKB() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
        sizeof(pmc)))
    {
        return pmc.WorkingSetSize / 1024; // KB 단위
    }
    return 0;
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

    auto ix_of = [&](StorageIndex gi)->StorageIndex { return gi % nx; };
    auto iy_of = [&](StorageIndex gi)->StorageIndex { return gi / nx; };

    for (StorageIndex r = 0; r < (StorageIndex)m_loc; ++r) {
        // 전역 행 ID
        StorageIndex gi = static_cast<StorageIndex>(fst_row) + r;
        StorageIndex ix = ix_of(gi), iy = iy_of(gi);

        auto push = [&](StorageIndex gcol, double val) {
            // 행은 로컬 r, 열은 전역 gcol
            T.emplace_back(r, gcol, val);
            };

        if (iy > 0)        push(gi - nx, -1.0);  // 아래(남)
        if (ix > 0)        push(gi - 1, -1.0);  // 좌
        {
            int deg = 0;
            if (iy > 0)      ++deg;
            if (ix > 0)      ++deg;
            if (ix + 1 < nx) ++deg;
            if (iy + 1 < ny) ++deg;
            push(gi, (double)deg + shift);       // 대각
        }
        if (ix + 1 < nx)  push(gi + 1, -1.0);   // 우
        if (iy + 1 < ny)  push(gi + nx, -1.0);   // 위(북)
    }

    SpMat A_loc((StorageIndex)m_loc, n_glob);
    A_loc.setFromTriplets(T.begin(), T.end());
    A_loc.makeCompressed(); // RowMajor=CSR 고정
    return A_loc;
}

int main(int argc, char* argv[]) 
{
    MPI_Init(&argc, &argv);

    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    //------------------------- SuperLU 2D 프로세스 그리드 -------------------------
    gridinfo_t grid;
    {
        int_t nprow = 1, npcol = size; // 1 x P (입력은 연속 행 블록)
        superlu_gridinit(MPI_COMM_WORLD, (int)nprow, (int)npcol, &grid);
    }
    const int iam = grid.iam;
    const int nprocs = grid.nprow * grid.npcol;
    if (iam >= nprocs) { superlu_gridexit(&grid); MPI_Finalize(); return 0; }

    //------------------------- 전역 문제 셋업 -------------------------
    int_t nx = 10, ny = 10;
    const int_t m_glob = nx * ny;
    const int_t n_glob = m_glob;
    const int   nrhs = 1;
    const double shift = 1.0;

    //------------------------- SuperLU 옵션/구조체 -------------------------
    superlu_dist_options_t options;
    set_default_options_dist(&options);
    // 기본값: options.Fact = DOFACT (매번 요인분해)
    options.ColPerm = NATURAL;      // (필요시 METIS_AT_PLUS_A 등)
    options.RowPerm = NOROWPERM;    // 또는 LargeDiag
    options.Equil = NO;
    options.IterRefine = NOREFINE;
    options.PrintStat = NO;           // 통계 프린트 억제
    options.ParSymbFact = NO;

    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t        LUstruct;
    SuperLUStat_t      stat;
    dSOLVEstruct_t     SOLVEstruct; // pdgssvx가 내부에서 채움(첫 호출시)

    dScalePermstructInit(m_glob, n_glob, &ScalePermstruct);
    dLUstructInit(n_glob, &LUstruct);
    PStatInit(&stat);










    //// 가상의 반복 계산 루프 (여기서 행렬 사이즈가 변경된다고 가정)
    //for (int it = 0; it < 1000000; ++it) {


    //    // ------------------------- SuperLU 2D 프로세스 그리드 재설정 -------------------------
    //    // (프로세스 그리드가 행렬 사이즈에 따라 달라질 수 있다면)
    //    // 현재 코드에서는 grid.nprow, grid.npcol이 MPI_COMM_WORLD_SIZE에 고정되어 있어
    //    // 행렬 사이즈 변화에 직접 영향을 받지 않으므로, 사실상 매번 재초기화할 필요는 없을 수 있습니다.
    //    // 하지만 grid.iam >= nprocs 조건처럼 프로세스 수가 행렬 크기에 따라 제한될 수 있다면
    //    // 재설정하는 것이 안전합니다.
    //    // 여기서는 간단히 다시 초기화하는 것으로 가정합니다.
    //    int_t nprow = 1, npcol = size; // 1 x P (입력은 연속 행 블록)
    //    superlu_gridinit(MPI_COMM_WORLD, (int)nprow, (int)npcol, &grid);
    //    const int iam = grid.iam;
    //    const int nprocs = grid.nprow * grid.npcol;
    //    if (iam >= nprocs) { superlu_gridexit(&grid); MPI_Finalize(); return 0; }


    //    // ------------------------- SuperLU 옵션/구조체 다시 설정 -------------------------
    //    // 행렬 사이즈가 변경되었으므로, 모든 구조체를 새 사이즈에 맞게 초기화해야 합니다.
    //    set_default_options_dist(&options); // 옵션도 초기 상태로 되돌립니다.
    //    options.ColPerm = NATURAL;
    //    options.RowPerm = NOROWPERM;
    //    options.Equil = NO;
    //    options.IterRefine = NOREFINE;
    //    options.PrintStat = NO;
    //    options.ParSymbFact = NO;
    //    // 행렬 사이즈가 변경되었으므로, 무조건 DOFACT여야 합니다. (기본값이므로 생략 가능)
    //    options.Fact = DOFACT; // 또는 options.Fact = NOTRANS;

    //    dScalePermstructInit(m_glob, n_glob, &ScalePermstruct);
    //    dLUstructInit(n_glob, &LUstruct);
    //    PStatInit(&stat); // Stat도 초기화 (이전 Stat를 해제했으므로)


    //    // 로컬 연속 행 분할
    //    int_t m_loc = 0, fst_row = 0;
    //    block_partition(m_glob, size, rank, m_loc, fst_row);

    //    if (rank == 0 && it == 0) {
    //        std::cout << "[grid] nprow=" << grid.nprow
    //            << " npcol=" << grid.npcol << "\n";
    //    }
    //    // 디버그용: 각 랭크의 블록 범위
    //    // std::cout << "[rank " << rank << "] m_loc=" << m_loc << " fst_row=" << fst_row << "\n";

    //    // 로컬 CSR 생성 (전역 열 인덱스 사용)
    //    using StorageIndex = int_t;
    //    static_assert(std::is_same<StorageIndex, int_t>::value, "Index type mismatch with SuperLU int_t");
    //    Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>
    //        A_loc = build_local_5pt_laplacian_rowmajor<StorageIndex>(nx, ny, shift, m_loc, fst_row);

    //    // SuperLU 분산 입력(CompRowLoc) 구성
    //    //  - nzval          : values (double*)
    //    //  - colind         : 전역 열 인덱스(int_t*)
    //    //  - rowptr         : 로컬 행 기준 CSR 포인터(길이 m_loc+1, 0-기반)
    //    const int_t nnz_loc = (int_t)A_loc.nonZeros();
    //    double* nzval = A_loc.valuePtr();
    //    int_t* colind = A_loc.innerIndexPtr();
    //    int_t* rowptr = A_loc.outerIndexPtr();

    //    SuperMatrix A; // SLU_NR_loc, SLU_D, SLU_GE
    //    dCreate_CompRowLoc_Matrix_dist(&A,
    //        m_glob, n_glob, nnz_loc, m_loc, fst_row,
    //        nzval, colind, rowptr,
    //        SLU_NR_loc, SLU_D, SLU_GE);

    //    // RHS b: x_true = 1 → A*1 = (대각 합) 이지만 여기서는 간단히 b=shift로 두고 검증
    //    // (진짜 검증을 하려면 b = A * ones(n) 을 만들어야 정확)
    //    std::vector<double> B((size_t)m_loc, shift);
    //    std::vector<double> berr(1, 0.0);
    //    const int ldb = (int)m_loc;



    //    // 풀기
    //    int info = 0;
    //    pdgssvx(&options, &A,
    //        &ScalePermstruct,
    //        B.data(), ldb, nrhs,
    //        &grid,
    //        &LUstruct, &SOLVEstruct,
    //        berr.data(),
    //        &stat, &info);

    //    Destroy_SuperMatrix_Store_dist(&A);

    //    // PStatFree(&stat); // Stat는 루프 밖에서 초기화/해제되는 경우가 많으므로 여기에 두지 않는 것이 좋음
    //    dScalePermstructFree(&ScalePermstruct);
    //    dDestroy_LU(n_glob, &grid, &LUstruct); // 이전 N_glob 사용
    //    dLUstructFree(&LUstruct);
    //    dSolveFinalize(&options, &SOLVEstruct);
    //    superlu_gridexit(&grid); // 그리드도 재설정해야 할 수 있음 (아래 설명)


    //    // (선택) 메모리 통계
    //    std::uint64_t local_kb = getMemoryUsageKB();
    //    std::uint64_t max_kb = 0, sum_kb = 0;
    //    MPI_Reduce(&local_kb, &max_kb, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    //    MPI_Reduce(&local_kb, &sum_kb, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    //    if (rank == 0) {
    //        std::cout << "[it " << it << "] Max RSS: " << max_kb
    //            << " KB, Sum RSS: " << sum_kb << " KB\n";
    //    }

    //}

    //// ------------------------- 반복문 종료 후 최종 자원 해제 -------------------------
    //// 마지막 반복에서 할당된 자원 해제
    //PStatFree(&stat);
    //dScalePermstructFree(&ScalePermstruct);
    //dDestroy_LU(n_glob, &grid, &LUstruct);
    //dLUstructFree(&LUstruct);
    //dSolveFinalize(&options, &SOLVEstruct);
    //superlu_gridexit(&grid);








    //------------------------- 반복 계산 루프 -------------------------
    for (int it = 0; it < 1000000; ++it) {
        // (패턴/그리드 동일 가정. 값만 바꾸려면 shift 등 변경)

        // 로컬 연속 행 분할
        int_t m_loc = 0, fst_row = 0;
        block_partition(m_glob, size, rank, m_loc, fst_row);

        if (rank == 0 && it == 0) {
            std::cout << "[grid] nprow=" << grid.nprow
                << " npcol=" << grid.npcol << "\n";
        }
        // 디버그용: 각 랭크의 블록 범위
        // std::cout << "[rank " << rank << "] m_loc=" << m_loc << " fst_row=" << fst_row << "\n";

        // 로컬 CSR 생성 (전역 열 인덱스 사용)
        using StorageIndex = int_t;
        static_assert(std::is_same<StorageIndex, int_t>::value, "Index type mismatch with SuperLU int_t");
        Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>
            A_loc = build_local_5pt_laplacian_rowmajor<StorageIndex>(nx, ny, shift, m_loc, fst_row);

        // SuperLU 분산 입력(CompRowLoc) 구성
        //  - nzval          : values (double*)
        //  - colind         : 전역 열 인덱스(int_t*)
        //  - rowptr         : 로컬 행 기준 CSR 포인터(길이 m_loc+1, 0-기반)
        const int_t nnz_loc = (int_t)A_loc.nonZeros();
        double* nzval = A_loc.valuePtr();
        int_t* colind = A_loc.innerIndexPtr();
        int_t* rowptr = A_loc.outerIndexPtr();

        SuperMatrix A; // SLU_NR_loc, SLU_D, SLU_GE
        dCreate_CompRowLoc_Matrix_dist(&A,
            m_glob, n_glob, nnz_loc, m_loc, fst_row,
            nzval, colind, rowptr,
            SLU_NR_loc, SLU_D, SLU_GE);

        // RHS b: x_true = 1 → A*1 = (대각 합) 이지만 여기서는 간단히 b=shift로 두고 검증
        // (진짜 검증을 하려면 b = A * ones(n) 을 만들어야 정확)
        std::vector<double> B((size_t)m_loc, shift);
        std::vector<double> berr(1, 0.0);
        const int ldb = (int)m_loc;

        // 풀기
        int info = 0;
        pdgssvx(&options, &A,
            &ScalePermstruct,
            B.data(), ldb, nrhs,
            &grid,
            &LUstruct, &SOLVEstruct,
            berr.data(),
            &stat, &info);

        if (info != 0) {
            std::cerr << "[rank " << rank << "] pdgssvx info=" << info << "\n";
            // 실패시 루프 중단
            Destroy_SuperMatrix_Store_dist(&A);
            break;
        }

        // (선택) berr[0] 출력: 후행 정제 사용 시 유효. NOREFINE이면 의미 제한적.
        // if (rank == 0) std::cout << "berr=" << berr[0] << "\n";

        // SuperMatrix A 정리
        // !! 중요: dCreate_CompRowLoc_Matrix_dist 로 만든 것은 이걸로 파괴 !!
        PStatFree(&stat);
        Destroy_SuperMatrix_Store_dist(&A);
        dDestroy_LU(n_glob, &grid, &LUstruct);
        options.Fact = SamePattern;
        PStatInit(&stat);
        //dSolveFinalize(&options, &SOLVEstruct); 
        //options.Fact = SamePattern_SameRowPerm;


        // (선택) 메모리 통계
        std::uint64_t local_kb = getMemoryUsageKB();
        std::uint64_t max_kb = 0, sum_kb = 0;
        MPI_Reduce(&local_kb, &max_kb, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_kb, &sum_kb, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "[it " << it << "] Max RSS: " << max_kb
                << " KB, Sum RSS: " << sum_kb << " KB\n";
        }

    }

    //------------------------- 자원 해제 -------------------------
    PStatFree(&stat);
    dScalePermstructFree(&ScalePermstruct);

    // 요인/통신 구조물 정리
    dDestroy_LU(n_glob, &grid, &LUstruct);
    dLUstructFree(&LUstruct);

    // SOLVEstruct 정리 (pdgssvx가 내부 생성)
    dSolveFinalize(&options, &SOLVEstruct);

    superlu_gridexit(&grid);


















//
//    {
//        // 1D Row x Col 그리드. 여기선 (1 x nprocs)로 두는 것이 무난.
//        gridinfo_t grid;
//        superlu_gridinit(MPI_COMM_WORLD, 1, size, &grid);
//        if (grid.iam == -1) throw std::runtime_error("superlu_gridinit failed");
//
//        //----- SuperLU 옵션 -----
//        superlu_dist_options_t options;
//        set_default_options_dist(&options);
//        // 필요 시 옵션 커스터마이즈:
//
//        options.PrintStat = NO;                      // 로그 억제
//
//        //----- 행렬 NR_loc 생성 (CSR → CompRowLoc) -----
//        // SuperLU_DIST는 로컬 행렬을 다음으로 요구:
//        // dCreate_CompRowLoc_Matrix_dist(&A, M, N, nnz_loc, m_loc, fst_row, a, colind, rowptr, SLU_NR_loc, SLU_D, SLU_GE)
//        const int_t m_loc = rows;
//        const int_t N = mesh_.ncells_global, M = mesh_.ncells_global;
//        const int_t fst_row = mesh_.start_proc_cell_global;
//
//
//        // nnz_loc
//        int_t nnz_loc = static_cast<int_t>(A_matrix.i_str_CSR[rows] - A_matrix.i_str_CSR[0]);
//
//        // SuperLU int_t로 보장 (입력이 int64면 복사/다운캐스트)
//        std::vector<int_t> rowptr_t(rows + 1);
//        for (std::size_t i = 0; i < rows + 1; ++i) rowptr_t[i] = static_cast<int_t>(A_matrix.i_str_CSR[i]);
//        std::vector<int_t> colind_t(nnz_loc);
//        for (std::size_t i = 0; i < nnz_loc; ++i) colind_t[i] = static_cast<int_t>(A_matrix.j_displ_CSR[i]);
//
//        // SuperMatrix A (분산 NR_loc)
//        SuperMatrix A;
//        dCreate_CompRowLoc_Matrix_dist(
//            &A, M, N, nnz_loc, m_loc, fst_row,
//            A_matrix.m_nonZero.data(),
//            colind_t.data(),
//            rowptr_t.data(),
//            SLU_NR_loc, SLU_D, SLU_GE);
//
//        //----- RHS / 해 생성 -----
//        // SuperLU는 B, X를 column-major (ldb = m_loc) 로 기대.
//        // 여기서는 nrhs=1 가정, 여러 RHS면 B, X를 (m_loc x nrhs)로 준비.
//
//        // 분산 RHS/X 컨테이너
//        // Create_Dense_Matrix_dist 는 프로세스별 로컬 블록 포인터를 받음
//        //SuperMatrix Bmat, Xmat;
//        //dCreate_Dense_Matrix_dist(&Bmat, m_loc, 1, B_vector.data(), m_loc, SLU_DN, SLU_D, SLU_GE);
//        //dCreate_Dense_Matrix_dist(&Xmat, m_loc, 1, dX.data(), m_loc, SLU_DN, SLU_D, SLU_GE);
//
//        //----- 구조체들 -----
//        dScalePermstruct_t ScalePermstruct;
//        dLUstruct_t        LUstruct;
//        dSOLVEstruct_t     SOLVEstruct;
//        SuperLUStat_t     stat;
//        std::vector<double> berr(1, 0.0);
//        int               info = 0;
//
//        dScalePermstructInit(M, N, &ScalePermstruct);  // (사실 정방: M=N)
//        dLUstructInit(N, &LUstruct);
//
//        PStatInit(&stat);  // 통계
//
//        //----- Solve -----
//        // pdgssvx: 분산 NR_loc 입력을 받아 LU 분해 + 풀이
//        pdgssvx(&options,
//            &A,
//            &ScalePermstruct,
//            B_vector.data(), m_loc, 1, &grid,
//            &LUstruct, &SOLVEstruct,
//            berr.data(),
//            &stat, &info);
//
//        if (info != 0) {
//            // info > 0: singular pivot 등, info < 0: 잘못된 인자
//            superlu_gridexit(&grid);
//            throw std::runtime_error("pdgssvx failed, info = " + std::to_string(info));
//        }
//
//
//        //----- 자원 해제 -----
//        // (A, B, X 의 실제 값배열은 사용자 소유/스택이므로 파괴자에서 해제하지 않음)
//        //Destroy_SuperMatrix_Store_dist(&Bmat);
//        //Destroy_SuperMatrix_Store_dist(&Xmat);
//        //Destroy_CompRowLoc_Matrix_dist(&A);
//        Destroy_SuperMatrix_Store_dist(&A);
//
//        // LU data
//        dDestroy_LU(N, &grid, &LUstruct);
//        dScalePermstructFree(&ScalePermstruct);
//        dLUstructFree(&LUstruct);
//
//        if (options.SolveInitialized) {
//            dSolveFinalize(&options, &SOLVEstruct);
//        }
//
//        PStatFree(&stat);
//        superlu_gridexit(&grid);
//
//
//        for (long long row = 0; row < rows; ++row) dX[row] = B_vector[row];
//    }
//
//
//    // ---- 옵션 ----
//    int_t nprow = 1, npcol = 1;        // SuperLU 2D grid (입력은 1D 연속 행 블록)
//    int_t nx = 50, ny = 50;
//    int    nrhs = 1;
//    double shift = 1.0;
//
//    for (int i = 1; i < argc; ++i) {
//        std::string a = argv[i];
//        if (a == "-h" || a == "--help") {
//            std::printf("Options:\n");
//            std::printf("\t-r <int>     : process rows      (default " IFMT ")\n", nprow);
//            std::printf("\t-c <int>     : process cols      (default " IFMT ")\n", npcol);
//            std::printf("\t-nx <int>    : grid Nx           (default " IFMT ")\n", nx);
//            std::printf("\t-ny <int>    : grid Ny           (default " IFMT ")\n", ny);
//            std::printf("\t-shift <real>: diagonal shift    (default %.3f)\n", shift);
//            std::printf("\t-nrhs <int>  : # of RHS cols     (default %d)\n", nrhs);
//            MPI_Finalize(); return 0;
//        }
//        else if (a == "-r" && i + 1 < argc) nprow = (int_t)std::atoi(argv[++i]);
//        else if (a == "-c" && i + 1 < argc) npcol = (int_t)std::atoi(argv[++i]);
//        else if (a == "-nx" && i + 1 < argc) nx = (int_t)std::atoi(argv[++i]);
//        else if (a == "-ny" && i + 1 < argc) ny = (int_t)std::atoi(argv[++i]);
//        else if (a == "-shift" && i + 1 < argc) shift = std::atof(argv[++i]);
//        else if (a == "-nrhs" && i + 1 < argc) nrhs = std::atoi(argv[++i]);
//    }
//    if (nrhs <= 0) nrhs = 1;
//
//    // ---- SuperLU 프로세스 그리드 ----
//    gridinfo_t grid;
//    superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);
//    const int iam = grid.iam;
//    const int nprocs = grid.nprow * grid.npcol;
//    if (iam >= nprocs) { superlu_gridexit(&grid); MPI_Finalize(); return 0; }
//
//    const int_t m_glob = nx * ny;
//    const int_t n_glob = m_glob;
//
//    // ---- 로컬 연속 행 블록 ----
//    int_t m_loc = 0, fst_row = 0;
//    block_partition(m_glob, nprocs, iam, m_loc, fst_row);
//
//    // ---- 로컬 Eigen CSR(RowMajor) 생성 ----
//    using StorageIndex = int_t; // SuperLU의 int_t와 폭 일치
//    static_assert(std::is_same<StorageIndex, int_t>::value, "Index type mismatch.");
//
//    using SpMatRM = Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>;
//    SpMatRM A_loc = build_local_5pt_laplacian_rowmajor<StorageIndex>(nx, ny, shift, m_loc, fst_row);
//
//    // Eigen RowMajor(CSR)의 내부 포인터:
//    //  - valuePtr()        : nzval (nnz_loc)
//    //  - innerIndexPtr()   : colind (nnz_loc)  ← 글로벌 열 인덱스
//    //  - outerIndexPtr()   : rowptr (m_loc+1)  ← 로컬 행 시작 포인터(0부터)
//    const int_t nnz_loc = static_cast<int_t>(A_loc.nonZeros());
//    double* nzval = const_cast<double*>(A_loc.valuePtr());
//    int_t* colind = const_cast<int_t*>(reinterpret_cast<const int_t*>(A_loc.innerIndexPtr()));
//    int_t* rowptr = const_cast<int_t*>(reinterpret_cast<const int_t*>(A_loc.outerIndexPtr()));
//
//    // ---- SuperLU 분산 입력(CompRowLoc) 생성 ----
//    SuperMatrix A;
//    dCreate_CompRowLoc_Matrix_dist(&A,
//        m_glob, n_glob, nnz_loc, m_loc, fst_row,
//        nzval, colind, rowptr,
//        SLU_NR_loc, SLU_D, SLU_GE);
//
//    // ---- RHS: x_true=1 ⇒ b=shift (각 로컬 행 동일) ----
//    const int ldb = (int)m_loc;
//    std::vector<double> B((size_t)m_loc * nrhs, shift);
//    std::vector<double> berr((size_t)nrhs, 0.0);
//
//    // ---- 옵션/구조체 ----
//    superlu_dist_options_t options; set_default_options_dist(&options);
//    dScalePermstruct_t ScalePermstruct; dLUstruct_t LUstruct;
//    dScalePermstructInit(m_glob, n_glob, &ScalePermstruct);
//    dLUstructInit(n_glob, &LUstruct);
//    SuperLUStat_t stat; PStatInit(&stat);
//
//    int info = 0;
//
//    // ---- 버전 차이: SOLVEstruct_t 유무 처리 ----
//#if defined(SUPERLU_DIST_MAJOR_VERSION) && (SUPERLU_DIST_MAJOR_VERSION >= 6)
//    dSOLVEstruct_t SOLVEstruct;
//    pdgssvx(&options, &A,
//        &ScalePermstruct,
//        B.data(), ldb, nrhs,
//        &grid,
//        &LUstruct, &SOLVEstruct,
//        berr.data(),
//        &stat, &info);
//#else
//    pdgssvx(&options, &A,
//        &ScalePermstruct,
//        B.data(), ldb, nrhs,
//        &grid,
//        &LUstruct,
//        berr.data(),
//        &stat, &info);
//#endif
//
//    if (info && iam == 0) std::fprintf(stderr, "pdgssvx failed: info=%d\n", info);
//
//    // ---- 분산 ∞-norm 상대 오차: ||X-1||_∞ ----
//    double loc_max = 0.0;
//    for (int rhs = 0; rhs < nrhs; ++rhs) {
//        for (int_t i = 0; i < m_loc; ++i) {
//            double xi = B[(size_t)rhs * ldb + i];
//            double err = std::abs(xi - 1.0);
//            if (err > loc_max) loc_max = err;
//        }
//    }
//    double glob_max = 0.0;
//    MPI_Allreduce(&loc_max, &glob_max, 1, MPI_DOUBLE, MPI_MAX, grid.comm);
//    if (iam == 0) std::printf("Relative inf-norm error vs 1: %e\n", glob_max);
//
//    PStatPrint(&options, &stat, &grid);
//
//    // ---- 정리 ----
//    PStatFree(&stat);
//#if defined(SUPERLU_DIST_MAJOR_VERSION) && (SUPERLU_DIST_MAJOR_VERSION >= 6)
//    dSolveFinalize(&options, &SOLVEstruct);
//#endif
//    dDestroy_LU(n_glob, &grid, &LUstruct);
//    dScalePermstructFree(&ScalePermstruct);
//    dLUstructFree(&LUstruct);
//
//    // 입력 래퍼만 파괴(Eigen이 버퍼 소유)
//    Destroy_SuperMatrix_Store_dist(&A);
//
//    superlu_gridexit(&grid);
//    MPI_Finalize();
//    return (info == 0) ? 0 : 1;


    MPI_Finalize();
    return 0;
}

