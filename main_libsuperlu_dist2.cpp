// pddrive_ABdist_eigen_rowmajor.cpp
// �л� �Է�(ABdist) + Eigen::SparseMatrix(RowMajor/CSR) �� SuperLU_DIST pdgssvx

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
#    include "superlu_dist_config.h"  // SUPERLU_DIST_* ���� ��ũ�� ���� �� ����
#  endif
#endif
}

size_t getMemoryUsageKB() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
        sizeof(pmc)))
    {
        return pmc.WorkingSetSize / 1024; // KB ����
    }
    return 0;
}

// -----------------------------
// �۷ι� ���� �� ��� �й�
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
// ���� CSR(Eigen RowMajor) ����: 2D nx��ny 5-����Ʈ ���ö�þ�
// (�밢 = �̿��� + shift, �����밢 = -1)
// - �� ��ũ�� ���� �� ��� [fst_row, fst_row + m_loc)
// - �� �ε����� **�۷ι�** �ε������� ��
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
        // ���� �� ID
        StorageIndex gi = static_cast<StorageIndex>(fst_row) + r;
        StorageIndex ix = ix_of(gi), iy = iy_of(gi);

        auto push = [&](StorageIndex gcol, double val) {
            // ���� ���� r, ���� ���� gcol
            T.emplace_back(r, gcol, val);
            };

        if (iy > 0)        push(gi - nx, -1.0);  // �Ʒ�(��)
        if (ix > 0)        push(gi - 1, -1.0);  // ��
        {
            int deg = 0;
            if (iy > 0)      ++deg;
            if (ix > 0)      ++deg;
            if (ix + 1 < nx) ++deg;
            if (iy + 1 < ny) ++deg;
            push(gi, (double)deg + shift);       // �밢
        }
        if (ix + 1 < nx)  push(gi + 1, -1.0);   // ��
        if (iy + 1 < ny)  push(gi + nx, -1.0);   // ��(��)
    }

    SpMat A_loc((StorageIndex)m_loc, n_glob);
    A_loc.setFromTriplets(T.begin(), T.end());
    A_loc.makeCompressed(); // RowMajor=CSR ����
    return A_loc;
}

int main(int argc, char* argv[]) 
{
    MPI_Init(&argc, &argv);

    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    //------------------------- SuperLU 2D ���μ��� �׸��� -------------------------
    gridinfo_t grid;
    {
        int_t nprow = 1, npcol = size; // 1 x P (�Է��� ���� �� ���)
        superlu_gridinit(MPI_COMM_WORLD, (int)nprow, (int)npcol, &grid);
    }
    const int iam = grid.iam;
    const int nprocs = grid.nprow * grid.npcol;
    if (iam >= nprocs) { superlu_gridexit(&grid); MPI_Finalize(); return 0; }

    //------------------------- ���� ���� �¾� -------------------------
    int_t nx = 10, ny = 10;
    const int_t m_glob = nx * ny;
    const int_t n_glob = m_glob;
    const int   nrhs = 1;
    const double shift = 1.0;

    //------------------------- SuperLU �ɼ�/����ü -------------------------
    superlu_dist_options_t options;
    set_default_options_dist(&options);
    // �⺻��: options.Fact = DOFACT (�Ź� ���κ���)
    options.ColPerm = NATURAL;      // (�ʿ�� METIS_AT_PLUS_A ��)
    options.RowPerm = NOROWPERM;    // �Ǵ� LargeDiag
    options.Equil = NO;
    options.IterRefine = NOREFINE;
    options.PrintStat = NO;           // ��� ����Ʈ ����
    options.ParSymbFact = NO;

    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t        LUstruct;
    SuperLUStat_t      stat;
    dSOLVEstruct_t     SOLVEstruct; // pdgssvx�� ���ο��� ä��(ù ȣ���)

    dScalePermstructInit(m_glob, n_glob, &ScalePermstruct);
    dLUstructInit(n_glob, &LUstruct);
    PStatInit(&stat);










    //// ������ �ݺ� ��� ���� (���⼭ ��� ����� ����ȴٰ� ����)
    //for (int it = 0; it < 1000000; ++it) {


    //    // ------------------------- SuperLU 2D ���μ��� �׸��� �缳�� -------------------------
    //    // (���μ��� �׸��尡 ��� ����� ���� �޶��� �� �ִٸ�)
    //    // ���� �ڵ忡���� grid.nprow, grid.npcol�� MPI_COMM_WORLD_SIZE�� �����Ǿ� �־�
    //    // ��� ������ ��ȭ�� ���� ������ ���� �����Ƿ�, ��ǻ� �Ź� ���ʱ�ȭ�� �ʿ�� ���� �� �ֽ��ϴ�.
    //    // ������ grid.iam >= nprocs ����ó�� ���μ��� ���� ��� ũ�⿡ ���� ���ѵ� �� �ִٸ�
    //    // �缳���ϴ� ���� �����մϴ�.
    //    // ���⼭�� ������ �ٽ� �ʱ�ȭ�ϴ� ������ �����մϴ�.
    //    int_t nprow = 1, npcol = size; // 1 x P (�Է��� ���� �� ���)
    //    superlu_gridinit(MPI_COMM_WORLD, (int)nprow, (int)npcol, &grid);
    //    const int iam = grid.iam;
    //    const int nprocs = grid.nprow * grid.npcol;
    //    if (iam >= nprocs) { superlu_gridexit(&grid); MPI_Finalize(); return 0; }


    //    // ------------------------- SuperLU �ɼ�/����ü �ٽ� ���� -------------------------
    //    // ��� ����� ����Ǿ����Ƿ�, ��� ����ü�� �� ����� �°� �ʱ�ȭ�ؾ� �մϴ�.
    //    set_default_options_dist(&options); // �ɼǵ� �ʱ� ���·� �ǵ����ϴ�.
    //    options.ColPerm = NATURAL;
    //    options.RowPerm = NOROWPERM;
    //    options.Equil = NO;
    //    options.IterRefine = NOREFINE;
    //    options.PrintStat = NO;
    //    options.ParSymbFact = NO;
    //    // ��� ����� ����Ǿ����Ƿ�, ������ DOFACT���� �մϴ�. (�⺻���̹Ƿ� ���� ����)
    //    options.Fact = DOFACT; // �Ǵ� options.Fact = NOTRANS;

    //    dScalePermstructInit(m_glob, n_glob, &ScalePermstruct);
    //    dLUstructInit(n_glob, &LUstruct);
    //    PStatInit(&stat); // Stat�� �ʱ�ȭ (���� Stat�� ���������Ƿ�)


    //    // ���� ���� �� ����
    //    int_t m_loc = 0, fst_row = 0;
    //    block_partition(m_glob, size, rank, m_loc, fst_row);

    //    if (rank == 0 && it == 0) {
    //        std::cout << "[grid] nprow=" << grid.nprow
    //            << " npcol=" << grid.npcol << "\n";
    //    }
    //    // ����׿�: �� ��ũ�� ��� ����
    //    // std::cout << "[rank " << rank << "] m_loc=" << m_loc << " fst_row=" << fst_row << "\n";

    //    // ���� CSR ���� (���� �� �ε��� ���)
    //    using StorageIndex = int_t;
    //    static_assert(std::is_same<StorageIndex, int_t>::value, "Index type mismatch with SuperLU int_t");
    //    Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>
    //        A_loc = build_local_5pt_laplacian_rowmajor<StorageIndex>(nx, ny, shift, m_loc, fst_row);

    //    // SuperLU �л� �Է�(CompRowLoc) ����
    //    //  - nzval          : values (double*)
    //    //  - colind         : ���� �� �ε���(int_t*)
    //    //  - rowptr         : ���� �� ���� CSR ������(���� m_loc+1, 0-���)
    //    const int_t nnz_loc = (int_t)A_loc.nonZeros();
    //    double* nzval = A_loc.valuePtr();
    //    int_t* colind = A_loc.innerIndexPtr();
    //    int_t* rowptr = A_loc.outerIndexPtr();

    //    SuperMatrix A; // SLU_NR_loc, SLU_D, SLU_GE
    //    dCreate_CompRowLoc_Matrix_dist(&A,
    //        m_glob, n_glob, nnz_loc, m_loc, fst_row,
    //        nzval, colind, rowptr,
    //        SLU_NR_loc, SLU_D, SLU_GE);

    //    // RHS b: x_true = 1 �� A*1 = (�밢 ��) ������ ���⼭�� ������ b=shift�� �ΰ� ����
    //    // (��¥ ������ �Ϸ��� b = A * ones(n) �� ������ ��Ȯ)
    //    std::vector<double> B((size_t)m_loc, shift);
    //    std::vector<double> berr(1, 0.0);
    //    const int ldb = (int)m_loc;



    //    // Ǯ��
    //    int info = 0;
    //    pdgssvx(&options, &A,
    //        &ScalePermstruct,
    //        B.data(), ldb, nrhs,
    //        &grid,
    //        &LUstruct, &SOLVEstruct,
    //        berr.data(),
    //        &stat, &info);

    //    Destroy_SuperMatrix_Store_dist(&A);

    //    // PStatFree(&stat); // Stat�� ���� �ۿ��� �ʱ�ȭ/�����Ǵ� ��찡 �����Ƿ� ���⿡ ���� �ʴ� ���� ����
    //    dScalePermstructFree(&ScalePermstruct);
    //    dDestroy_LU(n_glob, &grid, &LUstruct); // ���� N_glob ���
    //    dLUstructFree(&LUstruct);
    //    dSolveFinalize(&options, &SOLVEstruct);
    //    superlu_gridexit(&grid); // �׸��嵵 �缳���ؾ� �� �� ���� (�Ʒ� ����)


    //    // (����) �޸� ���
    //    std::uint64_t local_kb = getMemoryUsageKB();
    //    std::uint64_t max_kb = 0, sum_kb = 0;
    //    MPI_Reduce(&local_kb, &max_kb, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    //    MPI_Reduce(&local_kb, &sum_kb, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    //    if (rank == 0) {
    //        std::cout << "[it " << it << "] Max RSS: " << max_kb
    //            << " KB, Sum RSS: " << sum_kb << " KB\n";
    //    }

    //}

    //// ------------------------- �ݺ��� ���� �� ���� �ڿ� ���� -------------------------
    //// ������ �ݺ����� �Ҵ�� �ڿ� ����
    //PStatFree(&stat);
    //dScalePermstructFree(&ScalePermstruct);
    //dDestroy_LU(n_glob, &grid, &LUstruct);
    //dLUstructFree(&LUstruct);
    //dSolveFinalize(&options, &SOLVEstruct);
    //superlu_gridexit(&grid);








    //------------------------- �ݺ� ��� ���� -------------------------
    for (int it = 0; it < 1000000; ++it) {
        // (����/�׸��� ���� ����. ���� �ٲٷ��� shift �� ����)

        // ���� ���� �� ����
        int_t m_loc = 0, fst_row = 0;
        block_partition(m_glob, size, rank, m_loc, fst_row);

        if (rank == 0 && it == 0) {
            std::cout << "[grid] nprow=" << grid.nprow
                << " npcol=" << grid.npcol << "\n";
        }
        // ����׿�: �� ��ũ�� ��� ����
        // std::cout << "[rank " << rank << "] m_loc=" << m_loc << " fst_row=" << fst_row << "\n";

        // ���� CSR ���� (���� �� �ε��� ���)
        using StorageIndex = int_t;
        static_assert(std::is_same<StorageIndex, int_t>::value, "Index type mismatch with SuperLU int_t");
        Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>
            A_loc = build_local_5pt_laplacian_rowmajor<StorageIndex>(nx, ny, shift, m_loc, fst_row);

        // SuperLU �л� �Է�(CompRowLoc) ����
        //  - nzval          : values (double*)
        //  - colind         : ���� �� �ε���(int_t*)
        //  - rowptr         : ���� �� ���� CSR ������(���� m_loc+1, 0-���)
        const int_t nnz_loc = (int_t)A_loc.nonZeros();
        double* nzval = A_loc.valuePtr();
        int_t* colind = A_loc.innerIndexPtr();
        int_t* rowptr = A_loc.outerIndexPtr();

        SuperMatrix A; // SLU_NR_loc, SLU_D, SLU_GE
        dCreate_CompRowLoc_Matrix_dist(&A,
            m_glob, n_glob, nnz_loc, m_loc, fst_row,
            nzval, colind, rowptr,
            SLU_NR_loc, SLU_D, SLU_GE);

        // RHS b: x_true = 1 �� A*1 = (�밢 ��) ������ ���⼭�� ������ b=shift�� �ΰ� ����
        // (��¥ ������ �Ϸ��� b = A * ones(n) �� ������ ��Ȯ)
        std::vector<double> B((size_t)m_loc, shift);
        std::vector<double> berr(1, 0.0);
        const int ldb = (int)m_loc;

        // Ǯ��
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
            // ���н� ���� �ߴ�
            Destroy_SuperMatrix_Store_dist(&A);
            break;
        }

        // (����) berr[0] ���: ���� ���� ��� �� ��ȿ. NOREFINE�̸� �ǹ� ������.
        // if (rank == 0) std::cout << "berr=" << berr[0] << "\n";

        // SuperMatrix A ����
        // !! �߿�: dCreate_CompRowLoc_Matrix_dist �� ���� ���� �̰ɷ� �ı� !!
        PStatFree(&stat);
        Destroy_SuperMatrix_Store_dist(&A);
        dDestroy_LU(n_glob, &grid, &LUstruct);
        options.Fact = SamePattern;
        PStatInit(&stat);
        //dSolveFinalize(&options, &SOLVEstruct); 
        //options.Fact = SamePattern_SameRowPerm;


        // (����) �޸� ���
        std::uint64_t local_kb = getMemoryUsageKB();
        std::uint64_t max_kb = 0, sum_kb = 0;
        MPI_Reduce(&local_kb, &max_kb, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_kb, &sum_kb, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "[it " << it << "] Max RSS: " << max_kb
                << " KB, Sum RSS: " << sum_kb << " KB\n";
        }

    }

    //------------------------- �ڿ� ���� -------------------------
    PStatFree(&stat);
    dScalePermstructFree(&ScalePermstruct);

    // ����/��� ������ ����
    dDestroy_LU(n_glob, &grid, &LUstruct);
    dLUstructFree(&LUstruct);

    // SOLVEstruct ���� (pdgssvx�� ���� ����)
    dSolveFinalize(&options, &SOLVEstruct);

    superlu_gridexit(&grid);


















//
//    {
//        // 1D Row x Col �׸���. ���⼱ (1 x nprocs)�� �δ� ���� ����.
//        gridinfo_t grid;
//        superlu_gridinit(MPI_COMM_WORLD, 1, size, &grid);
//        if (grid.iam == -1) throw std::runtime_error("superlu_gridinit failed");
//
//        //----- SuperLU �ɼ� -----
//        superlu_dist_options_t options;
//        set_default_options_dist(&options);
//        // �ʿ� �� �ɼ� Ŀ���͸�����:
//
//        options.PrintStat = NO;                      // �α� ����
//
//        //----- ��� NR_loc ���� (CSR �� CompRowLoc) -----
//        // SuperLU_DIST�� ���� ����� �������� �䱸:
//        // dCreate_CompRowLoc_Matrix_dist(&A, M, N, nnz_loc, m_loc, fst_row, a, colind, rowptr, SLU_NR_loc, SLU_D, SLU_GE)
//        const int_t m_loc = rows;
//        const int_t N = mesh_.ncells_global, M = mesh_.ncells_global;
//        const int_t fst_row = mesh_.start_proc_cell_global;
//
//
//        // nnz_loc
//        int_t nnz_loc = static_cast<int_t>(A_matrix.i_str_CSR[rows] - A_matrix.i_str_CSR[0]);
//
//        // SuperLU int_t�� ���� (�Է��� int64�� ����/�ٿ�ĳ��Ʈ)
//        std::vector<int_t> rowptr_t(rows + 1);
//        for (std::size_t i = 0; i < rows + 1; ++i) rowptr_t[i] = static_cast<int_t>(A_matrix.i_str_CSR[i]);
//        std::vector<int_t> colind_t(nnz_loc);
//        for (std::size_t i = 0; i < nnz_loc; ++i) colind_t[i] = static_cast<int_t>(A_matrix.j_displ_CSR[i]);
//
//        // SuperMatrix A (�л� NR_loc)
//        SuperMatrix A;
//        dCreate_CompRowLoc_Matrix_dist(
//            &A, M, N, nnz_loc, m_loc, fst_row,
//            A_matrix.m_nonZero.data(),
//            colind_t.data(),
//            rowptr_t.data(),
//            SLU_NR_loc, SLU_D, SLU_GE);
//
//        //----- RHS / �� ���� -----
//        // SuperLU�� B, X�� column-major (ldb = m_loc) �� ���.
//        // ���⼭�� nrhs=1 ����, ���� RHS�� B, X�� (m_loc x nrhs)�� �غ�.
//
//        // �л� RHS/X �����̳�
//        // Create_Dense_Matrix_dist �� ���μ����� ���� ��� �����͸� ����
//        //SuperMatrix Bmat, Xmat;
//        //dCreate_Dense_Matrix_dist(&Bmat, m_loc, 1, B_vector.data(), m_loc, SLU_DN, SLU_D, SLU_GE);
//        //dCreate_Dense_Matrix_dist(&Xmat, m_loc, 1, dX.data(), m_loc, SLU_DN, SLU_D, SLU_GE);
//
//        //----- ����ü�� -----
//        dScalePermstruct_t ScalePermstruct;
//        dLUstruct_t        LUstruct;
//        dSOLVEstruct_t     SOLVEstruct;
//        SuperLUStat_t     stat;
//        std::vector<double> berr(1, 0.0);
//        int               info = 0;
//
//        dScalePermstructInit(M, N, &ScalePermstruct);  // (��� ����: M=N)
//        dLUstructInit(N, &LUstruct);
//
//        PStatInit(&stat);  // ���
//
//        //----- Solve -----
//        // pdgssvx: �л� NR_loc �Է��� �޾� LU ���� + Ǯ��
//        pdgssvx(&options,
//            &A,
//            &ScalePermstruct,
//            B_vector.data(), m_loc, 1, &grid,
//            &LUstruct, &SOLVEstruct,
//            berr.data(),
//            &stat, &info);
//
//        if (info != 0) {
//            // info > 0: singular pivot ��, info < 0: �߸��� ����
//            superlu_gridexit(&grid);
//            throw std::runtime_error("pdgssvx failed, info = " + std::to_string(info));
//        }
//
//
//        //----- �ڿ� ���� -----
//        // (A, B, X �� ���� ���迭�� ����� ����/�����̹Ƿ� �ı��ڿ��� �������� ����)
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
//    // ---- �ɼ� ----
//    int_t nprow = 1, npcol = 1;        // SuperLU 2D grid (�Է��� 1D ���� �� ���)
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
//    // ---- SuperLU ���μ��� �׸��� ----
//    gridinfo_t grid;
//    superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);
//    const int iam = grid.iam;
//    const int nprocs = grid.nprow * grid.npcol;
//    if (iam >= nprocs) { superlu_gridexit(&grid); MPI_Finalize(); return 0; }
//
//    const int_t m_glob = nx * ny;
//    const int_t n_glob = m_glob;
//
//    // ---- ���� ���� �� ��� ----
//    int_t m_loc = 0, fst_row = 0;
//    block_partition(m_glob, nprocs, iam, m_loc, fst_row);
//
//    // ---- ���� Eigen CSR(RowMajor) ���� ----
//    using StorageIndex = int_t; // SuperLU�� int_t�� �� ��ġ
//    static_assert(std::is_same<StorageIndex, int_t>::value, "Index type mismatch.");
//
//    using SpMatRM = Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>;
//    SpMatRM A_loc = build_local_5pt_laplacian_rowmajor<StorageIndex>(nx, ny, shift, m_loc, fst_row);
//
//    // Eigen RowMajor(CSR)�� ���� ������:
//    //  - valuePtr()        : nzval (nnz_loc)
//    //  - innerIndexPtr()   : colind (nnz_loc)  �� �۷ι� �� �ε���
//    //  - outerIndexPtr()   : rowptr (m_loc+1)  �� ���� �� ���� ������(0����)
//    const int_t nnz_loc = static_cast<int_t>(A_loc.nonZeros());
//    double* nzval = const_cast<double*>(A_loc.valuePtr());
//    int_t* colind = const_cast<int_t*>(reinterpret_cast<const int_t*>(A_loc.innerIndexPtr()));
//    int_t* rowptr = const_cast<int_t*>(reinterpret_cast<const int_t*>(A_loc.outerIndexPtr()));
//
//    // ---- SuperLU �л� �Է�(CompRowLoc) ���� ----
//    SuperMatrix A;
//    dCreate_CompRowLoc_Matrix_dist(&A,
//        m_glob, n_glob, nnz_loc, m_loc, fst_row,
//        nzval, colind, rowptr,
//        SLU_NR_loc, SLU_D, SLU_GE);
//
//    // ---- RHS: x_true=1 �� b=shift (�� ���� �� ����) ----
//    const int ldb = (int)m_loc;
//    std::vector<double> B((size_t)m_loc * nrhs, shift);
//    std::vector<double> berr((size_t)nrhs, 0.0);
//
//    // ---- �ɼ�/����ü ----
//    superlu_dist_options_t options; set_default_options_dist(&options);
//    dScalePermstruct_t ScalePermstruct; dLUstruct_t LUstruct;
//    dScalePermstructInit(m_glob, n_glob, &ScalePermstruct);
//    dLUstructInit(n_glob, &LUstruct);
//    SuperLUStat_t stat; PStatInit(&stat);
//
//    int info = 0;
//
//    // ---- ���� ����: SOLVEstruct_t ���� ó�� ----
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
//    // ---- �л� ��-norm ��� ����: ||X-1||_�� ----
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
//    // ---- ���� ----
//    PStatFree(&stat);
//#if defined(SUPERLU_DIST_MAJOR_VERSION) && (SUPERLU_DIST_MAJOR_VERSION >= 6)
//    dSolveFinalize(&options, &SOLVEstruct);
//#endif
//    dDestroy_LU(n_glob, &grid, &LUstruct);
//    dScalePermstructFree(&ScalePermstruct);
//    dLUstructFree(&LUstruct);
//
//    // �Է� ���۸� �ı�(Eigen�� ���� ����)
//    Destroy_SuperMatrix_Store_dist(&A);
//
//    superlu_gridexit(&grid);
//    MPI_Finalize();
//    return (info == 0) ? 0 : 1;


    MPI_Finalize();
    return 0;
}

