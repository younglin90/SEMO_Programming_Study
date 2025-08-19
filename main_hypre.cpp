#include <HYPRE.h>
#include <HYPRE_utilities.h>
#include <HYPRE_krylov.h>
#include <HYPRE_parcsr_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_IJ_mv.h>
#include <mpi.h>

// Eigen
#include <Eigen/Sparse>
#include <Eigen/Dense> 

#include <iostream>

int main(int argc, char* argv[]) {
    // 0) MPI ����
    MPI_Init(&argc, &argv);
    int num_procs = 1, my_id = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    // 1) hypre �ʱ�ȭ
    HYPRE_Init();

    // 2) ���� ũ�� �� ���� ���� ���
    const HYPRE_BigInt global_n = 10;
    HYPRE_BigInt base = global_n / num_procs;
    HYPRE_BigInt rem = global_n % num_procs;

    HYPRE_BigInt ilower, iupper;
    if (my_id < rem) {
        ilower = my_id * (base + 1);
        iupper = ilower + (base + 1) - 1;
    }
    else {
        ilower = rem * (base + 1) + (my_id - rem) * base;
        iupper = ilower + base - 1;
    }
    const HYPRE_Int local_n = static_cast<HYPRE_Int>(iupper - ilower + 1);

    // 3) ����: 1D Poisson ��밢 ����� Eigen���� ����(����)
    Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_A(global_n, global_n);
    eigen_A.reserve(Eigen::VectorXi::Constant((int)global_n, 3));
    for (int k = 0; k < global_n; ++k) {
        if (k > 0)              eigen_A.insert(k, k - 1) = -1.0;
        eigen_A.insert(k, k) = 2.0;
        if (k < global_n - 1)   eigen_A.insert(k, k + 1) = -1.0;
    }
    eigen_A.makeCompressed();

    // 4) IJ Matrix ���� �� �� ����
    HYPRE_IJMatrix A_ij;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A_ij);
    HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A_ij);

    // ���� ������ ��(= ilower..iupper)�� ���ؼ��� �� ����
    std::vector<HYPRE_Int> ncols_row(1);
    std::vector<HYPRE_BigInt> cols;
    std::vector<double> vals;

    for (HYPRE_BigInt grow = ilower; grow <= iupper; ++grow) {
        int row = (int)grow;
        std::vector<HYPRE_BigInt> cols;
        std::vector<double> vals;

        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(eigen_A, row); it; ++it) {
            cols.push_back((HYPRE_BigInt)it.col());  // ���� ��¥�� '��'�� ����
            vals.push_back(it.value());
        }
        HYPRE_BigInt irow = grow;
        HYPRE_Int ncols = (HYPRE_Int)cols.size();

        int ierr = HYPRE_IJMatrixSetValues(
            A_ij,
            1,              // nrows: �� ���� 1�� ��
            &ncols,         // �� ���� nnz ���� �迭(���⼱ 1�� ���� ������)
            &irow,          // �� �ε��� �迭(���⼱ 1�� ���� ������)
            cols.data(),    // �� ���� �� �ε����� (HYPRE_BigInt[])
            vals.data()     // �� ���� ���� (double[])
        );
    }


    HYPRE_IJMatrixAssemble(A_ij);

    // ParCSR ��ü�� ������
    HYPRE_ParCSRMatrix A_par = nullptr;
    HYPRE_IJMatrixGetObject(A_ij, (void**)&A_par);

    // 5) RHS/�� ����: IJ Vector�� ���� �� �� ����
    HYPRE_IJVector b_ij, x_ij;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b_ij);
    HYPRE_IJVectorSetObjectType(b_ij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b_ij);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x_ij);
    HYPRE_IJVectorSetObjectType(x_ij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x_ij);

    std::vector<HYPRE_BigInt> rows(local_n);
    std::vector<double> bvals(local_n, 1.0), xvals(local_n, 0.0);
    for (HYPRE_Int i = 0; i < local_n; ++i) rows[i] = ilower + i;

    HYPRE_IJVectorSetValues(b_ij, local_n, rows.data(), bvals.data());
    HYPRE_IJVectorSetValues(x_ij, local_n, rows.data(), xvals.data());

    HYPRE_IJVectorAssemble(b_ij);
    HYPRE_IJVectorAssemble(x_ij);

    // ParCSR �����ͷ� ������
    HYPRE_ParVector b_par = nullptr, x_par = nullptr;
    HYPRE_IJVectorGetObject(b_ij, (void**)&b_par);
    HYPRE_IJVectorGetObject(x_ij, (void**)&x_par);

    // 6) PCG + BoomerAMG ����/�ع�
    HYPRE_Solver pcg = nullptr;
    HYPRE_Solver amg = nullptr;

    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg);
    HYPRE_PCGSetMaxIter(pcg, 100);
    HYPRE_PCGSetTol(pcg, 1e-6);
    HYPRE_PCGSetTwoNorm(pcg, 1);
    HYPRE_PCGSetPrintLevel(pcg, 2);
    HYPRE_PCGSetLogging(pcg, 1);

    HYPRE_BoomerAMGCreate(&amg);
    HYPRE_BoomerAMGSetPrintLevel(amg, 1);
    HYPRE_BoomerAMGSetCoarsenType(amg, 6);
    HYPRE_BoomerAMGSetRelaxType(amg, 3);
    HYPRE_BoomerAMGSetNumSweeps(amg, 1);
    HYPRE_BoomerAMGSetMaxLevels(amg, 20);

    HYPRE_PCGSetPrecond(pcg,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
        amg);

    HYPRE_ParCSRPCGSetup(pcg, A_par, b_par, x_par);
    HYPRE_ParCSRPCGSolve(pcg, A_par, b_par, x_par);

    // 7) �ظ� ���÷� ������ Eigen Vector�� ����
    std::vector<double> xloc(local_n, 0.0);
    HYPRE_IJVectorGetValues(x_ij, local_n, rows.data(), xloc.data());
    Eigen::Map<Eigen::VectorXd> x_local(xloc.data(), local_n);

    if (my_id == 0) {
        std::cout << "x(local, rank 0):\n" << x_local << std::endl;
    }

    // 8) ����
    HYPRE_BoomerAMGDestroy(amg);
    HYPRE_ParCSRPCGDestroy(pcg);

    HYPRE_IJMatrixDestroy(A_ij);
    HYPRE_IJVectorDestroy(b_ij);
    HYPRE_IJVectorDestroy(x_ij);

    HYPRE_Finalize();
    MPI_Finalize();
    return 0;
}