
#include <iostream>
#include <numbers>



#include "./src/ipc/distance/edge_edge.hpp"
#include "./src/ipc/distance/edge_edge_mollifier.hpp"
#include "./src/ipc/distance/point_triangle.hpp"

#include "./src/ipc_rigid/utils/eigen_ext.hpp"
#include "./src/ipc_rigid/physics/pose.hpp"
#include "./src/ipc_rigid/solvers/newton_solver.hpp"
#include "./src/ipc_rigid/utils/not_implemented_error.hpp"
#include "./src/ipc_rigid/physics/mass.hpp"

#include <boost/numeric/interval.hpp>
#include <fenv.h>
#include <iomanip>

// clang-format off
typedef boost::numeric::interval<
    double,
    boost::numeric::interval_lib::policies<
    boost::numeric::interval_lib::save_state<
    boost::numeric::interval_lib::rounded_transc_std<double> >,
    boost::numeric::interval_lib::checking_base<double> > >
    Interval;
// clang-format on

void show_fe_current_rounding_method(void)
{
    switch (fegetround()) {
    case FE_TONEAREST:
        std::cout << "FE_TONEAREST" << std::endl;
        break;
    case FE_DOWNWARD:
        std::cout << "FE_DOWNWARD" << std::endl;
        break;
    case FE_UPWARD:
        std::cout << "FE_UPWARD" << std::endl;
        break;
    case FE_TOWARDZERO:
        std::cout << "FE_TOWARDZERO" << std::endl;
        break;
    default:
        std::cout << "unknown" << std::endl;
    };
}



#include "./src/ipc_rigid/finite-diff/finitediff.hpp"
#include "./src/ipc_rigid/autodiff/autodiff_types.hpp"
#include "./src/ipc_rigid/barrier/barrier.hpp"
#include "./src/ipc_rigid/solvers/homotopy_solver.hpp"

#include "./src/ipc_rigid/physics/rigid_body.hpp"


using namespace ipc;
using namespace ipc::rigid;



int main(int argc, char** argv) {

    {
        using namespace ipc::rigid;
        int dim = 3;
        int num_bodies = 1000;
        Eigen::VectorXd dofs =
            Eigen::VectorXd::Random(num_bodies * Pose<double>::dim_to_ndof(dim));
        Poses<double> poses = Pose<double>::dofs_to_poses(dofs, dim);
        Eigen::VectorXd returned_dofs = Pose<double>::poses_to_dofs(poses);

        //CHECK((dofs - returned_dofs).squaredNorm() == Approx(0));
    }

    return 0;


    {
        int dim = 2;
        int num_vertices = 6 * 100;

        // Test vertices positions for given rb position
        Eigen::MatrixXd vertices = Eigen::MatrixXd::Random(num_vertices, dim);

        Eigen::MatrixXi edges =
            Eigen::VectorXd::LinSpaced(num_vertices, 0, num_vertices - 1)
            .cast<int>();
        edges = Eigen::MatrixXi(
            Eigen::Map<Eigen::MatrixXi>(edges.data(), num_vertices / dim, dim));

        double total_mass1;
        VectorMax3d center_of_mass1;
        MatrixMax3d moment_of_inertia1;
        compute_mass_properties(
            vertices, edges, total_mass1, center_of_mass1, moment_of_inertia1);

        double total_mass2 = compute_total_mass(vertices, edges);
        VectorMax3d center_of_mass2 = compute_center_of_mass(vertices, edges);
        MatrixMax3d moment_of_inertia2 = compute_moment_of_inertia(vertices, edges);

        std::cout << total_mass1 << " " << total_mass2 << std::endl;
        std::cout << (center_of_mass1 - center_of_mass2).norm() << std::endl;
        std::cout << (moment_of_inertia1 - moment_of_inertia2).norm() << std::endl;
        std::cout << center_of_mass1 << std::endl;
        std::cout << moment_of_inertia1 << std::endl;


    }



    return 0;

    {

        class AdHocProblem : public virtual BarrierProblem {
        public:
            AdHocProblem(int num_vars, double epsilon)
                : num_vars_(num_vars)
                , m_barrier_epsilon(epsilon)
            {
            }

            double compute_energy_term(
                const Eigen::VectorXd& x,
                Eigen::VectorXd& grad_Ex,
                Eigen::SparseMatrix<double>& hess_Ex,
                bool compute_grad = true,
                bool compute_hess = true) override
            {
                if (compute_grad) {
                    grad_Ex = x;
                }
                if (compute_hess) {
                    hess_Ex =
                        Eigen::MatrixXd::Identity(x.rows(), x.rows()).sparseView();
                }
                return x.squaredNorm() / 2.0;
            }

            double compute_barrier_term(
                const Eigen::VectorXd& x,
                Eigen::VectorXd& grad_Bx,
                Eigen::SparseMatrix<double>& hess_Bx,
                int& num_constraints,
                bool compute_grad = true,
                bool compute_hess = true) override
            {
                num_constraints = 1;

                // 1/2||x||² ≥ 1 → 1/2||x||² - 1 ≥ 0
                typedef AutodiffType<Eigen::Dynamic> Diff;
                Diff::activate(x.size());

                Eigen::Matrix<Diff::DDouble2, Eigen::Dynamic, 1> dx;

                Diff::DDouble2 b =
                    poly_log_barrier(dx.squaredNorm() / 2.0 - 1, m_barrier_epsilon);

                grad_Bx = b.getGradient();
                hess_Bx = b.getHessian().sparseView();

                return b.getValue();
            }

            double barrier_hessian(double x) const override
            {
                return poly_log_barrier_hessian(x, m_barrier_epsilon);
            }

            double barrier_activation_distance() const override
            {
                return m_barrier_epsilon;
            }
            void barrier_activation_distance(const double eps) override
            {
                m_barrier_epsilon = eps;
            }

            double barrier_stiffness() const override
            {
                return m_barrier_stiffness;
            }
            void barrier_stiffness(const double kappa) override
            {
                m_barrier_stiffness = kappa;
            }

            bool
                has_collisions(const Eigen::VectorXd&, const Eigen::VectorXd&) override
            {
                return false;
            }
            virtual double compute_earliest_toi(
                const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) override
            {
                return std::numeric_limits<double>::infinity();
            }
            virtual bool is_ccd_aligned_with_newton_update() override
            {
                return true;
            }

            int num_vars() const override { return num_vars_; }
            const VectorXb& is_dof_fixed() const override { return is_dof_fixed_; }
            const Eigen::VectorXd& starting_point() const { return x0; }

            double compute_min_distance(const Eigen::VectorXd& x) const override
            {
                return -1;
            }

            /// Get the world coordinates of the vertices
            Eigen::MatrixXd world_vertices(const Eigen::VectorXd& x) const override
            {
                throw NotImplementedError("no vertices");
            }

            /// Get the length of the diagonal of the worlds bounding box
            double world_bbox_diagonal() const override
            {
                throw NotImplementedError("no world bbox diagonal");
            }

            DiagonalMatrixXd mass_matrix() const override
            {
                DiagonalMatrixXd I(num_vars_);
                I.setIdentity();
                return I;
            }
            double average_mass() const override { return 1; }

            double timestep() const override { return 1; }

            int num_vars_;
            double m_barrier_epsilon;
            double m_barrier_stiffness;
            Eigen::VectorXd x0;
            VectorXb is_dof_fixed_;
        };


        int num_vars = 10;
        double epsilon = 1.0; //GENERATE(1.0, 0.5, 1e-1, 5e-2);
        AdHocProblem problem(num_vars, epsilon);

        Eigen::VectorXd x(num_vars);
        x.setConstant(1.0* 1.0);
       // x.setConstant(GENERATE(-1.0, 1.0) * GENERATE(1e-1, 1.0, 2.0 - 1e-3, 4.0));

        Eigen::VectorXd grad_fx;
        Eigen::SparseMatrix<double> hess_fx;
        double fx = problem.compute_objective(x, grad_fx, hess_fx);
        // If the function evaluates to infinity then the finite differences will
        // not work. I assume in the definition of the barrier gradient that
        // d/dx ∞ = 0.
        if (!std::isinf(fx)) {
            // Use a higher order finite difference method because the function near
            // the boundary becomes very non-linear. This problem worsens as the ϵ
            // of the boundary gets smaller.

            // Test ∇f
            Eigen::VectorXd finite_grad(problem.num_vars());
            finite_grad = ipc::rigid::eval_grad_objective_approx(problem, x);
            
            std::cout << fd::compare_gradient(finite_grad, grad_fx) << std::endl;
           // CHECK(fd::compare_gradient(finite_grad, grad_fx));

            // Test ∇²f
            Eigen::MatrixXd finite_hessian = eval_hess_objective_approx(problem, x);
            std::cout << fd::compare_jacobian(finite_hessian, Eigen::MatrixXd(hess_fx)) << std::endl;
           // CHECK(fd::compare_jacobian(finite_hessian, Eigen::MatrixXd(hess_fx)));
        }


    }

    return 0;

    {
        // Setup problem
        // -----------------------------------------------------------------
        class AdHocProblem : public virtual ipc::rigid::OptimizationProblem {
        public:
            AdHocProblem(int num_vars)
            {
                num_vars_ = num_vars;
                x0.resize(num_vars);
                x0.setRandom();
                is_dof_fixed_ = ipc::VectorXb::Zero(num_vars);
            }

            double compute_objective(
                const Eigen::VectorXd& x,
                Eigen::VectorXd& grad_fx,
                Eigen::SparseMatrix<double>& hess_fx,
                bool compute_grad = true,
                bool compute_hess = true) override
            {
                if (compute_grad) {
                    grad_fx = x;
                }
                if (compute_hess) {
                    hess_fx =
                        Eigen::MatrixXd::Identity(x.rows(), x.rows()).sparseView();
                }
                return x.squaredNorm() / 2.0;
            }

            bool
                has_collisions(const Eigen::VectorXd&, const Eigen::VectorXd&) override
            {
                return false;
            }
            double compute_earliest_toi(
                const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) override
            {
                return std::numeric_limits<double>::infinity();
            }
            bool is_ccd_aligned_with_newton_update() override { return true; }

            const Eigen::VectorXd& starting_point() const { return x0; }
            int num_vars() const override { return num_vars_; }
            const ipc::VectorXb& is_dof_fixed() const override { return is_dof_fixed_; }

            double compute_min_distance(const Eigen::VectorXd& x) const override
            {
                return -1;
            }

            /// Get the world coordinates of the vertices
            Eigen::MatrixXd world_vertices(const Eigen::VectorXd& x) const override
            {
                throw ipc::rigid::NotImplementedError("no vertices");
            }

            /// Get the length of the diagonal of the worlds bounding box
            double world_bbox_diagonal() const override
            {
                throw ipc::rigid::NotImplementedError("no world bbox diagonal");
            }

            ipc::DiagonalMatrixXd mass_matrix() const override
            {
                ipc::DiagonalMatrixXd I(num_vars_);
                I.setIdentity();
                return I;
            }
            double average_mass() const override { return 1; }

            double timestep() const override { return 1; }

            int num_vars_;
            ipc::VectorXb is_dof_fixed_;
            Eigen::VectorXd x0;
        };


        int num_vars = 10;

        AdHocProblem problem(num_vars);


        ipc::rigid::NewtonSolver solver;
        solver.set_problem(problem);
        solver.init_solve(problem.starting_point());
        ipc::rigid::OptimizationResults results = solver.solve(problem.starting_point());
        std::cout << (results.success) << std::endl;
        std::cout << results.x.squaredNorm() << std::endl;
        std::cout << results.minf << std::endl;
        //CHECK(results.x.squaredNorm() == Approx(0).margin(1e-6));
        //CHECK(results.minf == Approx(0).margin(1e-6));


    }



    {
        int num_vars = 1000;
        Eigen::VectorXd x(num_vars);
        x.setRandom();
        // f = x^2
        Eigen::VectorXd gradient = 2 * x;
        Eigen::SparseMatrix<double> hessian =
            ipc::SparseDiagonal<double>(2 * Eigen::VectorXd::Ones(num_vars));
        Eigen::VectorXd delta_x;
        ipc::rigid::NewtonSolver solver;
        solver.compute_direction(gradient, hessian, delta_x);
        std::cout << (x + delta_x).squaredNorm() << std::endl;
        //CHECK((x + delta_x).squaredNorm() == Approx(0.0));
    }

    //{
    //    Eigen::SparseMatrix<double> A =
    //        Eigen::MatrixXd::Random(100, 100).sparseView();
    //    double mu = ipc::rigid::make_matrix_positive_definite(A);
    //    //CAPTURE(mu);
    //    auto eig_vals = Eigen::MatrixXd(A).eigenvalues();
    //    for (int i = 0; i < eig_vals.size(); i++) {
    //        std::cout << eig_vals(i).real() << std::endl;
    //        //CHECK(eig_vals(i).real() >= Approx(0.0).margin(1e-12));
    //    }
    //}


    //{
    //    Interval theta = Interval(0.79358805865013693);
    //    Interval output = cos(theta);
    //    std::cout << std::setprecision(
    //        std::numeric_limits<double>::digits10 + 1)
    //        << "[" << output.lower() << ", " << output.upper() << "]"
    //        << std::endl;
    //    std::cout << "is empty: "
    //        << (output.lower() > output.upper() ? "true" : "false")
    //        << std::endl;
    //}

    //{
    //    double theta = 0.79358805865013693;
    //    show_fe_current_rounding_method();
    //    volatile double output1 = cos(theta);
    //    std::cout << std::setprecision(
    //        std::numeric_limits<double>::digits10 + 1)
    //        << output1 << std::endl;
    //    fesetround(FE_DOWNWARD);
    //    show_fe_current_rounding_method();
    //    volatile double output2 = cos(theta);
    //    std::cout << std::setprecision(
    //        std::numeric_limits<double>::digits10 + 1)
    //        << output2 << std::endl;
    //    fesetround(FE_UPWARD);
    //    show_fe_current_rounding_method();
    //    volatile double output3 = cos(theta);
    //    std::cout << std::setprecision(
    //        std::numeric_limits<double>::digits10 + 1)
    //        << output3 << std::endl;
    //    fesetround(FE_TOWARDZERO);
    //    show_fe_current_rounding_method();
    //    volatile double output4 = cos(theta);
    //    std::cout << std::setprecision(
    //        std::numeric_limits<double>::digits10 + 1)
    //        << output4 << std::endl;
    //}
    //return 0;



}

*/