

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>  // std::pair
#include <Eigen/Dense>


Eigen::Vector4d robustLeastSquares(
    const std::vector<Eigen::Vector3d>& xyz_ngbs,
    const std::vector<double>& data_ngbs,
    const Eigen::Vector3d& xyz_own,
    int maxIter = 5,
    double huber_k = 1.5,
    double beta = 1.e-3,
    double tol = 1.e-6
) {
    int N = static_cast<int>(xyz_ngbs.size());
    Eigen::MatrixXd D(4, N);
    Eigen::VectorXd y(N), w(N);

    // 1) 설계 행렬 D와 종속 변수 y 초기화
    //    동시에 이웃까지의 최대 거리로 초기 스케일 h 계산
    double maxDist = 0.0;
    std::vector<double> dist(N);
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d diff = xyz_ngbs[i] - xyz_own;
        dist[i] = diff.norm();
        maxDist = std::max(maxDist, dist[i]);

        D(0, i) = 1.0;
        D(1, i) = diff[0];
        D(2, i) = diff[1];
        D(3, i) = diff[2];
        y(i) = data_ngbs[i];
    }

    // 2) 거리 기반 초기 가중치
    double h = (maxDist > 0.0 ? maxDist : 1.0);
    for (int i = 0; i < N; ++i) {
        w(i) = std::exp(-std::pow(dist[i] / h, 2));
    }



    //// 2) D_tv (3N×4) 밀집 차분 행렬 구성
    ////    g = [φ0, φx, φy, φz] → D_tv * g = [φx, φy, φz] concatenated
    //Eigen::MatrixXd D_tv(3 * N, 4);
    //D_tv.setZero();
    //for (int i = 0; i < N; ++i) {
    //    D_tv(i, 1) = 1.0;  // ∂x φ
    //    D_tv(N + i, 2) = 1.0;  // ∂y φ
    //    D_tv(2 * N + i, 3) = 1.0; // ∂z φ
    //}
    //// 3) DtD = D_tv^T * D_tv  (4×4 정규화용 행렬)
    //Eigen::Matrix4d DtD = D_tv.transpose() * D_tv;


    //// 4) Split-Bregman 변수 초기화
    //Eigen::Vector4d x = Eigen::Vector4d::Zero();        // 해
    //Eigen::VectorXd d = Eigen::VectorXd::Zero(3 * N);     // auxiliary (∇x 근사)
    //Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * N);     // Bregman 변수

    //// 5) AtA, Aᵀy 초기 계산 (가중치 포함)
    //Eigen::Matrix4d AtA = D * w.asDiagonal() * D.transpose();
    //Eigen::Vector4d Aty = D * w.asDiagonal() * y;

    //double mu = 1.e-2;

    //// 6) 메인 반복
    //for (int iter = 0; iter < 100; ++iter) {
    //    // (1) x-update: solve (AtA + μ·DtD) x = Aᵀy + μ·D_tvᵀ (d - b)
    //    Eigen::Matrix4d LHS = AtA + mu * DtD;
    //    Eigen::Vector4d RHS = Aty + mu * D_tv.transpose() * (d - b);
    //    x = LHS.ldlt().solve(RHS);

    //    // (2) d-update: soft-thresholding (shrinkage)
    //    Eigen::VectorXd Dx = D_tv * x + b;
    //    double thresh = beta / mu;
    //    for (int i = 0; i < 3 * N; ++i) {
    //        double mag = std::abs(Dx[i]);
    //        d[i] = (mag > thresh) ? (mag - thresh) * (Dx[i] / mag) : 0.0;
    //    }

    //    // (3) b-update: Bregman 변수 갱신
    //    b += (D_tv * x - d);

    //    // (4) Huber-IRLS 가중치 업데이트
    //    Eigen::VectorXd r = D.transpose() * x - y;
    //    if (r.squaredNorm() < tol) break;

    //    for (int i = 0; i < N; ++i) {
    //        double abs_ri = std::abs(r[i]);
    //        double hub_w = (abs_ri <= huber_k) ? 1.0 : (huber_k / abs_ri);
    //        //w[i] = win[i] * hub_w;
    //        w[i] = hub_w;
    //    }
    //    // 가중치 변경 시 AtA, Aty 재계산
    //    AtA = D * w.asDiagonal() * D.transpose();
    //    Aty = D * w.asDiagonal() * y;
    //}
    //return x;





    Eigen::Vector4d g = Eigen::Vector4d::Zero();

    // IRLS 반복
    beta = 1.e2;
    Eigen::Matrix4d Iden = Eigen::Matrix4d::Identity();
    Iden(0, 0) = 0.0;

    for (int iter = 0; iter < 5; ++iter) {
        // 3) 가중 정규 방정식 구성: (Dᵀ W D + λI) g = Dᵀ W y
        Eigen::MatrixXd W = w.asDiagonal();
        Eigen::Matrix4d ATA = D * W * D.transpose();
        double lambda = beta * ATA.trace() / 3.0;
        Eigen::Matrix4d A_reg = ATA + lambda * Iden;
        //Eigen::Matrix4d A_reg = D * W * D.transpose();
        Eigen::Vector4d b_reg = D * W * y;


        // --- 행(row) 스케일링 ---------------------------------------------------
        Eigen::VectorXd rowNorm = A_reg.rowwise().lpNorm<1>();                  // L1 노름
        Eigen::VectorXd rowScale = rowNorm.cwiseMax(1e-12).cwiseInverse();     // 0 방지
        A_reg = rowScale.asDiagonal() * A_reg;                                 // Dr * A
        b_reg = rowScale.asDiagonal() * b_reg;                                 // Dr * b

        // --- 열(column) 스케일링 ----------------------------------------------
        Eigen::VectorXd colNorm = A_reg.colwise().lpNorm<1>();
        Eigen::VectorXd colScale = colNorm.cwiseMax(1e-12).cwiseInverse();
        A_reg = A_reg * colScale.asDiagonal();                                 // A * Dc
        // (b_reg는 열 스케일링에 영향받지 않음)

        // 4) 균등화된 선형계 풀기
        Eigen::Vector4d g_hat = A_reg.fullPivLu().solve(b_reg);

        // 5) 해역 복원(역-열 스케일링)
        g = colScale.asDiagonal() * g_hat;


        // 5) 잔차 계산 및 Huber 가중치 갱신
        Eigen::VectorXd r = D.transpose() * g - y;

        if (r.squaredNorm() < tol) break;

        for (int i = 0; i < N; ++i) {
            double abs_ri = std::abs(r(i));
            w(i) = (abs_ri <= huber_k)
                ? 1.0
                : (huber_k / abs_ri);
        }
    }

    return g;
}


int main() {


    // 1) 파일 열기
    std::ifstream in("input.txt");
    if (!in.is_open()) {
        std::cerr << "파일 열기 실패\n";
        return 1;
    }

    // 2) 데이터를 보관할 컨테이너
    std::vector<std::pair<int, double>> data;
    data.reserve(360);  // (선택) 예상 줄 수가 360개 정도라면 미리 예약

    // 3) 파일 끝까지 읽기
    int idx;
    double value;
    while (in >> idx >> value) {
        data.emplace_back(idx, value);
    }

    Eigen::Vector3d xyz_own = { data[0].second, data[1].second, data[2].second };
    std::vector<Eigen::Vector3d> xyz_ngbs(data.size() / 4 - 1);
    std::vector<double> data_ngbs(data.size() / 4 - 1);
    for (int i = 1; i < data.size() / 4; ++i) {
        xyz_ngbs[i-1] = { data[i * 4 + 0].second, data[i * 4 + 1].second, data[i * 4 + 2].second };
        data_ngbs[i-1] = data[i * 4 + 3].second;
    }

    //xyz_ngbs.resize(2);
    //data_ngbs.resize(2);

    Eigen::Vector4d coef = robustLeastSquares(xyz_ngbs, data_ngbs, xyz_own);
    // coef = [a0, a1, a2, a3]로, 모델 y ≈ a0 + a1 x + a2 y + a3 z

    std::cout << "Coefficients: " << coef.transpose() << std::endl;


    //Eigen::VectorXd Aflatten(10), Bvec(4);
    //Aflatten.setZero();
    //Bvec.setZero();
    //double data_avg = 0.0;
    //for (int i = 0; i < xyz_ngbs.size(); ++i) {
    //    //for (int i = 0; i < 4; ++i) {
    //    const auto& xyz_ngb = xyz_ngbs[i];
    //    const auto& ngb_value = data_ngbs[i];

    //    Eigen::Vector3d diff_xyz = xyz_ngb - xyz_own;

    //    double weight = 1.0;

    //    Aflatten.coeffRef(0) += weight;
    //    Aflatten.coeffRef(1) += weight * diff_xyz[0];
    //    Aflatten.coeffRef(2) += weight * diff_xyz[1];
    //    Aflatten.coeffRef(3) += weight * diff_xyz[2];
    //    Aflatten.coeffRef(4) += weight * diff_xyz[0] * diff_xyz[0];
    //    Aflatten.coeffRef(5) += weight * diff_xyz[0] * diff_xyz[1];
    //    Aflatten.coeffRef(6) += weight * diff_xyz[0] * diff_xyz[2];
    //    Aflatten.coeffRef(7) += weight * diff_xyz[1] * diff_xyz[1];
    //    Aflatten.coeffRef(8) += weight * diff_xyz[1] * diff_xyz[2];
    //    Aflatten.coeffRef(9) += weight * diff_xyz[2] * diff_xyz[2];

    //    Bvec.coeffRef(0) += weight * ngb_value;
    //    Bvec.coeffRef(1) += weight * ngb_value * diff_xyz[0];
    //    Bvec.coeffRef(2) += weight * ngb_value * diff_xyz[1];
    //    Bvec.coeffRef(3) += weight * ngb_value * diff_xyz[2];


    //    data_avg += ngb_value;

    //    //std::cout << diff_xyz.norm() << " " << ngb_value << std::endl;
    //}
    //data_avg /= (double)xyz_ngbs.size();




    //Eigen::Matrix4d Amat;
    //Amat << Aflatten[0], Aflatten[1], Aflatten[2], Aflatten[3],
    //    Aflatten[1], Aflatten[4], Aflatten[5], Aflatten[6],
    //    Aflatten[2], Aflatten[5], Aflatten[7], Aflatten[8],
    //    Aflatten[3], Aflatten[6], Aflatten[8], Aflatten[9];


    ////Eigen::VectorXd result = Amat.inverse()* Bvec;

    ////###############
    //// 행렬 균등화 (Row/Column Equilibration) 방법
    //// --- 1. 행(row)-스케일링 -------------------------------------------------
    //Eigen::VectorXd rowNorm = Amat.rowwise().template lpNorm<1>();   // L1-norm 사용
    //Eigen::VectorXd rowScale = rowNorm.cwiseMax(1.0e-12).cwiseInverse(); // 0 방지
    //Eigen::MatrixXd A_r = Amat.array().colwise() * rowScale.array();
    //Eigen::VectorXd b_r = Bvec.array() * rowScale.array();

    //// --- 2. 열(column)-스케일링 --------------------------------------------
    //Eigen::VectorXd colNorm = A_r.colwise().template lpNorm<1>();
    //Eigen::VectorXd colScale = colNorm.cwiseMax(1.0e-12).cwiseInverse(); // 0 방지
    //Eigen::MatrixXd A_rc = A_r.array().rowwise() * colScale.transpose().array();

    //// --- 3. SVD 해 계산 ------------------------------------------------------
    ////Eigen::VectorXd y = A_rc.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_r);    // ※ y = Dc⁻¹ x  (스케일 공간 해)
    ////Eigen::VectorXd y = A_rc.fullPivLu().solve(b_r);


    //// --- 2. IRLS 루프 ---------------------------------------------------------
    //Eigen::VectorXd w = Eigen::VectorXd::Ones(4);
    //Eigen::VectorXd g = Eigen::VectorXd::Zero(4);

    //// IRLS 파라미터 (정규분포 σ 추정 없이 표준화되지 않은 잔차라면
    //// 실험적으로 k=1.5, c=4.685 정도가 무난)
    //const double k = 1.5;
    //const double c = 4.685;
    //const double beta = 1.e-3;

    //for (int it = 0; it < 5; ++it) {
    //    // 2-1  가중행렬 W½
    //    Eigen::MatrixXd Aw = w.asDiagonal() * A_rc;
    //    Eigen::VectorXd bw = w.asDiagonal() * b_r;

    //    // 2-2  Tikhonov 정칙화 λI 추가 ---------------------★
    //    Eigen::MatrixXd ATA = Aw.transpose() * Aw;
    //    double lambda = beta * ATA.trace() / 4.0;       // β·tr(AᵀA)/n
    //    Eigen::MatrixXd RHS = ATA + lambda * Eigen::Matrix4d::Identity();
    //    Eigen::VectorXd rhs = Aw.transpose() * bw;

    //    // 2-3  해 g 계산  (4×4 이므로 LDLᵀ가 가장 빠름)
    //    g = RHS.ldlt().solve(rhs);

    //    // 2-4  잔차 r 계산
    //    Eigen::VectorXd r = (A_rc * g - b_r);

    //    // 2-5  새 가중치 w 계산 ---------------------------★
    //    for (int i = 0; i < 4; ++i) {
    //        w(i) = (std::abs(r(i)) <= k) ? 1.0 : k / std::abs(r(i));
    //        //w(i) = useTukey ? tukeyW(r(i), c)
    //        //    : huberW(r(i), k);
    //    }
    //}




    //// --- 4. 역-스케일링(원래 변수 복원) ---------------------------------------
    ////Eigen::VectorXd result = colScale.array() * y.array();
    //Eigen::VectorXd result = colScale.array() * g.array();
    ////###############



    //std::cout << data_avg << std::endl;

    //std::cout << result << std::endl;





	return 0;
}