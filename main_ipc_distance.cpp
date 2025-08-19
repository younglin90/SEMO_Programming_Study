#include <iostream>
#include <format>


#include "src/ipc/distance/point_point.hpp"
#include "src/ipc/distance/point_triangle.hpp"
#include "src/ipc/distance/edge_edge.hpp"
#include "src/ipc/distance/edge_edge_mollifier.hpp"

using vec3 = Eigen::Vector3d;
using Point = Eigen::Vector3d;

struct Segment {
	vec3 a, b;
};
struct Plane {
	vec3 n;
	double d;
};
struct Triangle {
	vec3 a, b, c;
};

//✓ 점 -> 세그먼트           GPC pseg0.a73, RTCD 127
vec3 closest(const vec3& p, const Segment& seg) {
	vec3 ab = seg.b - seg.a;
	double t = (p - seg.a).dot(ab) / ab.squaredNorm();
	t = std::clamp(t, 0.0, 1.0);
	return seg.a + t * ab;
}

vec3 closest(const vec3& p, const Plane& plane) {
	double t = (p.dot(plane.n) - plane.d) / plane.n.squaredNorm();
	return p - t * plane.n;
}

bool is_inside(const Point& p, const Plane& plane) {
	double dot = p.dot(plane.n);
	return std::abs(dot - plane.d) <= 1.e-12;
}
// 점 in 삼각형           GPC pseg0.b24, RTCD 203
bool is_inside(
	const Point& p,
	const Triangle& tri
) {
	// 우선 점이 삼각형 평면 위에 있는지 확인
	Plane plane;
	plane.n = (tri.b - tri.a).cross(tri.c - tri.a).normalized();
	plane.d = plane.n.dot(tri.a);
	if (is_inside(p, plane) == false) return false;

	vec3 pa = tri.a - p;
	vec3 pb = tri.b - p;
	vec3 pc = tri.c - p;

	vec3 u = pb.cross(pc);
	vec3 v = pc.cross(pa);
	if (u.dot(v) < 0.0) return false;
	vec3 w = pa.cross(pb);
	if (u.dot(w) < 0.0) return false;
	return true;
}
//vec3 closest(
//	const Point& p,
//	const Triangle& tri
//) {
//	Plane plane;
//	plane.n = (tri.b - tri.a).cross(tri.c - tri.a).normalized();
//	plane.d = plane.n.dot(tri.a);
//	Point cloPt = closest(p, plane);
//	double magSq0 = (p - cloPt).squaredNorm();
//	if (is_inside(cloPt, tri)) return cloPt;
//
//    Point c1 = closest(p, Segment{ tri.a, tri.b });
//    Point c2 = closest(p, Segment{ tri.b, tri.c });
//    Point c3 = closest(p, Segment{ tri.c, tri.a });
//
//	double magSseg1.a = (p - c1).squaredNorm();
//	double magSseg1.b = (p - c2).squaredNorm();
//	double magSq3 = (p - c3).squaredNorm();
//
//	if (magSseg1.a < magSseg1.b && magSseg1.a < magSq3) {
//		return c1;
//	}
//	else if (magSseg1.b < magSseg1.a && magSseg1.b < magSq3) {
//		return c2;
//	}
//	return c3;
//}


double point_plane_distance(
    const Eigen::Ref<const Eigen::Vector3d>&p,
    const Eigen::Ref<const Eigen::Vector3d>&origin,
    const Eigen::Ref<const Eigen::Vector3d>&normal)
{
    const double point_to_plane = (p - origin).dot(normal);
    return point_to_plane * point_to_plane / normal.squaredNorm();
}

double point_plane_distance(
    const Eigen::Ref<const Eigen::Vector3d>&p,
    const Eigen::Ref<const Eigen::Vector3d>&t0,
    const Eigen::Ref<const Eigen::Vector3d>&t1,
    const Eigen::Ref<const Eigen::Vector3d>&t2)
{
    return point_plane_distance(p, t0, (t1 - t0).cross(t2 - t0));
}


vec3 closest(
	const Point& p,
	const Triangle& tri
) {
	Plane plane;
	plane.n = (tri.b - tri.a).cross(tri.c - tri.a).normalized();
	plane.d = plane.n.dot(tri.a);
	Point cloPt = closest(p, plane);
	double magSq0 = (p - cloPt).squaredNorm();
	if (is_inside(cloPt, tri)) return cloPt;

    Segment e0; e0.a = tri.a; e0.b = tri.b;
    Segment e1; e1.a = tri.b; e1.b = tri.c;
    Segment e2; e2.a = tri.c; e2.b = tri.a;

    Point c1 = closest(p, Segment{ tri.a, tri.b });
    Point c2 = closest(p, Segment{ tri.b, tri.c });
    Point c3 = closest(p, Segment{ tri.c, tri.a });

    double magSq1 = (p - c1).squaredNorm();
    double magSq2 = (p - c2).squaredNorm();
    double magSq3 = (p - c3).squaredNorm();

    //double magSseg1.a_t = (e0.a - p).head<3>().cross((e0.b - p).head<3>()).squaredNorm()
    //    / (e0.b - e0.a).squaredNorm();
    //double magSseg1.b_t = (e1.a - p).head<3>().cross((e1.b - p).head<3>()).squaredNorm()
    //    / (e1.b - e1.a).squaredNorm();
    //double magSq3_t = (e2.a - p).head<3>().cross((e2.b - p).head<3>()).squaredNorm()
    //    / (e2.b - e2.a).squaredNorm();

    //std::cout << c1.transpose() << std::endl;
    //std::cout << c2.transpose() << std::endl;
    //std::cout << c3.transpose() << std::endl;
    //std::cout << (p - c2).squaredNorm() << std::endl;
    //std::cout << std::format("{} {} {}", magSseg1.a, magSseg1.b, magSq3) << std::endl;
    //std::cout << std::format("{} {} {}", magSseg1.a_t, magSseg1.b_t, magSq3_t) << std::endl;

    Point result = c1;
    if (magSq2 < magSq1) result = c2;
    if (magSq3 < magSq2) result = c3;
    return result;
	//if (magSseg1.a < magSseg1.b && magSseg1.a < magSq3) {
	//	return c1;
	//}
	//else if (magSseg1.b < magSseg1.a && magSseg1.b < magSq3) {
	//	return c2;
	//}
	//return c3;
}

std::tuple<double, vec3, vec3> closest_distance(const Point& p, const Triangle& tri) {
	auto p_clo = closest(p, tri);
	double dist = (p - p_clo).norm();
	return std::make_tuple(dist, p, p_clo);
}

//
//std::tuple<double, vec3, vec3> closest_distance(const Segment& seg0, const Segment& seg1) {
//
//    // min ||(v11 +γ1(v12 −v11)) − (v21 +γ2(v22 −v21))||
//    // s.t. 0 ≤ γ1,γ2 ≤ 1.
//    //(ab0.dot(seg0.a) + gamma1 * ab0.squaredNorm()) - (ab0.dot(seg1.a) + gamma2 * ab0.dot(ab1));
//    //(ab1.dot(seg0.a) + gamma1 * ab1.dot(ab0)) - (ab1.dot(seg1.a) + gamma2 * ab1.squaredNorm());
//    vec3 e0 = (seg0.b - seg0.a);
//    vec3 e1 = (seg1.b - seg1.a);
//    Eigen::Matrix2d mat;
//    mat <<
//        e0.squaredNorm(), -e0.dot(e1),
//        e1.dot(e0), -e1.squaredNorm();
//    Eigen::Vector2d b;
//    b <<
//        -e0.dot(seg0.a) + e0.dot(seg1.a),
//        -e1.dot(seg0.a) + e1.dot(seg1.a);
//    Eigen::Vector2d gamma = mat.inverse() * b;
//
//    double dist = std::numeric_limits<double>::infinity();
//    vec3 v0, v1;
//
//    double mag0 = (seg0.a - seg1.a).squaredNorm();
//    double mag1 = (seg0.a - seg1.b).squaredNorm();
//    double mag2 = (seg0.b - seg1.a).squaredNorm();
//    double mag3 = (seg0.b - seg1.b).squaredNorm();
//
//    if (dist > mag0) {
//        v0 = seg0.a; v1 = seg1.a; dist = mag0;
//    }
//    if (dist > mag1) {
//        v0 = seg0.a; v1 = seg1.b; dist = mag1;
//    }
//    if (dist > mag2) {
//        v0 = seg0.b; v1 = seg1.a; dist = mag2;
//    }
//    if (dist > mag3) {
//        v0 = seg0.b; v1 = seg1.b; dist = mag3;
//    }
//    return std::make_tuple(std::sqrt(dist), v0, v1);
//
//
//
//
//
//    //if (std::abs(mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0)) < 1.e-12) {
//    //    // 두 segment 는 거의 평행함
//    //    if (double dist_tmp = (seg0.a - seg1.a).squaredNorm();
//    //        dist_tmp < dist) {
//    //        v0 = seg0.a; v1 = seg1.a;
//    //        dist = dist_tmp;
//    //    }
//    //    if (double dist_tmp = (seg0.a - seg1.b).squaredNorm();
//    //        dist_tmp < dist) {
//    //        v0 = seg0.a; v1 = seg1.b;
//    //        dist = dist_tmp;
//    //    }
//    //    if (double dist_tmp = (seg0.b - seg1.a).squaredNorm();
//    //        dist_tmp < dist) {
//    //        v0 = seg0.b; v1 = seg1.a;
//    //        dist = dist_tmp;
//    //    }
//    //    if (double dist_tmp = (seg0.b - seg1.b).squaredNorm();
//    //        dist_tmp < dist) {
//    //        v0 = seg0.b; v1 = seg1.b;
//    //        dist = dist_tmp;
//    //    }
//    //    return std::make_tuple(dist, v0, v1);
//
//    //}
//
//
//    //if (gamma[0] >= 0.0 && gamma[0] <= 1.0 && gamma[1] >= 0.0 && gamma[1] <= 1.0) {
//    //    v0 = seg0.a + gamma[0] * ab0;
//    //    v1 = seg1.a + gamma[1] * ab1;
//    //    dist = (v0 - v1).squaredNorm();
//    //}
//    //else {
//    //    if (gamma[0] >= 0.0 && gamma[0] <= 1.0) {
//    //        v0 = seg0.a + gamma[0] * ab0;
//    //        if (double dist_tmp = (v0 - seg1.a).squaredNorm();
//    //            dist_tmp < dist) {
//    //            v1 = seg1.a;
//    //            dist = dist_tmp;
//    //        }
//    //        if (double dist_tmp = (v0 - seg1.b).squaredNorm();
//    //            dist_tmp < dist) {
//    //            v1 = seg1.b;
//    //            dist = dist_tmp;
//    //        }
//    //    }
//    //    else if (gamma[1] >= 0.0 && gamma[1] <= 1.0) {
//    //        v1 = seg1.a + gamma[1] * ab1;
//    //        if (double dist_tmp = (v1 - seg0.a).squaredNorm();
//    //            dist_tmp < dist) {
//    //            v0 = seg0.a;
//    //            dist = dist_tmp;
//    //        }
//    //        if (double dist_tmp = (v1 - seg0.b).squaredNorm();
//    //            dist_tmp < dist) {
//    //            v0 = seg0.b;
//    //            dist = dist_tmp;
//    //        }
//    //    }
//    //    else {
//    //        if (double dist_tmp = (seg0.a - seg1.a).squaredNorm();
//    //            dist_tmp < dist) {
//    //            v0 = seg0.a; v1 = seg1.a;
//    //            dist = dist_tmp;
//    //        }
//    //        if (double dist_tmp = (seg0.a - seg1.b).squaredNorm();
//    //            dist_tmp < dist) {
//    //            v0 = seg0.a; v1 = seg1.b;
//    //            dist = dist_tmp;
//    //        }
//    //        if (double dist_tmp = (seg0.b - seg1.a).squaredNorm();
//    //            dist_tmp < dist) {
//    //            v0 = seg0.b; v1 = seg1.a;
//    //            dist = dist_tmp;
//    //        }
//    //        if (double dist_tmp = (seg0.b - seg1.b).squaredNorm();
//    //            dist_tmp < dist) {
//    //            v0 = seg0.b; v1 = seg1.b;
//    //            dist = dist_tmp;
//    //        }
//    //    }
//    //}
//
//    //return std::make_tuple(std::sqrt(dist), v0, v1);
//}



std::tuple<double, vec3, vec3> closest_distance(
    const Segment& seg0, const Segment& seg1) {
    // 방향 벡터 및 초기 설정
    vec3 u = seg0.b - seg0.a, v = seg1.b - seg1.a, w = seg0.a - seg1.a;
    double a = u.dot(u), b = u.dot(v), c = v.dot(v);
    double d = u.dot(w), e = v.dot(w);
    constexpr double epsilon = 1e-15;

    // 분모 D 계산 및 특이케이스 처리
    double D = a * c - b * b;
    double sc = 0, sN = 0, sD = D;
    double tc = 0, tN = 0, tD = D;
    bool sClamped = false, tClamped = false;  // s와 t에 대해 클램핑 발생 여부

    if (D < epsilon) { // 선분이 평행한 경우
        sN = 0.0;
        sD = 1.0;
        tN = e;
        tD = c;
    }
    else { // 일반적인 경우
        sN = b * e - c * d;
        tN = a * e - b * d;
        // s에 대한 클램핑
        if (sN < 0.0) {
            sN = 0.0;
            sClamped = true;
            tN = e;
            tD = c;
        }
        else if (sN > sD) {
            sN = sD;
            sClamped = true;
            tN = e + b;
            tD = c;
        }
    }

    // t에 대한 클램핑
    if (tN < 0) {
        tN = 0;
        tClamped = true;
        if (!sClamped) {
            sN = std::clamp(-d / a, 0.0, 1.0);
        }
    }
    else if (tN > tD) {
        tN = tD;
        tClamped = true;
        if (!sClamped) {
            sN = std::clamp((b - d) / a, 0.0, 1.0);
        }
    }

    // 파라미터 계산
    sc = (std::abs(sN) < epsilon) ? 0 : sN / sD;
    tc = (std::abs(tN) < epsilon) ? 0 : tN / tD;
    vec3 closestP = seg0.a + u * sc;
    vec3 closestQ = seg1.a + v * tc;
    double minDist = (closestP - closestQ).norm();

    // 내부 해가 클램핑 없이 발생했다면 최적해이므로 바로 반환
    if (!sClamped && !tClamped && (0 <= sc && sc <= 1 && 0 <= tc && tc <= 1)) {
        return std::make_tuple(minDist, closestP, closestQ);
    }

    // 그렇지 않다면, 엔드포인트 조합 검사
    minDist = std::numeric_limits<double>::infinity();
    auto checkEndpoints = [&](const vec3& p, const vec3& q) {
        double dist = (p - q).norm();
        if (dist < minDist) {
            minDist = dist;
            closestP = p;
            closestQ = q;
        }
        };
    auto closestPointOnSegment = [](const vec3& point,
        const vec3& segA,
        const vec3& segB) -> vec3 {
            vec3 dir = segB - segA;
            double t = (point - segA).dot(dir) / dir.dot(dir);
            t = std::clamp(t, 0.0, 1.0);
            return segA + dir * t;
        };

    checkEndpoints(seg0.a, closestPointOnSegment(seg0.a, seg1.a, seg1.b));
    checkEndpoints(seg0.b, closestPointOnSegment(seg0.b, seg1.a, seg1.b));
    checkEndpoints(closestPointOnSegment(seg1.a, seg0.a, seg0.b), seg1.a);
    checkEndpoints(closestPointOnSegment(seg1.b, seg0.a, seg0.b), seg1.b);

    checkEndpoints(seg0.a, seg1.a);
    checkEndpoints(seg0.a, seg1.b);
    checkEndpoints(seg0.b, seg1.a);
    checkEndpoints(seg0.b, seg1.b);

    return std::make_tuple(minDist, closestP, closestQ);
}



int main() {

	//Eigen::Vector3d a(0, 0, 0);
	//Eigen::Vector3d b(1, 1, 1);

	//auto dist = ipc::point_point_distance(a, b);

	//std::cout << dist << std::endl;

	//Eigen::Vector3d c(1, 0, 1);
	//Eigen::Vector3d d(1, 1, 0);

	//auto dist2 = ipc::point_triangle_distance(a, b, c, d);
	//std::cout << dist2 << std::endl;

	//auto ea0 = Eigen::Vector3d(-1, 0, 0);
	//auto ea1 = Eigen::Vector3d(1, 0, 0);
	//auto eb0 = Eigen::Vector3d(-1, 1e-9, 0);
	//auto eb1 = Eigen::Vector3d(1, -1e-9, 0);
	//auto dist3 = ipc::edge_edge_cross_squarednorm(ea0, ea1, eb0, eb1);
	//auto dist4 = ipc::edge_edge_distance(ea0, ea1, eb0, eb1);
	//auto thres = ipc::edge_edge_mollifier_threshold(ea0, ea1, eb0, eb1);
	//auto dist5 = ipc::edge_edge_mollifier(ea0, ea1, eb0, eb1, thres);
	//std::cout << dist3 << std::endl;
	//std::cout << dist4 << std::endl;
	//std::cout << thres << std::endl;
	//std::cout << dist5 << std::endl;


	//auto p = Eigen::Vector3d(0.229443, 0.100068, -0.432831);
	//Triangle tri;
	//tri.a = Eigen::Vector3d(0.744484, 0.571552, 0.063586);
	//tri.b = Eigen::Vector3d(0.622139, 0.362257, -0.001514);
	//tri.c = Eigen::Vector3d(0.601947, 0.421252, 0.217214);


    auto p = Eigen::Vector3d(0.477741, 0.0296258, 0.14452);
    Triangle tri;
    tri.a = Eigen::Vector3d(0.494398, -0.0224514, -0.0711794);
    tri.b = Eigen::Vector3d(0.886229, 0.569113, -0.094847);
    tri.c = Eigen::Vector3d(0.744484, 0.571552, 0.063586);



    //auto seg0_a = Eigen::Vector3d(0.477741, 0.0296258, 0.14452);
    //auto seg0_b = Eigen::Vector3d(0.494398, -0.0224514, -0.0711794);
    //auto seg1_a = Eigen::Vector3d(0.886229, 0.569113, -0.094847);
    //auto seg1_b = Eigen::Vector3d(0.744484, 0.571552, 0.063586);


    //auto p = Eigen::Vector3d(0.5, 0.5, 0);
    //Triangle tri;
    //tri.a = Eigen::Vector3d(0.744484, 0.571552, 1);
    //tri.b = Eigen::Vector3d(0.622139, 0.362257, -1);
    //tri.c = Eigen::Vector3d(0.601947, 0.421252, 1);


    std::cout << (p - tri.b).norm() << std::endl;

    std::cout << std::sqrt(ipc::point_triangle_distance(p, tri.a, tri.b, tri.c)) << std::endl;
    std::cout << ipc::point_triangle_distance_gradient(p, tri.a, tri.b, tri.c).transpose() << std::endl;
    std::cout << ipc::point_triangle_distance_hessian(p, tri.a, tri.b, tri.c) << std::endl;


	auto [dist, v_own, v_ngb] = closest_distance(p, tri);
    std::cout << (dist) << std::endl;
    std::cout << 2.0 * (v_own - v_ngb).transpose() << std::endl;


    Segment seg0, seg1;
    //seg0.a = p; seg0.b = tri.a;
    //seg1.a = tri.b; seg1.b = tri.c;

    //seg0.a = { 0.477741,  0.0296258, 0.14452 };
    //seg0.b = { 0.494398,  -0.0224514, -0.0711794 };
    //seg1.a = { 0.886229,  0.569113, -0.094847 };
    //seg1.b = { 0.744484,  0.571552, 0.063586 };

    //seg0.a = { 0.00643876,  0.486236, -0.11672 };
    //seg0.b = { 0.0364487,  0.270033, -0.419123 };
    //seg1.a = { 1.23894,  0.565477, -0.0580213 };
    //seg1.b = { 1.47868,  0.361564, 0.0600782 };

    //seg0.a = { -0.336064,  -0.370064, -0.0106807 };
    //seg0.b = { -0.443651,  -0.214583, -0.0844262 };
    //seg1.a = { 1.27944,  0.300068, -0.232831 };
    //seg1.b = { 1.09036,  0.28027, -0.291861 };

    seg0.a = { 0.477741,  0.0296258, 0.14452 };
    seg0.b = { 0.494398,  -0.0224514, -0.0711794 };
    seg1.a = { 0.474973,  0.0730834, -0.0418091 };
    seg1.b = { 0.509064,  -0.111875, 0.0182835 };


    vec3 ppp = closest(seg1.a, seg0);
    std::cout << (ppp - seg1.a).norm() << std::endl;

    std::cout << std::sqrt(ipc::edge_edge_distance(seg0.a, seg0.b, seg1.a, seg1.b)) << std::endl;
    std::cout << ipc::edge_edge_distance_gradient(seg0.a, seg0.b, seg1.a, seg1.b).transpose() << std::endl;
    auto [dist2, v_own2, v_ngb2] = closest_distance(seg0, seg1);
    std::cout << (dist2) << std::endl;
    std::cout << v_own2.transpose() << std::endl;
    std::cout << v_ngb2.transpose() << std::endl;
    std::cout << (v_own2 - v_ngb2).transpose() << std::endl;



}