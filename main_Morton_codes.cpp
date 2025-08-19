
#include <iostream>
#include <vector>
#include <bitset>
#include <cmath>
#include <algorithm>

#include "D:\work\libmorton\include\libmorton\morton.h"

//
//// Morton 코드를 계산하는 함수
//// x, y, z는 3차원 좌표이며, 각 좌표는 double 타입입니다.
//uint64_t mortonEncode(double x, double y, double z) {
//    // 좌표 값을 32비트 정수로 변환 (여기서는 10진수 좌표를 1024 단위로 스케일링)
//    uint32_t ix = static_cast<uint32_t>(x * 1024);
//    uint32_t iy = static_cast<uint32_t>(y * 1024);
//    uint32_t iz = static_cast<uint32_t>(z * 1024);
//
//    // 비트를 인터리빙하여 Morton 코드를 생성하는 함수
//    auto interleaveBits = [](uint32_t n) {
//        uint64_t result = 0;
//        for (int i = 0; i < 32; ++i) {
//            result |= ((n >> i) & 1ULL) << (3 * i);
//        }
//        return result;
//        };
//
//    // 각 차원의 비트를 인터리빙하여 최종 Morton 코드 생성
//    return (interleaveBits(ix) | (interleaveBits(iy) << 1) | (interleaveBits(iz) << 2));
//}

//#include <cstdint>
//#include <bit>
//
//uint64_t mortonEncode(double x, double y, double z) {
//    // 좌표를 32비트 정수로 변환 (0-1 범위를 가정)
//    uint32_t ix = static_cast<uint32_t>(x * 0xFFFFFFFF);
//    uint32_t iy = static_cast<uint32_t>(y * 0xFFFFFFFF);
//    uint32_t iz = static_cast<uint32_t>(z * 0xFFFFFFFF);
//
//    // 비트 확산 (Bit Spread) - C++20의 bit_width 사용
//    auto spread = [](uint32_t x) {
//        uint64_t result = x;
//        result = (result | (result << 16)) & 0x0000FFFF0000FFFFUL;
//        result = (result | (result << 8)) & 0x00FF00FF00FF00FFULL;
//        result = (result | (result << 4)) & 0x0F0F0F0F0F0F0F0FULL;
//        result = (result | (result << 2)) & 0x3333333333333333ULL;
//        result = (result | (result << 1)) & 0x5555555555555555ULL;
//        return result;
//        };
//
//    // Morton 코드 생성
//    return spread(ix) | (spread(iy) << 1) | (spread(iz) << 2);
//}

//
//// 좌표를 정수로 스케일링하는 함수
//uint64_t scaleCoordinate(double coord, double min, double max, int bits) {
//    // 정규화: [min, max] 범위를 [0, 2^bits - 1]로 매핑
//    double normalized = (coord - min) / (max - min);
//    return static_cast<uint64_t>(std::round(normalized * ((1ULL << bits) - 1)));
//}
//
//// 비트를 확장하여 64비트 정수로 만드는 함수
//uint64_t expandBits(uint64_t v) {
//    v = (v | (v << 32)) & 0x1F00000000FFFF;
//    v = (v | (v << 16)) & 0x1F0000FF0000FF;
//    v = (v | (v << 8)) & 0x100F00F00F00F00F;
//    v = (v | (v << 4)) & 0x10C30C30C30C30C3;
//    v = (v | (v << 2)) & 0x1249249249249249;
//    return v;
//}
//
//// 3차원 Morton 코드를 인코딩하는 함수
//uint64_t mortonEncode(double x, double y, double z, double min = -1.e9, double max = 1.e9) {
//    int bits = 21; // 각 좌표당 21비트, 총 63비트 사용
//    uint64_t scaledX = scaleCoordinate(x, min, max, bits);
//    uint64_t scaledY = scaleCoordinate(y, min, max, bits);
//    uint64_t scaledZ = scaleCoordinate(z, min, max, bits);
//
//    uint64_t interleavedX = expandBits(scaledX);
//    uint64_t interleavedY = expandBits(scaledY) << 1;
//    uint64_t interleavedZ = expandBits(scaledZ) << 2;
//
//    return interleavedX | interleavedY | interleavedZ;
//}


// 비트를 3개씩 분리하는 함수
inline uint64_t splitBy3(uint64_t a) {
    a &= 0x1fffff; // 상위 21비트만 사용 (64비트 Morton 코드에 맞게)
    a = (a | a << 32) & 0x1f00000000ffff;
    a = (a | a << 16) & 0x1f0000ff0000ff;
    a = (a | a << 8) & 0x100f00f00f00f00f;
    a = (a | a << 4) & 0x10c30c30c30c30c3;
    a = (a | a << 2) & 0x1249249249249249;
    return a;
}

// 주어진 좌표를 Morton 코드로 변환하는 함수
uint64_t mortonEncode(double x, double y, double z,
    double minX = 0.1, double maxX = 1110.7) {

    // 좌표를 [0, 1] 범위로 정규화
    double normX = (x - minX) / (maxX - minX);
    double normY = (y - minX) / (minX - minX);
    double normZ = (z - minX) / (minX - minX);

    // 정규화된 좌표를 21비트 정수로 변환
    uint64_t intX = static_cast<uint64_t>(normX * ((1 << 21) - 1));
    uint64_t intY = static_cast<uint64_t>(normY * ((1 << 21) - 1));
    uint64_t intZ = static_cast<uint64_t>(normZ * ((1 << 21) - 1));

    // 좌표의 비트를 interleave하여 Morton 코드 생성
    uint64_t mortonCode = (splitBy3(intX) | (splitBy3(intY) << 1) | (splitBy3(intZ) << 2));

    return mortonCode;
}

// Octree 노드 구조체 정의
struct OctreeNode {
    double x, y, z; // 노드의 중심 좌표
    uint64_t mortonCode; // Morton 코드

    OctreeNode(double x_, double y_, double z_)
        : x(x_), y(y_), z(z_), mortonCode(mortonEncode(x_, y_, z_)) {}
};

// Octree 노드를 순서대로 출력하는 함수
void printOctreeNodes(const std::vector<OctreeNode>& nodes) {
    for (const auto& node : nodes) {
        std::cout << "Node at (" << node.x << ", " << node.y << ", " << node.z
            << ") -> Morton Code: " << std::bitset<64>(node.mortonCode)
            << " (" << node.mortonCode << ")" << std::endl;
    }
}

int main() {


    uint_fast32_t max_valid = (1 << 22) - 1;
    std::cout << max_valid << std::endl;
    for (int i = 0; i < 101; ++i) {
        uint_fast32_t x1 = static_cast<uint_fast32_t>(
            (0.0 + (double)i * 0.01) * max_valid);
        uint_fast32_t y1 = static_cast<uint_fast32_t>(
            (0.0 + (double)i * 0.01) * max_valid);
        uint_fast32_t z1 = static_cast<uint_fast32_t>(
            (0.0 + (double)i * 0.01) * max_valid);
        std::cout << libmorton::morton3D_64_encode(x1, y1, z1) << std::endl;
    }

    //// Octree에 포함된 노드들 생성 (임의의 좌표)
    //std::vector<OctreeNode> nodes = {
    //    {0.5, 0.5, 0.5},
    //    {0.4, 0.6, 0.2},
    //    {10.4, 1110.6, 10.2},
    //    {10.4, 1110.6, 10.3},
    //    {10.4, 1110.7, 10.2},
    //    {0.1, 0.2, 0.3},
    //    {0.7, 0.8, 0.9},
    //    {10.5, 1110.6, 10.2}
    //};

    //// Morton 코드에 따라 정렬 (Morton 코드가 작은 순서대로 정렬)
    //std::sort(nodes.begin(), nodes.end(), [](const OctreeNode& a, const OctreeNode& b) {
    //    return a.mortonCode < b.mortonCode;
    //    });

    //// 정렬된 노드 출력
    //printOctreeNodes(nodes);

    return 0;
}