
#include <iostream>
#include <vector>
#include <bitset>
#include <cmath>
#include <algorithm>

#include "D:\work\libmorton\include\libmorton\morton.h"

//
//// Morton �ڵ带 ����ϴ� �Լ�
//// x, y, z�� 3���� ��ǥ�̸�, �� ��ǥ�� double Ÿ���Դϴ�.
//uint64_t mortonEncode(double x, double y, double z) {
//    // ��ǥ ���� 32��Ʈ ������ ��ȯ (���⼭�� 10���� ��ǥ�� 1024 ������ �����ϸ�)
//    uint32_t ix = static_cast<uint32_t>(x * 1024);
//    uint32_t iy = static_cast<uint32_t>(y * 1024);
//    uint32_t iz = static_cast<uint32_t>(z * 1024);
//
//    // ��Ʈ�� ���͸����Ͽ� Morton �ڵ带 �����ϴ� �Լ�
//    auto interleaveBits = [](uint32_t n) {
//        uint64_t result = 0;
//        for (int i = 0; i < 32; ++i) {
//            result |= ((n >> i) & 1ULL) << (3 * i);
//        }
//        return result;
//        };
//
//    // �� ������ ��Ʈ�� ���͸����Ͽ� ���� Morton �ڵ� ����
//    return (interleaveBits(ix) | (interleaveBits(iy) << 1) | (interleaveBits(iz) << 2));
//}

//#include <cstdint>
//#include <bit>
//
//uint64_t mortonEncode(double x, double y, double z) {
//    // ��ǥ�� 32��Ʈ ������ ��ȯ (0-1 ������ ����)
//    uint32_t ix = static_cast<uint32_t>(x * 0xFFFFFFFF);
//    uint32_t iy = static_cast<uint32_t>(y * 0xFFFFFFFF);
//    uint32_t iz = static_cast<uint32_t>(z * 0xFFFFFFFF);
//
//    // ��Ʈ Ȯ�� (Bit Spread) - C++20�� bit_width ���
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
//    // Morton �ڵ� ����
//    return spread(ix) | (spread(iy) << 1) | (spread(iz) << 2);
//}

//
//// ��ǥ�� ������ �����ϸ��ϴ� �Լ�
//uint64_t scaleCoordinate(double coord, double min, double max, int bits) {
//    // ����ȭ: [min, max] ������ [0, 2^bits - 1]�� ����
//    double normalized = (coord - min) / (max - min);
//    return static_cast<uint64_t>(std::round(normalized * ((1ULL << bits) - 1)));
//}
//
//// ��Ʈ�� Ȯ���Ͽ� 64��Ʈ ������ ����� �Լ�
//uint64_t expandBits(uint64_t v) {
//    v = (v | (v << 32)) & 0x1F00000000FFFF;
//    v = (v | (v << 16)) & 0x1F0000FF0000FF;
//    v = (v | (v << 8)) & 0x100F00F00F00F00F;
//    v = (v | (v << 4)) & 0x10C30C30C30C30C3;
//    v = (v | (v << 2)) & 0x1249249249249249;
//    return v;
//}
//
//// 3���� Morton �ڵ带 ���ڵ��ϴ� �Լ�
//uint64_t mortonEncode(double x, double y, double z, double min = -1.e9, double max = 1.e9) {
//    int bits = 21; // �� ��ǥ�� 21��Ʈ, �� 63��Ʈ ���
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


// ��Ʈ�� 3���� �и��ϴ� �Լ�
inline uint64_t splitBy3(uint64_t a) {
    a &= 0x1fffff; // ���� 21��Ʈ�� ��� (64��Ʈ Morton �ڵ忡 �°�)
    a = (a | a << 32) & 0x1f00000000ffff;
    a = (a | a << 16) & 0x1f0000ff0000ff;
    a = (a | a << 8) & 0x100f00f00f00f00f;
    a = (a | a << 4) & 0x10c30c30c30c30c3;
    a = (a | a << 2) & 0x1249249249249249;
    return a;
}

// �־��� ��ǥ�� Morton �ڵ�� ��ȯ�ϴ� �Լ�
uint64_t mortonEncode(double x, double y, double z,
    double minX = 0.1, double maxX = 1110.7) {

    // ��ǥ�� [0, 1] ������ ����ȭ
    double normX = (x - minX) / (maxX - minX);
    double normY = (y - minX) / (minX - minX);
    double normZ = (z - minX) / (minX - minX);

    // ����ȭ�� ��ǥ�� 21��Ʈ ������ ��ȯ
    uint64_t intX = static_cast<uint64_t>(normX * ((1 << 21) - 1));
    uint64_t intY = static_cast<uint64_t>(normY * ((1 << 21) - 1));
    uint64_t intZ = static_cast<uint64_t>(normZ * ((1 << 21) - 1));

    // ��ǥ�� ��Ʈ�� interleave�Ͽ� Morton �ڵ� ����
    uint64_t mortonCode = (splitBy3(intX) | (splitBy3(intY) << 1) | (splitBy3(intZ) << 2));

    return mortonCode;
}

// Octree ��� ����ü ����
struct OctreeNode {
    double x, y, z; // ����� �߽� ��ǥ
    uint64_t mortonCode; // Morton �ڵ�

    OctreeNode(double x_, double y_, double z_)
        : x(x_), y(y_), z(z_), mortonCode(mortonEncode(x_, y_, z_)) {}
};

// Octree ��带 ������� ����ϴ� �Լ�
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

    //// Octree�� ���Ե� ���� ���� (������ ��ǥ)
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

    //// Morton �ڵ忡 ���� ���� (Morton �ڵ尡 ���� ������� ����)
    //std::sort(nodes.begin(), nodes.end(), [](const OctreeNode& a, const OctreeNode& b) {
    //    return a.mortonCode < b.mortonCode;
    //    });

    //// ���ĵ� ��� ���
    //printOctreeNodes(nodes);

    return 0;
}