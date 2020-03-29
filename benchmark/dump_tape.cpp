#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

int main(int argc, char **argv)
{
    libfive::Tree t = libfive::Tree::X();
    if (argc == 2) {
        std::ifstream ifs;
        ifs.open(argv[1]);
        if (ifs.is_open()) {
            auto a = libfive::Archive::deserialize(ifs);
            t = a.shapes.front().tree;
        } else {
            fprintf(stderr, "Could not open file %s\n", argv[1]);
            exit(1);
        }
    } else {
        auto X = libfive::Tree::X();
        auto Y = libfive::Tree::Y();
        auto Z = libfive::Tree::Z();
        t = min(sqrt((X + 0.5)*(X + 0.5) + Y*Y + Z*Z) - 0.25,
                sqrt((X - 0.5)*(X - 0.5) + Y*Y + Z*Z) - 0.25);
    }

    std::cout << R"(__global__ void evalRawTape(int image_size_px, int32_t* image)
{

    uint32_t px = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t py = threadIdx.y + blockIdx.y * blockDim.y;

    if (px >= image_size_px && py >= image_size_px) {
        return;
    }

    const float x = 2.0f * ((px + 0.5f) / image_size_px - 0.5f);
    const float y = 2.0f * ((py + 0.5f) / image_size_px - 0.5f);
    const float z = 0.0f;

)";

    const auto ordered = t.orderedDfs();
    for (auto& o : ordered) {
        using namespace libfive::Opcode;

        switch (o->op) {
            case CONSTANT:
            case VAR_X:
            case VAR_Y:
            case VAR_Z:
            case VAR_FREE:
                continue;
            default:
                break;
        }
        std::cout << "    const float v" << (unsigned long)o.id() << " = ";
        std::string lhs, rhs;
        if (o->lhs) {
            if (o->lhs->op == CONSTANT) {
                lhs = std::to_string(o->lhs->value) + "f";
            } else if (o->lhs->op == VAR_X) {
                lhs = "x";
            } else if (o->lhs->op == VAR_Y) {
                lhs = "y";
            } else if (o->lhs->op == VAR_Z) {
                lhs = "z";
            } else {
                lhs = "v" + std::to_string((unsigned long)o.lhs().id());
            }
        }
        if (o->rhs) {
            if (o->rhs->op == CONSTANT) {
                rhs = std::to_string(o->rhs->value) + "f";
            } else if (o->lhs->op == VAR_X) {
                rhs = "x";
            } else if (o->lhs->op == VAR_Y) {
                rhs = "y";
            } else if (o->lhs->op == VAR_Z) {
                rhs = "z";
            } else {
                rhs = "v" + std::to_string((unsigned long)o.rhs().id());
            }
        }


        switch (o->op) {
            case CONSTANT:
            case VAR_X:
            case VAR_Y:
            case VAR_Z:
            case VAR_FREE:
                break;

            case CONST_VAR:
                std::cout << lhs;
                break;

            case OP_SQUARE:
                std::cout << lhs << " * " << lhs;
                break;
            case OP_NEG:
                std::cout << "-" << lhs;
                break;
            case OP_SQRT:
            case OP_SIN:
            case OP_COS:
            case OP_TAN:
            case OP_ASIN:
            case OP_ACOS:
            case OP_ATAN:
            case OP_EXP:
            case OP_ABS:
            case OP_LOG:
                std::cout << toOpString(o->op) << "(" << lhs << ")";
                break;
            case OP_RECIP:
                std::cout << "1 / " << lhs;
                break;

            case OP_ADD:
            case OP_SUB:
            case OP_MUL:
            case OP_DIV:
                std::cout << "" << lhs << " " << toOpString(o->op) << " " << rhs;
                break;

            case OP_MIN:
            case OP_MAX:
            case OP_ATAN2:
            case OP_POW:
                std::cout << toOpString(o->op);
                std::cout << "(" << lhs << ", " << rhs << ")";
                break;

            case OP_MOD:
                std::cout << lhs << " % " << rhs;
                break;

            case OP_NTH_ROOT:
                std::cout << "pow(" << lhs << ", 1/" << rhs << ")";
                break;
            default:
                std::cerr << "Cannot print opcode " << toScmString(o->op) << "\n";
        }
        std::cout << ";\n";
    }

    std::cout << "    if (v" << (unsigned long)ordered.rbegin()->id() << R"( < 0.0f) {
        image[px + py * image_size_px] = 255;
    } else {
        image[px + py * image_size_px] = 0;
    }
}
)";
}
