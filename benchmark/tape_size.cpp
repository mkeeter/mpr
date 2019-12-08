#include "../gui/interpreter.hpp"
#include "renderable.hpp"

int main(int, char**)
{
    std::string txt =  R"(But this rough magic I
here abjure, and when
I have required some
heavenly music, which even
now I do, to work mine
end upon their senses that
this airy charm is for, I'll
break my staff, bury it
certain fathoms in the
earth, and deeper than did
ever plummet sound
I'll drown my book.)";

    Interpreter interpreter;
    std::cout << '[';
    for (unsigned i=1; i < txt.size(); ++i) {
        std::cerr << "\r" << i << " / " << txt.size() << "     ";
        auto s = txt.substr(0, i);
        interpreter.eval("(text \"" + s + "\")");
        assert(interpreter.shapes.size() == 1);
        auto r = Renderable::build(interpreter.shapes.begin()->second, 512, 2);
        std::cout << '[' << r->tape.num_clauses << ", " << r->tape.num_regs << "],\n";
    }
    std::cout << "]\n";
}
