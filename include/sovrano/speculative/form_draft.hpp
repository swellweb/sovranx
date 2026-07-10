#pragma once

#include <string>

namespace sovrano::speculative {

// The prompter ("suggeritore"): grammar as a draft source.
//
// Grammar-constrained decoding uses structure to FORBID tokens; this is
// the inversion — using structure to PROPOSE them speculatively. In
// structured output (numbered lists, bullets, JSON), a share of tokens is
// syntax, not thought: "2. " after "1. ", the bullet that opens the next
// line, the quote that must follow a comma. Those tokens are predictable
// on content the model has NEVER produced before — where prompt-lookup
// and the generation archive are blind.
//
// propose() is pure text-in/text-out: given the tail of the text produced
// so far, it returns the structural continuation it would bet on (empty
// when no rule fires). The caller tokenizes the proposal and feeds it to
// the standard speculative verifier: a wrong bet costs one rejected
// batch slot, a right one saves a full forward pass.
class FormDraft {
public:
    // `tail` = the last few hundred characters generated so far.
    // Rules (v1):
    //   - numbered lists:  "...\n3. xyz\n"        -> "4. "
    //   - bullet lists:    "...\n- xyz\n"          -> "- "
    //   - JSON object key: '...","' or '...{ "'    -> nothing yet (v2)
    static std::string propose(const std::string& tail);
};

}  // namespace sovrano::speculative
