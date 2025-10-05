# Streamlit wrapper
import streamlit as st

# CSS
st.markdown("""
<style>
/* Outer text: Times New Roman */
html, body, [class*="css"] { font-family: 'Times New Roman', serif; }

/* Text area / code input: Monospace */
textarea, input, .stTextArea, .stTextInput { font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Proof Checker", layout="centered")
st.title(";* Propositional Proof Checker (Prototype)")
st.write("Enter your proof below (one step per line).")

# Text area for user input
proof_text = st.text_area("Proof steps:", height=250, value="")

# Button triggers proof checking
if st.button("Check Proof"):
    try:
        # Call your existing proof-checking function
        result = check_proof(proof_text)  # <- keep your original function
        st.success(result)
    except Exception as e:
        st.error(f"Error: {e}")
# Minimal propositional proof checker prototype
# - Supports atoms (uppercase letters), parentheses, operators: ~ (not), & (and), | (or), -> (implication)
# - Proof format: list of steps (dicts). Step fields:
#     id: int
#     formula: string (parsed)
#     rule: 'premise' | 'assume' | 'and_elim' | 'and_intro' | 'imp_elim' | 'imp_intro'
#     refs: list of referenced step ids (optional)
#     subproof: list of steps (optional) for rules like imp_intro
#
# This is intentionally small and readable. It checks structure and simple rule side-conditions.
# Run the demo proofs at the bottom to see it in action.

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# --- Formula AST ---
@dataclass(frozen=True)
class Formula:
    kind: str  # 'atom','not','and','or','imp'
    val: Optional[str] = None
    left: Optional['Formula'] = None
    right: Optional['Formula'] = None
    
    def __str__(self):
        if self.kind == 'atom':
            return self.val
        if self.kind == 'not':
            return f"~{self.left}"
        if self.kind == 'and':
            return f"({self.left} & {self.right})"
        if self.kind == 'or':
            return f"({self.left} | {self.right})"
        if self.kind == 'imp':
            return f"({self.left} -> {self.right})"
        return "?"

# --- Parser (simple recursive descent) ---
token_spec = [
    ('SKIP', r'[ \t]+'),
    ('ARROW', r'->'),
    ('AND', r'&'),
    ('OR', r'\|'),
    ('NOT', r'~'),
    ('LP', r'\('),
    ('RP', r'\)'),
    ('ATOM', r'[A-Za-z][A-Za-z0-9_]*'),
]
tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_spec)

class Token:
    def __init__(self, typ, val):
        self.type = typ
        self.val = val
    def __repr__(self):
        return f"Token({self.type},{self.val})"

def tokenize(s):
    for mo in re.finditer(tok_regex, s):
        typ = mo.lastgroup
        if typ == 'SKIP':
            continue
        val = mo.group(typ)
        yield Token(typ, val)
    yield Token('EOF','')

class ParserError(Exception): pass

class Parser:
    def __init__(self, s):
        self.tokens = list(tokenize(s))
        self.pos = 0
    def peek(self):
        return self.tokens[self.pos]
    def next(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t
    def parse(self):
        f = self.parse_imp()
        if self.peek().type != 'EOF':
            raise ParserError("Extra input after formula: " + repr(self.peek()))
        return f
    # implication is lowest precedence (right-assoc)
    def parse_imp(self):
        left = self.parse_or()
        if self.peek().type == 'ARROW':
            self.next()
            right = self.parse_imp()
            return Formula('imp', left=left, right=right)
        return left
    def parse_or(self):
        left = self.parse_and()
        while self.peek().type == 'OR':
            self.next()
            right = self.parse_and()
            left = Formula('or', left=left, right=right)
        return left
    def parse_and(self):
        left = self.parse_not()
        while self.peek().type == 'AND':
            self.next()
            right = self.parse_not()
            left = Formula('and', left=left, right=right)
        return left
    def parse_not(self):
        if self.peek().type == 'NOT':
            self.next()
            child = self.parse_not()
            return Formula('not', left=child)
        return self.parse_atom()
    def parse_atom(self):
        t = self.peek()
        if t.type == 'ATOM':
            self.next()
            return Formula('atom', val=t.val)
        if t.type == 'LP':
            self.next()
            inner = self.parse_imp()
            if self.peek().type != 'RP':
                raise ParserError("Missing )")
            self.next()
            return inner
        raise ParserError("Unexpected token: " + repr(t))

def parse_formula(s: str) -> Formula:
    return Parser(s).parse()

# --- Proof checker ---
class ProofError(Exception): pass

class ProofChecker:
    def __init__(self, steps: List[Dict[str, Any]]):
        self.raw_steps = steps
        self.env: Dict[int, Formula] = {}  # validated formulas by id
    
    def parse_step_formula(self, step):
        try:
            return parse_formula(step['formula'])
        except ParserError as e:
            raise ProofError(f"Parsing error in step {step.get('id')}: {e}")
    
    def lookup(self, ref_id: int) -> Formula:
        if ref_id not in self.env:
            raise ProofError(f"Reference to unknown or unchecked step id {ref_id}")
        return self.env[ref_id]
    
    def formulas_equal(self, f1: Formula, f2: Formula) -> bool:
        return f1 == f2  # dataclass equality works since fields are structural
    
    def check_and_elim(self, step_formula: Formula, refs: List[int]):
        if len(refs) != 1:
            raise ProofError("and_elim needs exactly 1 reference")
        conj = self.lookup(refs[0])
        if conj.kind != 'and':
            raise ProofError("and_elim reference is not a conjunction")
        if self.formulas_equal(step_formula, conj.left) or self.formulas_equal(step_formula, conj.right):
            return True
        raise ProofError("and_elim target is not a conjunct of referenced conjunction")
    
    def check_and_intro(self, step_formula: Formula, refs: List[int]):
        if len(refs) != 2:
            raise ProofError("and_intro needs exactly 2 references")
        a = self.lookup(refs[0])
        b = self.lookup(refs[1])
        expected = Formula('and', left=a, right=b)
        if self.formulas_equal(step_formula, expected):
            return True
        # also accept symmetric if user swapped refs
        expected2 = Formula('and', left=b, right=a)
        if self.formulas_equal(step_formula, expected2):
            return True
        raise ProofError("and_intro target does not match conjunction of refs")
    
    def check_imp_elim(self, step_formula: Formula, refs: List[int]):
        if len(refs) != 2:
            raise ProofError("imp_elim (MP) needs exactly 2 references: A and A->B")
        a = self.lookup(refs[0])
        imp = self.lookup(refs[1])
        if imp.kind != 'imp':
            # maybe refs reversed
            a, imp = imp, a
            if imp.kind != 'imp':
                raise ProofError("imp_elim requires one ref to be an implication A->B and the other to be A")
        if not self.formulas_equal(imp.left, a):
            raise ProofError(f"imp_elim mismatch: implication left side {imp.left} doesn't match {a}")
        if self.formulas_equal(step_formula, imp.right):
            return True
        raise ProofError("imp_elim target does not match implication consequent")
    
    def check_imp_intro(self, step: Dict[str, Any]):
        # expects a 'subproof' list where first step is an 'assume' of A and last step is B; infers A->B equals step['formula']
        if 'subproof' not in step:
            raise ProofError("imp_intro requires a 'subproof' field containing the subproof steps")
        sub = step['subproof']
        if len(sub) == 0:
            raise ProofError("empty subproof")
        # check the subproof locally with a new checker that allows the assumption to be present in env
        # first step must be assume
        if sub[0].get('rule') != 'assume':
            raise ProofError("first step of subproof must be an 'assume' introducing the antecedent")
        assumed_formula = parse_formula(sub[0]['formula'])
        # create a checker for the subproof that inherits global env but allows the assumption id to be present
        local_checker = ProofChecker([])
        # copy global validated formulas into local env (so subproof can reference them)
        local_checker.env = dict(self.env)
        # assign a synthetic id for the assumption inside the subproof
        # process steps with ids starting at a high offset to avoid clashes; but we will just treat refs as local ids
        # For simplicity, renumber subproof steps sequentially starting at 1 for lookup
        renum = {}
        local_env_ids = {}
        new_steps = []
        for i, s in enumerate(sub, start=1):
            s_copy = dict(s)
            s_copy['id'] = i
            new_steps.append(s_copy)
        # run through new_steps, allowing 'assume' to add the assumed_formula to env
        for s in new_steps:
            fid = s['id']
            rule = s.get('rule')
            f = local_checker.parse_step_formula(s)
            if rule == 'assume':
                local_checker.env[fid] = f
                continue
            # handle rules supported inside subproof: premise, assume, and_elim, and_intro, imp_elim
            if rule == 'premise':
                local_checker.env[fid] = f
                continue
            if rule == 'and_elim':
                refs = s.get('refs', [])
                # map refs: if they are >0 and refer to subproof numbering, keep as is; else allow references to global by string 'G:ID'
                # For simplicity, assume refs refer to local numbering if <= len(new_steps), else error
                local_checker.check_and_elim(f, refs)
                local_checker.env[fid] = f
                continue
            if rule == 'and_intro':
                refs = s.get('refs', [])
                local_checker.check_and_intro(f, refs)
                local_checker.env[fid] = f
                continue
            if rule == 'imp_elim':
                refs = s.get('refs', [])
                local_checker.check_imp_elim(f, refs)
                local_checker.env[fid] = f
                continue
            raise ProofError(f"Unsupported rule '{rule}' inside subproof")
        # after local checking, the last formula is the subproof conclusion
        conclusion = local_checker.env[len(new_steps)]
        # now check that the top-level step formula equals (assumed_formula -> conclusion)
        expected_imp = Formula('imp', left=assumed_formula, right=conclusion)
        declared = self.parse_step_formula(step)
        if self.formulas_equal(declared, expected_imp):
            return True
        raise ProofError("imp_intro target formula does not equal implication from assumed antecedent to subproof conclusion")
    
    def check_step(self, step: Dict[str, Any]):
        rule = step.get('rule')
        if not rule:
            raise ProofError("missing rule in step")
        f = self.parse_step_formula(step)
        if rule == 'premise':
            # accept any formula
            return f
        if rule == 'assume':
            return f
        if rule == 'and_elim':
            refs = step.get('refs', [])
            self.check_and_elim(f, refs)
            return f
        if rule == 'and_intro':
            refs = step.get('refs', [])
            self.check_and_intro(f, refs)
            return f
        if rule == 'imp_elim':
            refs = step.get('refs', [])
            self.check_imp_elim(f, refs)
            return f
        if rule == 'imp_intro':
            self.check_imp_intro(step)
            return f
        raise ProofError(f"Unknown or unsupported rule '{rule}'")
    
    def run(self):
        # top-level steps processed sequentially; ids must be unique integers
        for step in self.raw_steps:
            sid = step.get('id')
            if not isinstance(sid, int):
                raise ProofError("each step must have an integer 'id'")
            if sid in self.env:
                raise ProofError(f"duplicate step id {sid}")
            checked_formula = self.check_step(step)
            # on success, store in env
            self.env[sid] = checked_formula
        return True

# --- Demo proofs ---
proof1 = [
    {"id":1, "formula":"(P & Q)", "rule":"premise"},
    {"id":2, "formula":"P", "rule":"and_elim", "refs":[1]}
]

# Proof of P -> (Q -> P) using nested subproofs
proof2 = [
    {"id":1, "formula":"(P -> (Q -> P))", "rule":"premise"},  # we will instead *construct* it using imp_intro, so show intended as target
]

# Let's construct the imp_intro proof as a top-level step with a subproof
proof_imp = [
    {"id":1, "formula":"(P -> (Q -> P))", "rule":"imp_intro", "subproof":[
        {"id":101, "formula":"P", "rule":"assume"},
        {"id":102, "formula":"(Q -> P)", "rule":"imp_intro", "subproof":[
            {"id":201, "formula":"Q", "rule":"assume"},
            {"id":202, "formula":"P", "rule":"assume"}  # here we assume P from outer scope; for demo, we mark it as assume again
        ]}
    ]}
]

# For the nested imp_intro demo above, our simple subproof checker expects local renumbered ids and direct structure
# Let's build a correct version: outer subproof assumes P, inner subproof assumes Q and reuses P by asserting it via a referent.
# To keep it simple for this prototype, we allow the inner to 'assume' P as well and conclude P.
proof_imp_simple = [
    {"id":1, "formula":"(P -> (Q -> P))", "rule":"imp_intro", "subproof":[
        {"id":1, "formula":"P", "rule":"assume"},
        {"id":2, "formula":"(Q -> P)", "rule":"imp_intro", "subproof":[
            {"id":1, "formula":"Q", "rule":"assume"},
            {"id":2, "formula":"P", "rule":"assume"}
        ]}
    ]}
]

def try_proof(steps):
    try:
        pc = ProofChecker(steps)
        ok = pc.run()
        print("Proof accepted. Final environment:")
        for k,v in sorted(pc.env.items()):
            print(f"  {k}: {v}")
    except ProofError as e:
        print("Proof error:", e)

print("=== Demo 1: and_elim ===")
try_proof(proof1)
print("\n=== Demo 2: implication introduction (nested) - simple variant ===")
try_proof(proof_imp_simple)

# Example of a malformed proof to show error messaging
bad_proof = [
    {"id":1, "formula":"(P & Q)", "rule":"premise"},
    {"id":2, "formula":"Q", "rule":"and_elim", "refs":[1]},  # actually Q is fine; try a wrong target
    {"id":3, "formula":"R", "rule":"and_elim", "refs":[1]}   # wrong: R not a conjunct
]
print("\n=== Demo 3: malformed proof (expect error) ===")
try_proof(bad_proof)
