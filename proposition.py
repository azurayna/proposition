# streamlit_proof_checker.py
import streamlit as st
import json
import traceback

# Page config must be first
st.set_page_config(page_title="Proof Checker", layout="centered")

# CSS
st.markdown("""
<style>
/* Outer text: Times New Roman */
html, body, [class*="css"] { font-family: 'Times New Roman', serif; }

/* Text area / code input: Monospace fallback */
textarea, input, .stTextArea, .stTextInput { font-family: 'Times New Roman', monospace; }
</style>
""", unsafe_allow_html=True)

st.title("Propositional Proof Checker (Prototype)")
st.write("Paste a JSON array of proof steps (one list entry per step). See the example below and the demo proofs in the console.")

# Example JSON (helpful default)
example = json.dumps([
    {"id": 1, "formula": "(P & Q)", "rule": "premise"},
    {"id": 2, "formula": "P", "rule": "and_elim", "refs": [1]}
], indent=2)

proof_text = st.text_area("Proof steps (JSON):", height=300, value=example)

if st.button("Check Proof"):
    try:
        result = check_proof(proof_text)
        if isinstance(result, dict) and result.get("ok"):
            st.success(result["message"])
            st.subheader("Final validated environment")
            # display as simple table
            for sid, f in sorted(result["env"].items()):
                st.write(f"{sid}: {f}")
        else:
            # error returned as string
            st.error(result if isinstance(result, str) else str(result))
    except Exception as e:
        st.error("Internal error: " + str(e))
        st.text(traceback.format_exc())

# ----- (Below this line is your proof-checker implementation; unchanged except for imports used above) -----
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
        if 'subproof' not in step:
            raise ProofError("imp_intro requires a 'subproof' field containing the subproof steps")
        sub = step['subproof']
        if len(sub) == 0:
            raise ProofError("empty subproof")
        if sub[0].get('rule') != 'assume':
            raise ProofError("first step of subproof must be an 'assume' introducing the antecedent")
        assumed_formula = parse_formula(sub[0]['formula'])
        # create a checker for the subproof that inherits global env but allows the assumption to be present
        local_checker = ProofChecker([])
        local_checker.env = dict(self.env)
        new_steps = []
        for i, s in enumerate(sub, start=1):
            s_copy = dict(s)
            s_copy['id'] = i
            new_steps.append(s_copy)
        for s in new_steps:
            fid = s['id']
            rule = s.get('rule')
            f = local_checker.parse_step_formula(s)
            if rule == 'assume':
                local_checker.env[fid] = f
                continue
            if rule == 'premise':
                local_checker.env[fid] = f
                continue
            if rule == 'and_elim':
                refs = s.get('refs', [])
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
            if rule == 'imp_intro':
                # recursion for nested imp_intro
                local_checker.check_imp_intro(s)
                local_checker.env[fid] = f
                continue
            raise ProofError(f"Unsupported rule '{rule}' inside subproof")
        conclusion = local_checker.env[len(new_steps)]
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
        for step in self.raw_steps:
            sid = step.get('id')
            if not isinstance(sid, int):
                raise ProofError("each step must have an integer 'id'")
            if sid in self.env:
                raise ProofError(f"duplicate step id {sid}")
            checked_formula = self.check_step(step)
            self.env[sid] = checked_formula
        return True

# Top-level wrapper: read JSON and run
def check_proof(proof_text: str):
    """
    Input: proof_text (JSON array of step dicts)
    Output: dict with ok/message and environment on success; else error string
    """
    try:
        steps = json.loads(proof_text)
    except Exception as e:
        return f"JSON parsing error: {e}"

    if not isinstance(steps, list):
        return "Input must be a JSON array (list) of proof steps."

    try:
        pc = ProofChecker(steps)
        pc.run()
        # return environment as map sid -> str(formula)
        env_readable = {sid: str(f) for sid, f in pc.env.items()}
        return {"ok": True, "message": "Proof accepted.", "env": env_readable}
    except ProofError as e:
        tb = traceback.format_exc()
        return f"Proof error: {e}\n\nTraceback:\n{tb}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Unexpected error: {e}\n\nTraceback:\n{tb}"
