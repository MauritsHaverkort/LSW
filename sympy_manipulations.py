import sympy as sp
from sympy import Add, Mul, Pow, Symbol 
from sympy.physics.quantum import Dagger

def extract_terms_of_order(expr, order_of_operator):
        expr = sp.expand(expr)
        expr_terms = expr.as_ordered_terms()
        new_expr = sp.S(0) #this ensures that it is still a sympy expression
        
        for term in expr_terms:
            if total_power(term) == order_of_operator:
                new_expr += term
        return new_expr


def _is_nc_symbol(x):
    # check if x is a noncommutative Symbol or Dagger of one
    return (isinstance(x, Symbol) and (x.is_commutative is False)) or (x.func is Dagger and _is_nc_symbol(x.args[0]))

def _count_in_factor(f):
    # Dagger(a) -> 1 if a is a noncommutative Symbol
    if f.func.__name__ == "Dagger" and _is_nc_symbol(f.args[0]):
        return 1

    # Powers
    if isinstance(f, Pow):
        base, exp = f.as_base_exp()
        if base.func.__name__ == "Dagger" and _is_nc_symbol(base.args[0]):
            if not exp.is_integer:
                raise ValueError(f"Non-integer power {exp} on {f}")
            return abs(int(exp))
        if _is_nc_symbol(base):
            if not exp.is_integer:
                raise ValueError(f"Non-integer power {exp} on {f}")
            return abs(int(exp))
        # (a*b)**n or powers of non-atomic bases are ignored
        return 0

    # Products
    if isinstance(f, Mul):
        temp = [_count_in_factor(arg) for arg in f.args]
        return sum(temp)

    # Single noncommutative symbol
    if _is_nc_symbol(f):
        return 1

    # Everything else (numbers, commutative symbols, functions, etc.)
    return 0

def total_power(expr):
    """
    Count total occurrences ("power") of noncommutative Symbols per additive term.
    - If `expr` is a sum, returns a list (one count per term, ordered).
    - Otherwise returns a single integer.
    Notes:
    * Dagger(a) counts as 1; Dagger(a)**n counts as n.
    * Negative integer powers contribute |n| (e.g., a**-2 contributes 2).
    * Powers of compound noncommutative products like (a*b)**2 are not expanded and contribute 0.
    """
    if isinstance(expr, Add):
        terms = expr.as_ordered_terms()
        return [_count_in_factor(t) for t in terms]
    return _count_in_factor(expr)

from collections import defaultdict

def nc_coeff_dict(expr):
    """
    Return {noncommutative_monomial: coefficient} for expr.
    expr is expanded first.
    All commutative factors (numbers, I, pi, functions of commutative symbols, …)
    are absorbed into the coefficient. The key is a product of ONLY
    noncommutative factors (order preserved).
    """
    expr = sp.expand(expr)
    out = defaultdict(lambda: sp.S.Zero)

    for term in sp.Add.make_args(expr):                # split sum into terms
        coeff, factors = term.as_coeff_mul(rational=False)  # pull out numeric coeff (incl. I)
        for f in factors:
            if f.is_commutative:
                coeff *= f                             # absorb commutative stuff
            else:
                # keep noncommutative factor in the monomial (order matters)
                out[None] = None  # just to ensure dict type; no effect

        # rebuild monomial from the *noncommutative* factors in order
        nc_factors = [f for f in factors if not f.is_commutative]
        monomial = sp.Mul(*nc_factors) if nc_factors else sp.S.One
        out[monomial] += coeff

    # optional: drop the 1 key if you don't want the "pure scalar" term
    if sp.S.One in out and out[sp.S.One] == 0:
        out.pop(sp.S.One)
    return dict(out)

    