from sympy import Function
from sympy.printing.pretty.stringpict import prettyForm
import sympy as sp
from collections import defaultdict

class VisualDelta(Function):
    def _latex(self,printer):
        up = printer._print(self.args[0])
        down = printer._print(self.args[1])
        return r"\delta^{%s}_{%s}" % (up, down)
    
    def _pretty(self,printer):
        up = printer._print(self.args[0])
        down = printer._print(self.args[1])

        return prettyForm(f"δ^{up}_{down}")
    

def collect_color_structure(expr, nc_symbol, delta_func):
    """
    Collects a SymPy expression first by the power of Nc, 
    and then groups terms with the exact same SUNDelta chains.
    """
    # 1. Expand to ensure a flat sum of terms
    expanded_expr = sp.expand(expr)
    
    # 2. Collect by powers of Nc (returns a dictionary)
    nc_dict = sp.collect(expanded_expr, nc_symbol, evaluate=False)
    
    final_expr = 0
    for nc_pow, coeff in nc_dict.items():
        delta_dict = defaultdict(lambda: 0)
        
        # 3. Iterate over the terms in the coefficient
        # sp.Add.make_args handles both single terms and sums securely
        for term in sp.Add.make_args(coeff):
            # Split the term into its multiplied factors
            factors = sp.Mul.make_args(term)
            
            # Separate SUNDelta factors from constants and amplitudes
            delta_factors = [f for f in factors if f.func == delta_func]
            other_factors = [f for f in factors if f.func != delta_func]
            
            delta_chain = sp.Mul(*delta_factors)
            rest = sp.Mul(*other_factors)
            
            # Group by the exact delta chain
            delta_dict[delta_chain] += rest
            
        # 4. Reconstruct the cleanly factored coefficient
        grouped_coeff = 0
        for d_chain, rest_sum in delta_dict.items():
            grouped_coeff += d_chain * rest_sum
            
        final_expr += nc_pow * grouped_coeff
        
    return final_expr