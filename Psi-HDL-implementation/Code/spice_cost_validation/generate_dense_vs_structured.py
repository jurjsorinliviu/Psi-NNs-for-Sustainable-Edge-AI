# Matched DENSE vs STRUCTURED Verilog-A pair, both TRAINED to the same memristor
# read-characteristic I(V) (clean matched comparison, controlled error):
#   dense      : over-parameterized MLP [1-16-16-1] (288 weight terms)
#   structured : compact discovered-scale MLP [1-4-4-1] (24 weight terms)
# Both reproduce the same behavior within a small tolerance; the structured export
# has ~12x fewer explicit operations -> tests simulator-side cost of structure.
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

V = torch.linspace(-1, 1, 400).reshape(-1, 1)
I_tgt = 1e-3*torch.tanh(V/0.25) + 2e-4*V
yscale = float(I_tgt.abs().max())
xv = np.linspace(-1,1,400)
ref = (1e-3*np.tanh(xv/0.25) + 2e-4*xv)          # analytic reference current
denom = np.max(np.abs(ref))

class MLP(nn.Module):
    def __init__(s, h):
        super().__init__(); s.l1=nn.Linear(1,h); s.l2=nn.Linear(h,h); s.l3=nn.Linear(h,1)
    def forward(s,x): return s.l3(torch.tanh(s.l2(torch.tanh(s.l1(x)))))

def train(h, iters=3000, lr=1e-2):
    torch.manual_seed(0)
    m=MLP(h); opt=torch.optim.Adam(m.parameters(),lr=lr)
    for e in range(iters):
        opt.zero_grad(); loss=((m(V)-I_tgt/yscale)**2).mean(); loss.backward(); opt.step()
    W1=m.l1.weight.detach().numpy().reshape(-1).copy(); b1=m.l1.bias.detach().numpy().copy()
    W2=m.l2.weight.detach().numpy().copy();             b2=m.l2.bias.detach().numpy().copy()
    W3=m.l3.weight.detach().numpy().reshape(-1).copy(); b3=float(m.l3.bias.detach().numpy()[0])
    return h,W1,b1,W2,b2,W3,b3

def fwd(h,W1,b1,W2,b2,W3,b3,x):
    a=np.tanh(W1.reshape(-1,1)*x + b1.reshape(-1,1)); a=np.tanh(W2@a + b2.reshape(-1,1))
    return yscale*(W3.reshape(1,-1)@a + b3).reshape(-1)

def terms(W1,W2,W3): return int(W1.size+W2.size+W3.size)

def num(v): return f"({v:.10e})"
def emit(modname,h,W1,b1,W2,b2,W3,b3):
    decl=", ".join([f"h1_{j}" for j in range(h)]+[f"h2_{k}" for k in range(h)])
    L=["// Auto-generated neural memristor read-characteristic compact model",
       f"// {modname}: explicit forward pass, OpenVAF-compatible scalar literals",
       '`include "constants.vams"','`include "disciplines.vams"','',
       f"module {modname}(p, n);","    inout p, n;","    electrical p, n;",
       f"    real Vin, {decl}, Iout;","    analog begin","        Vin = V(p, n);"]
    for j in range(h): L.append(f"        h1_{j} = tanh({num(W1[j])}*Vin + {num(b1[j])});")
    for k in range(h):
        t=[f"{num(W2[k,j])}*h1_{j}" for j in range(h)]+[num(b2[k])]
        L.append(f"        h2_{k} = tanh({' + '.join(t)});")
    ot=[f"{num(W3[k])}*h2_{k}" for k in range(h)]+[num(b3)]
    L+= [f"        Iout = ({yscale:.10e})*({' + '.join(ot)});","        I(p, n) <+ Iout;","    end","endmodule",""]
    return "\n".join(L)

dense  = train(16)
struct = train(4)
Id = fwd(*dense, xv); Is = fwd(*struct, xv)
ed = float(np.sqrt(np.mean((Id-ref)**2))/denom)
es = float(np.sqrt(np.mean((Is-ref)**2))/denom)
eds= float(np.sqrt(np.mean((Is-Id)**2))/(np.max(np.abs(Id))+1e-30))
print(f"dense [1-16-16-1] terms={terms(dense[1],dense[3],dense[5])} err-vs-ref={ed:.2%}")
print(f"struct[1-4-4-1]   terms={terms(struct[1],struct[3],struct[5])} err-vs-ref={es:.2%}")
print(f"structured-vs-dense rel RMS={eds:.2%}")
open("mem_dense.va","w").write(emit("mem_dense",*dense))
open("mem_struct.va","w").write(emit("mem_struct",*struct))
print("wrote mem_dense.va, mem_struct.va")
