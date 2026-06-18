import subprocess, time, os, numpy as np
WD = "C:/ngspice_work"
NG = "ngspice"

# Testbench templates: static 2-terminal neural device driven by V1.
TBS = {
 "DC read":      "dc V1 -1 1 0.001",
 "Pulse trans.": "tran 0.02u 4m uic",
 "Sinusoid":     "tran 0.02u 4m uic",
}
SRC = {
 "DC read":      "V1 a 0 0",
 "Pulse trans.": "V1 a 0 PULSE(-1 1 0 1u 1u 20u 50u)",
 "Sinusoid":     "V1 a 0 SIN(0 1 1k)",
}
MODELS = {"dense":"mem_dense", "struct":"mem_struct"}

def cir(model, tb):
    out=f"{WD}/ab_{model}_{tb.split()[0]}.txt".replace(' ','')
    return f"""* AB {model} {tb}
{SRC[tb]}
N1 a 0 dut
.model dut {MODELS[model]}()
.control
pre_osdi {WD}/{MODELS[model]}.osdi
{TBS[tb]}
wrdata {out} i(v1)
quit
.endc
.end
""", out

def run(model, tb, reps=5):
    code, out = cir(model, tb)
    cf=f"{WD}/ab_{model}_{tb.split()[0]}.cir".replace(' ','')
    open(cf,"w").write(code)
    best=1e9; ok=True; pts=0
    for _ in range(reps):
        if os.path.exists(out): os.remove(out)
        t=time.perf_counter()
        r=subprocess.run([NG,"-b",cf],capture_output=True,text=True,timeout=120)
        dt=time.perf_counter()-t
        log=(r.stdout+r.stderr).lower()
        conv = (r.returncode==0) and not any(k in log for k in
                ["timestep too small","aborted","singular","no convergence","fatal"])
        ok = ok and conv
        if os.path.exists(out):
            pts=sum(1 for _ in open(out))
        best=min(best,dt)
    return best, pts, ok, out

rows=[]; data={}
for tb in TBS:
    for model in MODELS:
        wt,pts,ok,out=run(model,tb)
        arr=np.loadtxt(out) if os.path.exists(out) else None
        data[(tb,model)]=arr
        rows.append([model,tb,"yes",("yes" if ok else "NO"),f"{wt*1000:.1f}",pts])

# output error: structured vs dense per testbench (col 1 = i)
print(f"{'model':8} {'testbench':13} {'compiled':9} {'converged':10} {'wall_ms':8} {'points':7} {'err_vs_dense':12}")
for r in rows:
    model,tb=r[0],r[1]
    err=""
    if model=="struct":
        d=data.get((tb,"dense")); s=data.get((tb,"struct"))
        if d is not None and s is not None and d.shape==s.shape:
            iD,iS=d[:,1],s[:,1]
            err=f"{np.sqrt(np.mean((iS-iD)**2))/(np.max(np.abs(iD))+1e-30):.2%}"
    print(f"{r[0]:8} {r[1]:13} {r[2]:9} {r[3]:10} {r[4]:8} {str(r[5]):7} {err:12}")
