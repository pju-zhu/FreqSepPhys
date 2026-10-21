import ast, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import brentq

import argparse
parser=argparse.ArgumentParser(description='Validate isolated Gaussian filters with manuscript-defined parameters.')
parser.add_argument('--model-source',type=Path,default=Path(__file__).resolve().parents[2]/'neural_methods/model/FreqSepPhys.py')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/'核验数据'
source=args.model_source.read_text()
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GaussianLowPass3d')
scope={'torch':torch,'nn':nn,'F':F}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<GaussianLowPass3d from source>','exec'),scope)
Gaussian=scope['GaussianLowPass3d']
torch.set_num_threads(2)
results=[]
for name,sigma,fs in [('慢流',1.8,7.5),('快流',0.8,15.0)]:
    K=2*math.ceil(2*sigma)+1
    M=K//2
    n=np.arange(-M,M+1)
    g=np.exp(-n*n/(2*sigma*sigma));g/=g.sum()
    def H(f):return float(np.dot(g,np.cos(2*np.pi*f*n/fs)))
    half=brentq(lambda f:H(f)-.5,0,fs/2)
    halfpower=brentq(lambda f:H(f)-2**-.5,0,fs/2)
    freqs=np.array([.1,.25,.5,.52,.75,1.,1.17,1.5,2.,2.5,3.,min(3.5,fs*.49)])
    module=Gaussian(1,kernel_size=K,sigma=sigma)
    # Pure isolated filters, with paper-defined arguments; not a trained-network test.
    depth=3
    spatial=2*depth*M+1
    ts=np.arange(128)/fs
    wave=np.sin(2*np.pi*freqs[:,None]*ts[None,:])
    y=torch.tensor(wave[:,None,:,None,None],dtype=torch.float32).expand(-1,1,-1,spatial,spatial).contiguous()
    errors=[]
    for m in range(1,depth+1):
        y=module(y)
        margin=m*M
        center=y[:,0,margin:-margin,spatial//2,spatial//2].detach().numpy()
        gains=[]
        for i,f in enumerate(freqs):
            t=ts[margin:-margin]
            basis=np.stack([np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t),np.ones_like(t)],axis=1)
            coeff=np.linalg.lstsq(basis,center[i],rcond=None)[0]
            gains.append(float(np.hypot(coeff[0],coeff[1])))
        expected=np.array([abs(H(f))**m for f in freqs])
        err=float(np.max(np.abs(np.array(gains)-expected)))
        assert err<5e-6,(name,m,err)
        errors.append({'isolated_filter_repetitions':m,'max_absolute_amplitude_error':err,'measured_gains':gains,'analytic_gains':expected.tolist()})
    results.append({'stream':name,'sigma':sigma,'kernel':K,'feature_fs_Hz':fs,'weights':g.tolist(),'half_amplitude_Hz':half,'half_power_Hz':halfpower,'untruncated_FWHM_feature_samples':2*np.sqrt(2*np.log(2))*sigma,'untruncated_FWHM_ms':2*np.sqrt(2*np.log(2))*sigma/fs*1000,'amplitude_at_0.5Hz':abs(H(.5)),'amplitude_at_3Hz':abs(H(3)),'test_frequencies_Hz':freqs.tolist(),'sinusoidal_checks':errors})
report={'basis':'Author confirmed that current repository code was modified after the paper; manuscript parameters and experimental results are authoritative. K=2ceil(2sigma)+1, slow K=9, fast K=5. Numerical validation concerns explicitly configured filters, not historical training reproduction.','validation':'Isolated original GaussianLowPass3d class with explicitly supplied manuscript parameters; CPU float32, spatially constant sinusoids, temporal/spatial boundaries excluded; repetitions1,2,3 are pure linear cascades, not FreqSepBlock cascades.','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'results':results}
(root/'paper_parameter_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for x in results:
 print(x['stream'],'K=',x['kernel'],'half_amplitude=',x['half_amplitude_Hz'],'half_power=',x['half_power_Hz'],'FWHM_ms=',x['untruncated_FWHM_ms'],'max_error=',max(e['max_absolute_amplitude_error'] for e in x['sinusoidal_checks']))
