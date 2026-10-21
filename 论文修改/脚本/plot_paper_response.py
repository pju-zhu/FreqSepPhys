import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
base=Path(__file__).resolve().parents[1]
out=base/'图表'
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
r=json.loads((base/'核验数据'/'paper_parameter_check.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(10.2,7.3),layout='constrained')
curves=[]
for col,x in enumerate(r['results']):
    fs=x['feature_fs_Hz'];K=x['kernel'];M=K//2
    f=np.linspace(0,fs/2,2001);n=np.arange(-M,M+1);g=np.array(x['weights'])
    H=np.cos(2*np.pi*f[:,None]*n[None,:]/fs)@g
    ax=axs[0,col]
    ax.plot(f,np.abs(H),'k-',label='单层低通（离散计算）')
    ax.plot(f,np.abs(1-H),'k--',label='单层残差（离散计算）')
    obs=x['sinusoidal_checks'][0]
    ax.plot(x['test_frequencies_Hz'],obs['measured_gains'],'o',ms=3,mfc='white',mec='black',label='滤波算子正弦核验')
    for value,marker,label in [(x['half_amplitude_Hz'],'s','幅值50%'),(x['half_power_Hz'],'^','功率50%')]:
        gain=.5 if marker=='s' else np.sqrt(.5)
        ax.plot(value,gain,marker,color='black',ms=4)
        ax.annotate(f'{label}: {value:.3f} Hz',(value,gain),xytext=(9,-14 if marker=='s' else 10),textcoords='offset points',fontsize=8)
    ax.set(title=f'({chr(97+col)}) {x["stream"]}：K={K}，σ={x["sigma"]}，采样率{fs:g} Hz',xlabel='时间频率 / Hz',ylabel='幅值比',ylim=(0,1.13),xlim=(0,fs/2))
    ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.18)
    ax=axs[1,col]
    for m,style in [(1,'-'),(2,'--'),(3,'-.'),(6,':')]:
        ax.plot(f,np.abs(H)**m,color='black',ls=style,label=f'纯线性串联 m={m}')
        if m<=3:
            obs=x['sinusoidal_checks'][m-1]
            ax.plot(x['test_frequencies_Hz'],obs['measured_gains'],'o',ms=2,mfc='white',mec='black')
    ax.set(title=f'({chr(99+col)}) {x["stream"]}纯滤波重复作用参考',xlabel='时间频率 / Hz',ylabel='幅值比',ylim=(0,1.05),xlim=(0,fs/2))
    ax.legend(fontsize=8);ax.grid(alpha=.18)
    for i,fi in enumerate(f):curves.append([x['stream'],K,x['sigma'],fs,float(fi),float(abs(H[i])),float(abs(1-H[i])),float(abs(H[i])**2),float(abs(H[i])**3),float(abs(H[i])**6)])
fig.savefig(out/'图2_按现稿参数的离散频率响应.png',dpi=260)
fig.savefig(out/'图2_按现稿参数的离散频率响应.svg')
svg=out/'图2_按现稿参数的离散频率响应.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
fig.savefig(out/'图2_按现稿参数的离散频率响应.pdf')
with (base/'核验数据'/'frequency_curves.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f,lineterminator="\n");w.writerow(['分支','核长','sigma','特征采样率_Hz','频率_Hz','低通幅值','残差幅值','串联2次幅值','串联3次幅值','串联6次幅值']);w.writerows(curves)
print('Generated figure and',len(curves),'CSV rows')
