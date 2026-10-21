"""Apply checkpoint-confirmed reviewer-item-2 correction to existing documents."""
from pathlib import Path
import json,csv
from docx import Document
from docx.shared import RGBColor
from lxml import etree
B=Path(__file__).resolve().parents[1]
report=json.loads((B/'核验数据/paper_parameter_check.json').read_text())
assert report['results'][0]['kernel']==5
changes=[]
repls={
'给定平滑尺度σ，取奇数核长K=2⌈2σ⌉+1，令M=(K−1)/2。':'本方法慢、快流分解及低频路径平滑均采用固定奇数核长K=5，平滑尺度σ分别为1.8和0.8；令M=(K−1)/2。',
'保留σ=1.8/0.8、K=2⌈2σ⌉+1、原始帧率30 Hz和慢/快流4倍/2倍时间压缩':'根据原实验检查点核对，慢、快流分解及低频路径平滑的核长均为K=5，保留σ=1.8/0.8、原始帧率30 Hz和慢/快流4倍/2倍时间压缩',
'本次按现稿的实验结果及方法参数进行核验':'本次保持原实验结果不变，并依据作者确认用于论文评测的Epoch4检查点核验固定核参数',
'慢流σ=1.8、K=9':'慢流σ=1.8、K=5',
'慢流K=9':'慢流K=5',
'K=9 / 7.5 Hz':'K=5 / 7.5 Hz',
'0.799':'1.048','0.571':'0.765',
'1.26×10⁻⁶':'4.99×10⁻⁷','1.26e-06':'4.99e-07',
'验证对象为按稿件参数显式配置的高斯滤波算子及纯线性串联':'验证对象为使用原实验Epoch4检查点中所存核系数的高斯滤波算子及纯线性串联',
'核验采用空间恒定的正弦输入':'核验直接使用原实验Epoch4检查点保存的固定核系数，采用空间恒定的正弦输入',
'另以空间恒定的合成正弦输入进行算子级数值核验':'另直接使用Epoch4检查点保存的固定核系数，以空间恒定的合成正弦输入进行算子级数值核验',
'图2　按稿件参数计算的高斯核离散幅频响应与纯滤波串联参考':'图2　按原实验核参数计算的高斯核离散幅频响应与纯滤波串联参考',
'半高全宽不能直接替代截止频率。':'半高全宽不能直接替代截止频率，也不表示5点有限离散核的实测半高全宽。',
}

def replace(p,old,new):
 # Preserve formatting and any math/object runs outside the edited text span.
 while old in p.text:
  text=''.join(r.text for r in p.runs)
  start=text.find(old)
  if start<0: raise ValueError('Unsupported paragraph content')
  end=start+len(old);offset=0;first=True
  for r in p.runs:
   t=r.text;l=offset;h=offset+len(t);offset=h
   if h<=start or l>=end:continue
   left=t[:max(0,start-l)];right=t[max(0,end-l):]
   r.text=left+(new if first else '')+right
   r.font.color.rgb=RGBColor(255,0,0);first=False

def paras(d):
 yield from d.paragraphs
 for table in d.tables:
  for row in table.rows:
   for cell in row.cells:yield from cell.paragraphs

for name in ['频率分离双流网络的非接触式心率检测_修订稿.docx','修改说明.docx','专项材料/第2条_替换章节_标红.docx']:
 path=B/name;d=Document(path)
 tables=[etree.tostring(t._tbl) for t in d.tables]
 for p in paras(d):
  before=p.text
  for old,new in repls.items():replace(p,old,new)
  if before!=p.text:changes.append({'file':name,'before':before,'after':p.text})
 for rel in d.part.rels.values():
  if 'image' in rel.reltype and rel.target_ref.endswith('.png'):
   # In these documents the only PNG is the newly added frequency figure.
   rel.target_part._blob=(B/'图表/图2_按现稿参数的离散频率响应.png').read_bytes()
 if name.startswith('频率'):
  assert tables==[etree.tostring(t._tbl) for t in d.tables]
 d.save(path)
if changes: (B/'核验数据/Epoch4修订对照.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
with (B/'核验数据/sinusoidal_validation.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['分支','核长','sigma','特征采样率_Hz','纯滤波次数','频率_Hz','数值输出幅值','离散计算幅值','绝对差'])
 for r in report['results']:
  for c in r['sinusoidal_checks']:
   for freq,measured,analytic in zip(r['test_frequencies_Hz'],c['measured_gains'],c['analytic_gains']):
    w.writerow([r['stream'],r['kernel'],r['sigma'],r['feature_fs_Hz'],c['isolated_filter_repetitions'],freq,measured,analytic,abs(measured-analytic)])
print('Updated paragraphs:',len(changes))
