import json, math
from pathlib import Path
import cv2
import numpy as np
root=Path(__file__).resolve().parent.parent
official=root/"videos/official-npu-sp8.mp4"
omni=root/"videos/omni-bf16-sp8.mp4"
x,y=(cv2.VideoCapture(str(path)) for path in (official,omni))
for cap in (x,y):
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==269
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==832 and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==464
    assert abs(cap.get(cv2.CAP_PROP_FPS)-16)<1e-6
def ssim(a,b):
    def blur(v):
        return cv2.GaussianBlur(v,(11,11),1.5)[5:-5,5:-5]
    ma,mb=blur(a),blur(b)
    va,vb=blur(a*a)-ma*ma,blur(b*b)-mb*mb
    cov=blur(a*b)-ma*mb
    c1,c2=(.01*255)**2,(.03*255)**2
    return float((((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))).mean())
rows=[]
for _ in range(269):
    ok,a=x.read()
    ok2,b=y.read()
    assert ok and ok2
    a,b=a.astype(np.float32),b.astype(np.float32)
    rows.append((float(np.square(a-b).mean(dtype=np.float64)),ssim(a,b),float(np.abs(a-b).mean())))
def summarize(rows):
    mse,ssim,mae=map(float,np.mean(rows,axis=0))
    return dict(frames=len(rows),psnr_db=10*math.log10(255**2/mse) if mse else None,ssim=ssim,mae=mae)
blocks=[rows[:13]]+[rows[start:start+16] for start in range(13,269,16)]
results=dict(official=str(official),omni=str(omni),all=summarize(rows),chunks=[summarize(p) for p in blocks])
(root/"evidence/comparison.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
