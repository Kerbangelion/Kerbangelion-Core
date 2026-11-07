import numpy as np
from PIL import Image, ImageFilter
O='/tmp/kg/models/'; rng=np.random.default_rng(21)
def save(a,name): Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(O+name)
# ---- pyramid face (triangle (0,0)(1,0)(0.5,1); v=0 is the lake bottom, deck level at v=0.424) ----
N=1024; v,u=np.meshgrid(np.linspace(1,0,N),np.linspace(0,1,N),indexing='ij')
G=271/571.; half=(1-v)/2; s=(u-0.5)/np.maximum(half,1e-4)
base=np.zeros((N,N,4)); base[...]=[11,15,34,255]
gv=np.array(Image.fromarray((rng.random((64,64))*255).astype(np.uint8)).resize((N,N),Image.NEAREST))/255.
base[...,:3]*=(0.85+0.3*gv[...,None])
seam=(np.abs(((v*52)%1)-0.5)>0.47)|(np.abs(((s*14+14)%1)-0.5)>0.47)
hseam=(np.abs(((v*52)%1)-0.5)>0.47)
base[...,:3]=np.where(hseam[...,None],[18,23,42],base[...,:3]); base[...,:3]*=0.8   # faint horizontal seams only
save(base,'Pyramid_Face.png')
em=np.zeros((N,N,3))
band=np.exp(-((v-(G+0.63*(1-G)))/0.004)**2)      # the bright seam near the top
em+=np.array([150,210,255])*band[...,None]                  # no glowing panel grid
em=np.where((np.abs(s)<=1)[...,None],em,0)
# point lights: 4 on each slanted edge (one where the seam meets the edge) + one at the apex
FW,FH=2*571/1.2,571.0                            # face width at the base / height (m), for round dots
for t in (0.10,0.36,0.63,0.86,1.0):
    vv=G+t*(1-G)
    for side in ((-1,1) if t<1 else (0,)):
        uc=0.5+side*(1-vv)/2
        d=np.sqrt(((u-uc)*FW)**2+((v-vv)*FH)**2)
        core=np.clip(1-(d/6.5)**2,0,1); halo=np.exp(-(d/16.0)**2)*0.6
        em+=np.array([235,245,255])*core[...,None]+np.array([140,190,255])*halo[...,None]
save(em,'Pyramid_Face_Glow.png')
# ---- solar collector: concave cell face (u across the arc, v up) ----
W,H=512,2048; vv,uu=np.meshgrid(np.linspace(1,0,H),np.linspace(0,1,W),indexing='ij')
c=np.zeros((H,W,4)); c[...,3]=255; c[...,:3]=[34,42,62]
cell=(np.abs(((uu*40)%1)-0.5)>0.45)|(np.abs(((vv*300)%1)-0.5)>0.45)
c[...,:3]=np.where(cell[...,None],[70,80,98],c[...,:3])
c[...,:3]*=(0.85+0.3*(1-uu[...,None]*0+0.0))
diag=np.zeros((H,W))
for k in (0.22,0.5,0.78):                         # three bright diagonal bands like the reference collector
    diag=np.maximum(diag,np.exp(-((vv-(k+0.10*(uu-0.5)))/0.006)**2))
c[...,:3]=c[...,:3]*(1-diag[...,None])+np.array([200,205,212])*diag[...,None]
c[...,:3]*= (0.75+0.35*np.abs(uu[...,None]-0.5)*2)  # sheen towards the edges of the curve
save(c,'Solar_Cells.png')
g=np.zeros((H,W,3)); g+=np.array([30,24,10])*(1-cell[...,None])*0.0+np.array([60,60,70])*diag[...,None]
save(g,'Solar_Cells_Glow.png')
# ---- white back panels (tiling) and frame ----
M=512; yy,xx=np.mgrid[0:M,0:M]
b=np.zeros((M,M,4)); b[...,3]=160; b[...,:3]=[228,230,233]
b[...,:3]=np.where((((xx%128)<2)|((yy%64)<2))[...,None],[190,192,198],b[...,:3]); save(b,'Solar_Back.png')
f=np.zeros((M,M,4)); f[...,3]=180; f[...,:3]=[238,238,240]
f[...,:3]=np.where(((yy%128)<3)[...,None],[205,205,210],f[...,:3]); save(f,'Solar_Frame.png')
fg=np.zeros((M,M,3)); dots=((yy%128)>60)&((yy%128)<68)&((xx%256)>120)&((xx%256)<136)
fg[dots]=[255,170,70]; save(fg,'Solar_Frame_Glow.png')
print('ok')
