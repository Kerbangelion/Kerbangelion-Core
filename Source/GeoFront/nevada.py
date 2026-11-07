import sys; sys.path.insert(0,'/tmp/kg')
from kgc import *
from PIL import Image
R=[9216,5632,9728,6144]; X0,Y0=R[0],R[1]
LAT,LON=37.24,-115.81; RK=93.0
W,H=16384,8192
rows=np.arange(Y0,R[3])+0.5; cols=np.arange(X0,R[2])+0.5
lat=rows/H*180-90; lon=-(((cols/W)-0.25)*360)
lon=(lon+180)%360-180
LA,LO=np.meshgrid(lat,lon,indexing='ij')
yk=(LA-LAT)*111.2; xk=(LO-LON)*111.2*np.cos(np.radians(LA))
rk=np.hypot(xk,yk); ang=np.arctan2(yk,xk)
rng=np.random.default_rng(7)
def smoothstep(a,b,x): t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
# angular streak noise
def angnoise(freqs):
    n=np.zeros_like(ang)
    for f in freqs:
        ph=rng.uniform(0,2*np.pi,8); 
        n+=sum(np.sin(f*(k+1)*ang+ph[k])/(k+1) for k in range(3))/len(freqs)
    return n
streak=angnoise([7,13,29,53])
# ---------- heights ----------
vh=Region('nv','Earth_VertexHeight',R); v=vh.levels[0].astype(float); m_old=-10919+v/65535*19519.46387
ring=(rk>RK-3)&(rk<RK+3); hrim=np.median(m_old[ring]); print('rim median',hrim, 'center old', m_old[159,150])
FLOOR=200.0
wall=smoothstep(0.80*RK,RK,rk)                 # 0 on floor -> 1 at rim
z=FLOOR+(hrim+60-FLOOR)*wall                   # bowl surface
rim=140*np.exp(-((rk-RK)/9.0)**2)*(1+0.25*streak)
m_new=np.where(rk<RK, np.minimum(m_old+rim, z), m_old+rim*(rk<RK+40))
m_new=np.where(rk<RK*0.80, FLOOR, m_new)                     # perfectly flat floor
vh.levels[0]=np.clip(np.round((m_new+10919)/19519.46387*65535),0,65535).astype(np.uint16)
inside=rk<RK
print('changed px',(vh.levels[0]!=vh.orig[0]).sum())
# ---------- scaled height (BC4) ----------
hs=Region('nv','Earth_Height',R); a=hs.levels[0].astype(float)
a[...,0]=np.clip(a[...,0]+0.0130742*(m_new-m_old),0,255); a[...,1]=a[...,0]; a[...,2]=a[...,0]
hs.levels[0]=np.round(a).astype(np.uint8); mip_deltas(hs)
# ---------- normals (BC5) ----------
nm=Region('nv','Earth_Normal',R); a=nm.levels[0].astype(float)
gx_o=np.gradient(m_old,axis=1); gy_o=np.gradient(m_old,axis=0); gx_n=np.gradient(m_new,axis=1); gy_n=np.gradient(m_new,axis=0)
cx_,cy_=-0.0565529,-0.0247207
w_in=1-smoothstep(0.96*RK,1.0*RK,rk)
for ch,c,go,gn in ((0,cx_,gx_o,gx_n),(1,cy_,gy_o,gy_n)):
    keep=a[...,ch]+c*(gn-go); flat=127+c*gn
    a[...,ch]=(1-w_in)*keep+w_in*flat
nm.levels[0]=np.clip(np.round(a),0,255).astype(np.uint8); mip_deltas(nm)
# ---------- colours ----------
def crater_rgb(base):
    """base: HxWx3 float original colour -> new colour"""
    floorc=np.zeros(base.shape)                    # pitch black
    wallc=np.array([46,42,44.])[None,None,:]*(0.8+0.2*(1+streak[...,None])/2)
    ash=np.array([150,143,135.])[None,None,:]*(0.85+0.15*streak[...,None])
    t_wall=smoothstep(0.80*RK,0.99*RK,rk)[...,None]
    inner=floorc*(1-t_wall)+wallc*t_wall
    ejw=(np.exp(-((rk-RK)/14.0)**2)*0.85+0.45*np.clip(streak,0,None)*np.exp(-((rk-RK)/22.0)**2))[...,None]
    ejw=np.clip(ejw,0,0.85)*(rk>=RK*0.985)[...,None]
    outer=base*(1-ejw)+ash*ejw
    t_in=(1-smoothstep(0.985*RK,1.0*RK,rk))[...,None]
    return inner*t_in+outer*(1-t_in)
for name in ('Earth_Color','Earth_VertexColor'):
    rg=Region('nv',name,R); a=rg.levels[0].astype(float)
    a[...,:3]=crater_rgb(a[...,:3])
    rg.levels[0]=np.clip(np.round(a),0,255).astype(np.uint8); mip_deltas(rg)
    globals()['reg_'+name]=rg
# scatter (scaled scattering tint): darken inside
sc=Region('nv','Earth_Scatter',R); a=sc.levels[0].astype(float)
f=(smoothstep(0.8*RK,1.02*RK,rk))[...,None]; a[...,:3]*=f
sc.levels[0]=np.clip(np.round(a),0,255).astype(np.uint8); mip_deltas(sc)
# city lights: nothing survives inside
cl=Region('nv','EarthCities',R); a=cl.levels[0].astype(float)
k=smoothstep(0.98*RK,1.05*RK,rk)[...,None]; a[...,:3]*=k
cl.levels[0]=np.clip(np.round(a),0,255).astype(np.uint8); mip_deltas(cl)
# biomes
bm=Region('nv','Earth_Biomes',R); b=bm.levels[0].copy(); b[rk<RK]=12; bm.levels[0]=b
import os; os.makedirs('/tmp/kg/patch',exist_ok=True)
for rg in (vh,hs,nm,reg_Earth_Color,reg_Earth_VertexColor,sc,cl,bm):
    n=rg.patch('/tmp/kg/patch/nv_'+rg.name+'.patch'); print(rg.name,'records',n)
im=reg_Earth_Color.levels[0][...,:3][::-1,::-1]; Image.fromarray(np.ascontiguousarray(im)).resize((768,768)).save('/tmp/kg/nv_after.png')
hh=m_new[::-1,::-1]; Image.fromarray(np.clip((hh-0)/3500*255,0,255).astype(np.uint8)).resize((512,512)).save('/tmp/kg/nv_h.png')
np.save('/tmp/kg/nv_profile.npy', np.vstack([rk[159], m_old[159], m_new[159]]))
np.save('/tmp/kg/nv_vh_new.npy', vh.levels[0])
