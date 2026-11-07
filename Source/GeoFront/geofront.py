import sys; sys.path.insert(0,'/tmp/kg')
from kgc import *
from PIL import Image
R=[13824,5632,14336,6144]; X0,Y0=R[0],R[1]; W,H=16384,8192
vh=Region('tk','Earth_VertexHeight',R).levels[0].astype(float); G=-10919+vh/65535*19519.46387
VC=Region('tk','Earth_VertexColor',R).levels[0][...,:3].astype(float)
KLAT,KLON=35.25,139.03
MINLAT,MAXLAT=35.182,35.318; MINLON,MAXLON=138.880,139.097
N=512
def bspl(t):
    return np.stack([(1-t)**3/6,(3*t**3-6*t**2+4)/6,(-3*t**3+3*t**2+3*t+1)/6,t**3/6])
def sample(img,lat,lon):
    x=((0.25-lon/360)%1)*W-X0; y=(lat+90)/180*H-Y0
    x0=np.floor(x).astype(int); y0=np.floor(y).astype(int); wx=bspl(x-x0); wy=bspl(y-y0)
    out=0
    for j in range(4):
        for i in range(4):
            v=img[np.clip(y0-1+j,0,511),np.clip(x0-1+i,0,511)]
            w=(wy[j]*wx[i])
            out=out+(v*(w[...,None] if v.ndim==3 else w))
    return out
jj,ii=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
lat=MAXLAT-jj/N*(MAXLAT-MINLAT); lon=MINLON+ii/N*(MAXLON-MINLON)
g=sample(G,lat,lon); gc=sample(VC,lat,lon)
xk=(lon-KLON)*111.32*np.cos(np.radians(lat)); yk=(lat-KLAT)*111.0       # km from the KSC, x east / y north
def ss(a,b,x): t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
CX,CY=-6.2,0.3                         # GeoFront centre (km from KSC)
RF,RW=2.60,3.00                        # floor radius / wall-top radius (km)
rc=np.hypot(xk-CX,yk-CY); rk=np.hypot(xk,yk)
PL=650.0; FLOOR=PL-3*100              # 350 m, exact in the 3 m quantisation below
# mesa: flat 650 around the KSC and the crater, blending back into the real terrain
mesa=np.maximum(1-ss(2.8,5.0,rk), 1-ss(RW+0.6,RW+3.0,rc))
h=g*(1-mesa)+PL*mesa
# crater: steep wall with a slight terraced look, flat floor
wallt=ss(RF,RW,rc)
crater=FLOOR+(PL-FLOOR)*wallt
h=np.where(rc<RW, np.minimum(h,crater), h)
# pit under the inverted-pyramid lake (model sits on top). Pool centre in crater-local metres:
PX,PY=-0.30,0.0; PIT=0.34                     # 340 m half-size: water square 300 m + 40 m
px=(xk-CX)-PX; py=(yk-CY)-PY
inpit=(np.abs(px)<PIT)&(np.abs(py)<PIT)
h=np.where(inpit, FLOOR-300, h)          # lake bottom is at FLOOR+2-271
h=np.maximum(h,-58)                    # the deep sea floor is clamped (hidden under the LCL anyway)
S=3.0; OFF=PL-S*237                    # -61 m .. 704 m in 3 m steps, 650 and 350 exact
print('tile range', h.min(), h.max(), 'OFF',OFF,'max',OFF+255*S)
k=np.clip(np.round((h-OFF)/S),1,255).astype(np.uint8)
hq=OFF+S*k.astype(float)
print('KSC h', hq[np.unravel_index(np.argmin(rk),rk.shape)], 'floor', hq[np.unravel_index(np.argmin(rc),rc.shape)])
# colours: forest floor, grey armoured walls, concrete rim road; everything else = Sol's own vertex colour
forest=np.array([38,62,30.]); wall=np.array([92,92,96.]); rim=np.array([150,148,142.])
rng=np.random.default_rng(3); noise=np.array(Image.fromarray((rng.random((64,64))*255).astype(np.uint8)).resize((N,N),Image.BICUBIC))/255.
c=gc.copy()
fl=(1-ss(RF-0.05,RF+0.05,rc))[...,None]; wl=(ss(RF-0.05,RF+0.05,rc)*(1-ss(RW-0.02,RW+0.05,rc)))[...,None]
rimw=(np.exp(-((rc-RW-0.08)/0.07)**2))[...,None]
c=c*(1-fl)+forest*(0.8+0.4*noise[...,None])*fl
c=c*(1-wl)+wall*(0.85+0.3*(np.sin(rc*80)>0)[...,None])*wl     # horizontal armour bands
c=c*(1-rimw)+rim*rimw
c=np.clip(np.round(c),1,255).astype(np.uint8)
import os; os.makedirs('/tmp/kg/geofront',exist_ok=True)
Image.fromarray(k).convert('RGB').save('/tmp/kg/geofront/GeoFront_Height.png')
Image.fromarray(c).save('/tmp/kg/geofront/GeoFront_Color.png')
json.dump(dict(offset=OFF,deformity=S*255,minLat=MINLAT,maxLat=MAXLAT,minLon=MINLON,maxLon=MAXLON),open('/tmp/kg/geofront/tile.json','w'))
# where to put the statics (lat/lon of crater centre, pool, pyramid, tower) -> metres east/north of crater centre
def ll(xe,yn):  # km from KSC -> lat/lon
    la=KLAT+yn/111.0; lo=KLON+xe/(111.32*np.cos(np.radians(la))); return la,lo
pos={'pool':(CX+PX,CY+PY),'pyramid':(CX+PX+0.4758,CY+PY+0.4758),'tower':(CX+0.95,CY+0.45),'centre':(CX,CY)}
for kk,(xe,yn) in pos.items(): print(kk, ['%.5f'%v for v in ll(xe,yn)])
print('floor',FLOOR)
# preview
pv=np.clip((hq-0)/700*255,0,255).astype(np.uint8)
Image.fromarray(np.hstack([np.dstack([pv]*3),c])).save('/tmp/kg/geofront_prev.png')
# seam check: tile vs global along edges
e=np.concatenate([hq[0]-g[0],hq[-1]-g[-1],hq[:,0]-g[:,0],hq[:,-1]-g[:,-1]]); gg=np.concatenate([g[0],g[-1],g[:,0],g[:,-1]])
print('edge diff (land only)', np.abs(e[gg>-58]).max())
