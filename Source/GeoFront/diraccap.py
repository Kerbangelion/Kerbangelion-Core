# Black "Dirac Sea" cap: a spherical-cap mesh 5 m above the crater floor, draped over the foot of the walls.
import sys, math, numpy as np
sys.path.insert(0,'/tmp/kg'); sys.path.insert(0,'/tmp/mu_src/io_object_mu')
from modeler import Model, specular, COMMON, CF
RE=6371010.0; LAT0,LON0=37.24,-115.81; X0,Y0=9216,5632; W,H=16384,8192
v=np.load('/tmp/kg/nv_vh_new.npy').astype(float); G=-10919+v/65535*19519.46387
def bspl(t): return np.stack([(1-t)**3/6,(3*t**3-6*t**2+4)/6,(-3*t**3+3*t**2+3*t+1)/6,t**3/6])
def terrain(lat,lon):
    x=((0.25-lon/360)%1)*W-X0; y=(lat+90)/180*H-Y0
    x0=np.floor(x).astype(int); y0=np.floor(y).astype(int); wx=bspl(x-x0); wy=bspl(y-y0); out=0
    for j in range(4):
        for i in range(4): out=out+G[y0-1+j,x0-1+i]*wy[j]*wx[i]
    return out
def dest(s_km, az):   # great-circle destination
    d=s_km*1000/RE; la1=math.radians(LAT0); lo1=math.radians(LON0)
    la2=np.arcsin(np.sin(la1)*np.cos(d)+np.cos(la1)*np.sin(d)*np.cos(az))
    lo2=lo1+np.arctan2(np.sin(az)*np.sin(d)*np.cos(la1),np.cos(d)-np.sin(la1)*np.sin(la2))
    return np.degrees(la2),np.degrees(lo2)
def nvec(lat,lon):
    la,lo=np.radians(lat),np.radians(lon); return np.stack([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)],-1)
n0=nvec(LAT0,LON0); east=np.array([-math.sin(math.radians(LON0)),math.cos(math.radians(LON0)),0]); north=np.cross(n0,east)
FLOOR=200.0; LIFT=5.0; ORIG=FLOOR+LIFT
rings=np.concatenate([[0.0],np.arange(1,70,1.0),np.arange(70,80.01,0.2)])
NS=256; az=np.arange(NS)/NS*2*np.pi
grid=[]; keep=[]
for s in rings:
    if s==0: la,lo=np.array([LAT0]),np.array([LON0])
    else: la,lo=dest(s,az)
    h=np.maximum(terrain(la,lo)+LIFT, ORIG)
    p=(RE+h)[:,None]*nvec(la,lo)-(RE+ORIG)*n0
    loc=np.stack([p@east,p@north,p@n0],-1)
    grid.append((s,loc,terrain(la,lo)))
# stop the cap where the wall has climbed 40 m above the floor (all around)
last=0
for k,(s,loc,t) in enumerate(grid):
    if s>0 and (t>FLOOR+40).all(): break
    last=k
grid=grid[:last+1]; print('cap radius km', grid[-1][0], 'rings', len(grid))
m=Model('Kerbangelion_DiracSea')
V,Nn,UV,T=m.parts.setdefault('DiracBlack',([],[],[],[]))
centre=grid[0][1][0]; V.append(tuple(centre)); Nn.append((0,0,1)); UV.append((0.5,0.5))
idx=[[0]*NS]
for s,loc,t in grid[1:]:
    row=[]
    for k in range(NS):
        V.append(tuple(loc[k])); Nn.append((0,0,1)); UV.append((0.5+0.5*math.cos(az[k])*s/80,0.5+0.5*math.sin(az[k])*s/80)); row.append(len(V)-1)
    idx.append(row)
for k in range(NS): T.append((0,idx[1][k],idx[1][(k+1)%NS]))
for r in range(1,len(idx)-1):
    for k in range(NS):
        a,b,c,d=idx[r][k],idx[r][(k+1)%NS],idx[r+1][(k+1)%NS],idx[r+1][k]
        T.append((a,d,c)); T.append((a,c,b))
# orient: make every triangle face up (+Z)
V_=np.array(V); T2=[]
for a,b,c in T:
    n=np.cross(V_[b]-V_[a],V_[c]-V_[a]); T2.append((a,b,c) if n[2]>0 else (a,c,b))
T[:]=T2
print('verts',len(V),'tris',len(T))
from PIL import Image
Image.new('RGB',(8,8),(0,0,0)).save('/tmp/kg/models/Dirac_Black.png')
mat={'DiracBlack':{'shader':'KSP/Unlit','tex':{'_MainTex':'Dirac_Black.png'},'colors':{'_Color':(0,0,0,1),**COMMON},'floats':{**CF}}}
m.write_mu(mat); m.write_obj({'DiracBlack':('Dirac_Black.png',None)})
