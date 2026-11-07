# GeoFront statics v2: NERV HQ (inverted-pyramid lake + pyramid on its corner) and the solar collector.
# Authoring frame = Blender: X east, Y north, Z up, metres.
import sys, math, numpy as np
sys.path.insert(0,'/tmp/kg'); sys.path.insert(0,'/tmp/mu_src/io_object_mu')
from modelib import Model, emissive, specular, COMMON, CF, smooth_grid
# ---- dimensions ----
S=300.0; LIP=1.0; D=270.0; DK=560.0; SK=14.0          # lake half-size, water below deck, depth, deck half-size, skirt
BOT=-LIP-D                                             # lake bottom (z)
HP=300.0; B0=250.0                                     # pyramid height above the deck and half-base at deck level
slope=HP/B0                                            # pyramid face slope (1.2) - lake faces are D/S = 0.73, shallower
kc=B0*math.sqrt(2)/HP                                  # horizontal corner distance per metre of height
dP=(HP-BOT)*kc                                         # pyramid axis distance from the lake centre (along the diagonal)
PX=PY=dP/math.sqrt(2)                                  # so the SW edge of the pyramid runs down into the lake bottom
print('pyramid centre offset from lake centre: %.1f m east, %.1f m north'%(PX,PY))
def add_pyramid(m, ox, oy, emblem=True):
    A=(ox,oy,HP); hb=(HP-BOT)/slope                      # half side at the lake-bottom level
    c=[(ox-hb,oy-hb,BOT),(ox+hb,oy-hb,BOT),(ox+hb,oy+hb,BOT),(ox-hb,oy+hb,BOT)]
    for i in range(4):
        a,b=c[i],c[(i+1)%4]; out=np.array([(a[0]+b[0])/2-ox,(a[1]+b[1])/2-oy,hb/3])
        m.poly('PyramidFace',[a,b,A],[(0,0),(1,0),(0.5,1)],out)
    m.poly('PyramidFace',c[::-1],[(0,0)]*4,np.array([0,0,-1.0]))
    if emblem:   # square decal on the south face, a little above half height
        fn=np.array([0,-HP,B0]); fn/=np.linalg.norm(fn); up=np.array([0,B0,HP]); up/=np.linalg.norm(up)
        t0=0.744; half=38.0; ctr=np.array([ox,oy-B0*(1-t0),HP*t0])+fn*0.4
        q=[ctr+np.array([-half,0,0])-up*half, ctr+np.array([half,0,0])-up*half, ctr+np.array([half,0,0])+up*half, ctr+np.array([-half,0,0])+up*half]
        m.poly('PyramidEmblem',[tuple(x) for x in q],[(0,0),(1,0),(1,1),(0,1)],fn)
def add_lake(m):
    w=[(-S,-S,-LIP),(S,-S,-LIP),(S,S,-LIP),(-S,S,-LIP)]; apex=(0,0,BOT)
    for i in range(4):
        a,b=w[i],w[(i+1)%4]; inward=np.array([-(a[0]+b[0])/2,-(a[1]+b[1])/2,200.0])
        m.poly('LakeFace',[a,b,apex],[(0,0),(1,0),(0.5,1)],inward)
    ci=[(-S,-S),(S,-S),(S,S),(-S,S)]; co=[(-DK,-DK),(DK,-DK),(DK,DK),(-DK,DK)]
    for i in range(4):
        a,b=ci[i],ci[(i+1)%4]; A2,B2=co[i],co[(i+1)%4]
        m.quad_planar('Deck',[(a[0],a[1],0),(A2[0],A2[1],0),(B2[0],B2[1],0),(b[0],b[1],0)],np.array([0,0,1.0]),50.0,(0,1))
        m.quad_planar('Deck',[(a[0],a[1],0),(b[0],b[1],0),(b[0],b[1],-LIP),(a[0],a[1],-LIP)],np.array([-(a[0]+b[0]),-(a[1]+b[1]),0.0]),50.0,(0 if a[1]==b[1] else 1,2))
        m.quad_planar('Deck',[(A2[0],A2[1],0),(B2[0],B2[1],0),(B2[0],B2[1],-SK),(A2[0],A2[1],-SK)],np.array([(A2[0]+B2[0]),(A2[1]+B2[1]),0.0]),50.0,(0 if A2[1]==B2[1] else 1,2))
        r0,r1=S,S+6.0
        pa=[(a[0]*r0/S,a[1]*r0/S),(b[0]*r0/S,b[1]*r0/S),(b[0]*r1/S,b[1]*r1/S),(a[0]*r1/S,a[1]*r1/S)]
        m.poly('RimLight',[(x,y,0.06) for x,y in pa],[(0,0),(1,0),(1,1),(0,1)],np.array([0,0,1.0]))
MATS={'PyramidFace':emissive('Pyramid_Face.png','Pyramid_Face_Glow.png',spec=(0.22,0.26,0.36,1),shin=0.12),
      'PyramidEmblem':{'shader':'KSP/Alpha/Cutoff','tex':{'_MainTex':'Pyramid_Emblem.png'},'colors':{'_Color':(1,1,1,1),**COMMON},'floats':{'_Cutoff':0.5,**CF}},
      'LakeFace':emissive('Lake_Face.png','Lake_Face_Glow.png',spec=(0.9,0.95,1,1),shin=0.03),
      'Deck':specular('Deck_Concrete.png',spec=(0.15,0.15,0.15,1),shin=0.3),
      'RimLight':{'shader':'KSP/Unlit','tex':{'_MainTex':'Rim_Light.png'},'colors':{'_Color':(1,1,1,1),**COMMON},'floats':{**CF}},
      'PanelSolar':emissive('Solar_Cells.png','Solar_Cells_Glow.png',spec=(0.7,0.75,0.85,1),shin=0.06),
      'PanelShell':emissive('Solar_Back.png','Solar_Frame_Glow.png',spec=(0.35,0.35,0.38,1),shin=0.2),
      'PanelFrame':emissive('Solar_Frame.png','Solar_Frame_Glow.png',spec=(0.35,0.35,0.38,1),shin=0.2)}
OBJT={'PyramidFace':('Pyramid_Face.png','Pyramid_Face_Glow.png'),'PyramidEmblem':('Pyramid_Emblem.png',None),
      'LakeFace':('Lake_Face.png','Lake_Face_Glow.png'),'Deck':('Deck_Concrete.png',None),'RimLight':('Rim_Light.png',None),
      'PanelSolar':('Solar_Cells.png','Solar_Cells_Glow.png'),'PanelShell':('Solar_Back.png',None),'PanelFrame':('Solar_Frame.png','Solar_Frame_Glow.png')}
def sub(names): return {k:v for k,v in OBJT.items() if k in names}
# in-game: one static for the lake + pyramid so they can never drift apart
hq=Model('NERV_HQ'); add_lake(hq); add_pyramid(hq,PX,PY)
hq.write_mu({k:MATS[k] for k in hq.parts}, collider_mats=['LakeFace','Deck','PyramidFace'])
# reference exports
p=Model('NERV_Pyramid'); add_pyramid(p,0,0); p.write_obj(sub(p.parts))
l=Model('NERV_GeoFrontLake'); add_lake(l); l.write_obj(sub(l.parts))
hq.write_obj(sub(hq.parts))

# =============== solar collector: a third of a tall cone, as a thin curved shell ===============
def add_panel(m, face_deg):
    """face_deg: compass-free angle (deg, from +X towards +Y) the concave side looks at. Origin = cone axis at ground."""
    H=600.0; R0,R1=80.0,64.0; TH=4.0; SPAN=math.radians(120); NA=28; NV=24
    back=math.radians(face_deg)+math.pi
    th=back+np.linspace(-SPAN/2,SPAN/2,NA); zs=np.linspace(0,H,NV)
    T,Z=np.meshgrid(th,zs); Rz=R0+(R1-R0)*Z/H
    def shell(off, inner):
        r=Rz+off; P=np.stack([r*np.cos(T),r*np.sin(T),Z],-1)
        dr=(R1-R0)/H; n=np.stack([np.cos(T),np.sin(T),-dr*np.ones_like(T)],-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
        if inner: n=-n
        uv=np.stack([(T-th[0])/(th[-1]-th[0]),Z/H],-1)
        return P,n,uv
    P,n,uv=shell(0,True); smooth_grid(m,'PanelSolar',P,n,uv)              # concave solar face
    P,n,uv=shell(TH,False); uv[...,0]*=6; uv[...,1]*=15; smooth_grid(m,'PanelShell',P,n,uv)   # white back
    # frame: the two vertical edges, the top edge, as boxes
    def box(cx,cy,z0,z1,hx,hy,ang):
        ca,sa=math.cos(ang),math.sin(ang)
        pts=[(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]; ft=[(cx+x*ca-y*sa,cy+x*sa+y*ca) for x,y in pts]
        for i in range(4):
            a,b=ft[i],ft[(i+1)%4]; out=np.array([(a[0]+b[0])/2-cx,(a[1]+b[1])/2-cy,0.0])
            m.quad_planar('PanelFrame',[(a[0],a[1],z0),(b[0],b[1],z0),(b[0],b[1],z1),(a[0],a[1],z1)],out,40.0,(0 if abs(b[0]-a[0])>abs(b[1]-a[1]) else 1,2))
        m.quad_planar('PanelFrame',[(x,y,z1) for x,y in ft],np.array([0,0,1.0]),40.0,(0,1))
    for t in (th[0],th[-1]):
        for k in range(NV-1):   # follow the taper
            z0,z1=zs[k],zs[k+1]; r=R0+(R1-R0)*(z0+z1)/2/H+TH/2
            box(r*math.cos(t),r*math.sin(t),z0,z1,TH/2+3,4.0,t)
        r=R1+TH/2; box(r*math.cos(t),r*math.sin(t),H,H+45,1.5,1.5,t)        # antenna masts
    # top rim
    for k in range(NA-1):
        t=(th[k]+th[k+1])/2; r=R1+TH/2; L=r*(th[k+1]-th[k])/2+0.5
        box(r*math.cos(t),r*math.sin(t),H-3,H+2,TH/2+2,L,t+math.pi/2)
    # flared foot: sloped skirts on both sides of the shell + top band, along the arc
    def ring(r,z): return np.stack([r*np.cos(th),r*np.sin(th),np.full_like(th,z)],-1)
    for (ra,za,rb,zb,inner) in ((R0+TH,28,R0+TH+34,0,False),(R0,28,R0-26,0,True)):
        P=np.stack([ring(rb,zb),ring(ra,za)]); Pn=P.copy()
        d=np.array([rb-ra,0,za-zb]); d/=np.linalg.norm(d)
        n=np.stack([np.cos(th)*d[2]*np.sign(rb-ra)*(1 if not inner else 1), np.sin(th)*d[2]*np.sign(rb-ra), np.full_like(th,abs(d[0]))],-1)
        n=np.stack([n,n]); uv=np.stack([np.stack([np.linspace(0,8,NA),np.zeros(NA)],-1),np.stack([np.linspace(0,8,NA),np.ones(NA)],-1)])
        smooth_grid(m,'PanelFrame',P,n,uv)
    for t,s in ((th[0],-1),(th[-1],1)):   # foot end caps
        nrm=np.array([-math.sin(t)*s,math.cos(t)*s,0.0])
        q=[((R0-26)*math.cos(t),(R0-26)*math.sin(t),0),((R0+TH+34)*math.cos(t),(R0+TH+34)*math.sin(t),0),((R0+TH)*math.cos(t),(R0+TH)*math.sin(t),28),(R0*math.cos(t),R0*math.sin(t),28)]
        m.poly('PanelFrame',q,[(0,0),(1,0),(1,1),(0,1)],nrm)
# the collector stands NE of the HQ; its concave face looks back towards the lake (south-west-ish)
sp=Model('NERV_SolarCollector'); add_panel(sp, math.degrees(math.atan2(-450,-1250)))
sp.write_mu({k:MATS[k] for k in sp.parts}); sp.write_obj(sub(sp.parts))
for m in (hq,sp):
    print(m.name,'verts',sum(len(v[0]) for v in m.parts.values()),'tris',sum(len(v[3]) for v in m.parts.values()))
