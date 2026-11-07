# Builds the GeoFront statics as .mu (KSP) and .obj/.mtl. Authoring frame = Blender: X east, Y north, Z up, metres.
import sys, math, numpy as np
sys.path.insert(0,'/tmp/mu_src/io_object_mu')
from mu import Mu, MuObject, MuTransform, MuTagLayer, MuMesh, MuRenderer, MuMaterial, MuMatTex, MuTexture, MuColliderMesh
O='/tmp/kg/models/'
class Model:
    def __init__(s, name): s.name=name; s.parts={}          # material -> (verts, normals, uvs, tris)
    def poly(s, mat, pts, uvs, out=None):
        P=[np.array(p,float) for p in pts]
        n=np.cross(P[1]-P[0],P[2]-P[0])
        if out is not None and np.dot(n,out)<0: P=P[::-1]; uvs=uvs[::-1]; n=-n
        n=n/np.linalg.norm(n)
        V,Nn,UV,T=s.parts.setdefault(mat,([],[],[],[]))
        b=len(V)
        for p,uv in zip(P,uvs): V.append(tuple(p)); Nn.append(tuple(n)); UV.append(tuple(uv))
        for i in range(1,len(P)-1): T.append((b,b+i,b+i+1))
    def quad_planar(s, mat, pts, out, scale, axes):
        """planar UVs in metres/scale along two world axes"""
        uvs=[(p[axes[0]]/scale, p[axes[1]]/scale) for p in pts]; s.poly(mat,pts,uvs,out)
    # -------- export --------
    def write_obj(s, mats):
        with open(O+s.name+'.mtl','w') as m:
            for k,(tex,glow) in mats.items():
                m.write('newmtl %s\nKd 1 1 1\nmap_Kd %s\n'%(k,tex)+('map_Ke %s\nKe 1 1 1\n'%glow if glow else '')+'\n')
        with open(O+s.name+'.obj','w') as f:
            f.write('# Kerbangelion %s (Y-up, metres)\nmtllib %s.mtl\no %s\n'%(s.name,s.name,s.name))
            off=1
            for mat,(V,Nn,UV,T) in s.parts.items():
                for x,y,z in V: f.write('v %.4f %.4f %.4f\n'%(x,z,-y))
                for x,y,z in Nn: f.write('vn %.5f %.5f %.5f\n'%(x,z,-y))
                for a,b in UV: f.write('vt %.5f %.5f\n'%(a,b))
                f.write('usemtl %s\n'%mat)
                for t in T: f.write('f '+' '.join('%d/%d/%d'%(i+off,i+off,i+off) for i in t)+'\n')
                off+=len(V)
    def write_mu(s, matdefs, collider_mats=None):
        mu=Mu(s.name); mu.materials=[]; mu.textures=[]
        texidx={}
        def tex(name):
            if name not in texidx:
                t=MuTexture(); t.name=name; t.type=0; texidx[name]=len(mu.textures); mu.textures.append(t)
            return texidx[name]
        def mt(i): m=MuMatTex(); m.index=i; m.scale=(1.0,1.0); m.offset=(0.0,0.0); return m
        root=MuObject(s.name); root.transform=xf(s.name); root.tag_and_layer=tl()
        for mi,(mat,(V,Nn,UV,T)) in enumerate(s.parts.items()):
            d=matdefs[mat]; m=MuMaterial(); m.name=mat; m.shaderName=d['shader']
            m.colorProperties=dict(d.get('colors',{})); m.floatProperties3=dict(d.get('floats',{}))
            m.textureProperties={k:mt(tex(v)) for k,v in d['tex'].items()}
            mu.materials.append(m)
            ob=MuObject(mat); ob.transform=xf(mat); ob.tag_and_layer=tl()
            me=MuMesh(); me.verts=list(V); me.normals=list(Nn); me.uvs=list(UV); me.submeshes=[list(T)]
            ob.shared_mesh=me; r=MuRenderer(); r.materials=(mi,); r.castShadows=1; r.receiveShadows=1; ob.renderer=r
            root.children.append(ob)
        # one non-convex mesh collider holding the solid parts
        cv=[]; ct=[]
        for mat,(V,Nn,UV,T) in s.parts.items():
            if collider_mats is not None and mat not in collider_mats: continue
            b=len(cv); cv+=V; ct+=[(a+b,c+b,e+b) for a,c,e in T]
        co=MuObject('collider'); co.transform=xf('collider'); co.tag_and_layer=tl()
        c=MuColliderMesh(True); c.isTrigger=0; c.convex=0; cm=MuMesh(); cm.verts=cv; cm.submeshes=[ct]; c.mesh=cm; co.collider=c
        root.children.append(co)
        mu.obj=root; mu.write(O+s.name+'.mu')
def xf(name):
    t=MuTransform(); t.name=name; t.localPosition=(0.0,0.0,0.0); t.localRotation=(1.0,0.0,0.0,0.0); t.localScale=(1.0,1.0,1.0); return t
def tl():
    t=MuTagLayer(); t.tag='Untagged'; t.layer=15; return t
COMMON={'_RimColor':(0,0,0,0),'_TemperatureColor':(0,0,0,0),'_BurnColor':(1,1,1,1)}
CF={'_Opacity':1.0,'_RimFalloff':0.1,'_UnderwaterFogFactor':0.0}
def emissive(main, glow, spec=(0.6,0.7,0.9,1), shin=0.08, ecol=(1,1,1,1)):
    return {'shader':'KSP/Emissive/Specular','tex':{'_MainTex':main,'_Emissive':glow},
            'colors':{'_Color':(1,1,1,1),'_SpecColor':spec,'_EmissiveColor':ecol,**COMMON},'floats':{'_Shininess':shin,**CF}}
def specular(main, spec=(0.3,0.3,0.3,1), shin=0.2):
    return {'shader':'KSP/Specular','tex':{'_MainTex':main},'colors':{'_Color':(1,1,1,1),'_SpecColor':spec,**COMMON},'floats':{'_Shininess':shin,**CF}}


def smooth_grid(m, mat, P, Nrm, UVs, flip=False):
    """P,Nrm: (rows, cols, 3); UVs (rows, cols, 2). Quads between neighbours, smooth normals."""
    V,Nn,UV,T=m.parts.setdefault(mat,([],[],[],[]))
    R,C=P.shape[:2]; b=len(V)
    for r in range(R):
        for c in range(C):
            V.append(tuple(P[r,c])); Nn.append(tuple(Nrm[r,c])); UV.append(tuple(UVs[r,c]))
    for r in range(R-1):
        for c in range(C-1):
            a=b+r*C+c; bb=a+1; cc=a+C+1; d=a+C
            for t in ((a,bb,cc),(a,cc,d)):
                p0,p1,p2=(np.array(V[i]) for i in t); n=np.cross(p1-p0,p2-p0)
                if np.dot(n, np.array(Nn[t[0]]))<0: t=(t[0],t[2],t[1])
                T.append(t)
