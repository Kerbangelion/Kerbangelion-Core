import struct, numpy as np
def write_rgba8_dds(path, rgba):
    h, w = rgba.shape[:2]
    hdr = bytearray(128)
    struct.pack_into('<4sIIIIIII', hdr, 0, b'DDS ', 124, 0x1|0x2|0x4|0x1000|0x8, h, w, w*4, 0, 1)
    struct.pack_into('<II4sIIIII', hdr, 76, 32, 0x4, b'DX10', 0, 0, 0, 0, 0)
    struct.pack_into('<I', hdr, 108, 0x1000)               # caps: texture
    dx10 = struct.pack('<IIIII', 28, 3, 0, 1, 0)           # R8G8B8A8_UNORM, TEXTURE2D, 1 array elem
    with open(path, 'wb') as f:
        f.write(hdr); f.write(dx10); f.write(np.ascontiguousarray(rgba.astype(np.uint8)).tobytes())
