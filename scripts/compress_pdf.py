"""Shrink an exported slide PDF (one full-page image per slide) below a size limit, keeping its text layer.

    .venv\\Scripts\\python scripts\\compress_pdf.py "in.pdf" "out.pdf"

Text-heavy pages become 256-colour palette images at 2400 px wide, embedded losslessly as PNG-predicted
Flate streams. Photo-like pages (screenshots, gradients) become 1920 px JPEGs, whichever is smaller.
"""
import io
import struct
import sys

import pymupdf
from PIL import Image


def png_parts(buf):
    data, idat, plte, pos = buf[8:], b"", None, 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        typ, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if typ == b"PLTE":
            plte = body
        if typ == b"IDAT":
            idat += body
        pos += 12 + n
    return plte, idat


def compress(src, dst, width=2400, jpeg_quality=88):
    d = pymupdf.open(src)
    for page in d:
        for im in page.get_images(full=True):
            xref = im[0]
            img = Image.open(io.BytesIO(d.extract_image(xref)["image"])).convert("RGB")
            height = round(width * img.size[1] / img.size[0])
            pal = img.resize((width, height), Image.LANCZOS).quantize(
                colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            buf = io.BytesIO()
            pal.save(buf, "PNG", optimize=True)
            plte, idat = png_parts(buf.getvalue())
            jpg = io.BytesIO()
            img.resize((1920, round(1920 * img.size[1] / img.size[0])), Image.LANCZOS).save(
                jpg, "JPEG", quality=jpeg_quality, optimize=True, progressive=True, subsampling=0)
            if len(idat) + len(plte) < len(jpg.getvalue()) * 1.15:
                d.update_stream(xref, idat, compress=False)
                for key in ("SMask", "Mask", "Decode", "Interpolate"):
                    d.xref_set_key(xref, key, "null")
                d.xref_set_key(xref, "Filter", "/FlateDecode")
                d.xref_set_key(xref, "DecodeParms",
                               "<</Predictor 15/Colors 1/BitsPerComponent 8/Columns %d>>" % width)
                d.xref_set_key(xref, "ColorSpace",
                               "[/Indexed/DeviceRGB %d <%s>]" % (len(plte) // 3 - 1, plte.hex()))
                d.xref_set_key(xref, "BitsPerComponent", "8")
                d.xref_set_key(xref, "Width", str(width))
                d.xref_set_key(xref, "Height", str(height))
            else:
                page.replace_image(xref, stream=jpg.getvalue())
    d.save(dst, garbage=4, deflate=True, clean=True)


if __name__ == "__main__":
    compress(sys.argv[1], sys.argv[2])
    import os
    print("%s: %.2f MB" % (sys.argv[2], os.path.getsize(sys.argv[2]) / 1e6))
