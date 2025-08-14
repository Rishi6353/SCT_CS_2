#!/usr/bin/env python3
import argparse
import hashlib
import numpy as np
from PIL import Image
from math import gcd

# ---------- Helpers ----------

def mode_to_channels(mode: str) -> int:
    if mode in ("RGB", "YCbCr"):
        return 3
    if mode in ("RGBA", "CMYK"):
        return 4 if mode == "RGBA" else 4
    if mode in ("L", "P", "1", "I", "F", "LA"):
        # We convert to RGBA anyway, but tracking original mode helps restore it later
        return None
    return None

def modinv_256(a: int) -> int:
    # Only odd a have inverses modulo 256
    if a % 2 == 0:
        raise ValueError("mul:N requires N to be odd to be invertible modulo 256")
    # Since 256 = 2^8, for odd a, inverse exists. Use extended Euclid on modulus 256.
    t, new_t = 0, 1
    r, new_r = 256, a % 256
    while new_r != 0:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r
    if r > 1:
        raise ValueError("mul:N is not invertible modulo 256")
    if t < 0:
        t += 256
    return t

def key_to_seed(key: str, w: int, h: int) -> int:
    # Tie the seed to image size so permutations are consistent per (key, w, h)
    s = f"{key}|{w}x{h}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big", signed=False)

def parse_ops(ops_str: str):
    ops = []
    for raw in [x.strip() for x in ops_str.split(",") if x.strip()]:
        if ":" in raw:
            name, val = raw.split(":", 1)
            ops.append((name.strip().lower(), val.strip()))
        else:
            ops.append((raw.strip().lower(), None))
    return ops

def inverse_ops(ops):
    inv = []
    for name, val in reversed(ops):
        if name == "xor":
            inv.append((name, val))  # self-inverse
        elif name == "add":
            inv.append(("sub", val))
        elif name == "sub":
            inv.append(("add", val))
        elif name == "mul":
            # replace with mul:inv where inv is modular inverse of val
            k = int(val)
            inv_k = modinv_256(k)
            inv.append(("mul", str(inv_k)))
        elif name in ("invert", "transpose"):
            inv.append((name, None))  # self-inverse
        elif name == "swap_channels":
            perm = val.lower()
            base = "rgb"
            if sorted(perm) != sorted(base):
                raise ValueError("swap_channels expects a 3-letter permutation of 'rgb'")
            # find inverse permutation
            inv_perm = [None]*3
            for i, ch in enumerate(perm):
                inv_perm["rgb".index(ch)] = base[i]
            inv.append(("swap_channels", "".join(inv_perm)))
        elif name == "shuffle":
            inv.append((name, val))  # inverse handled by applying inverse permutation
        else:
            raise ValueError(f"Unknown operation: {name}")
    return inv

# ---------- Core pixel operations ----------

def to_rgba(arr: np.ndarray, mode: str):
    """Return (rgba_arr, original_mode)"""
    if mode == "RGBA":
        return arr, mode
    img = Image.fromarray(arr, mode=mode)
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    return rgba, mode

def from_rgba(rgba: np.ndarray, original_mode: str):
    img = Image.fromarray(rgba, mode="RGBA")
    return np.array(img.convert(original_mode))

def apply_xor(img: np.ndarray, n: int):
    return (img ^ n).astype(np.uint8)

def apply_add(img: np.ndarray, n: int):
    return (img + n).astype(np.uint8)

def apply_sub(img: np.ndarray, n: int):
    return (img - n).astype(np.uint8)

def apply_mul(img: np.ndarray, n: int):
    return ((img.astype(np.uint16) * n) % 256).astype(np.uint8)

def apply_invert(img: np.ndarray):
    return (255 - img).astype(np.uint8)

def apply_swap_channels(img: np.ndarray, perm: str):
    # img is RGBA; apply on first 3 channels, keep alpha
    pmap = {"r":0, "g":1, "b":2}
    idx = [pmap[c] for c in perm]
    rgb = img[..., :3]
    a = img[..., 3:4]
    sw = rgb[..., idx]
    return np.concatenate([sw, a], axis=-1)

def apply_transpose(img: np.ndarray):
    return np.transpose(img, (1,0,2))

def apply_shuffle(img: np.ndarray, key: str, decrypt: bool = False):
    h, w, c = img.shape
    seed = key_to_seed(key, w, h)
    rng = np.random.default_rng(seed)
    n = w * h
    perm = np.arange(n)
    rng.shuffle(perm)
    if decrypt:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(n)
        perm = inv
    flat = img.reshape(n, c)
    shuffled = flat[perm]
    return shuffled.reshape(h, w, c)

# ---------- Pipeline ----------

def run_pipeline(image: Image.Image, ops, mode: str):
    arr = np.array(image)
    rgba, original_mode = to_rgba(arr, image.mode)
    out = rgba.copy()

    for name, val in ops:
        if name == "xor":
            out = apply_xor(out, int(val))
        elif name == "add":
            out = apply_add(out, int(val))
        elif name == "sub":
            out = apply_sub(out, int(val))
        elif name == "mul":
            k = int(val)
            if k % 2 == 0:
                raise ValueError("mul:N requires N to be odd (invertible modulo 256).")
            out = apply_mul(out, k)
        elif name == "invert":
            out = apply_invert(out)
        elif name == "swap_channels":
            perm = val.lower()
            if len(perm) != 3 or sorted(perm) != sorted("rgb"):
                raise ValueError("swap_channels expects a 3-letter permutation of 'rgb'")
            out = apply_swap_channels(out, perm)
        elif name == "transpose":
            out = apply_transpose(out)
        elif name == "shuffle":
            out = apply_shuffle(out, val, decrypt=(mode=="decrypt"))
        else:
            raise ValueError(f"Unknown operation: {name}")

    # convert back to original mode
    final = from_rgba(out, image.mode)
    return Image.fromarray(final, mode=image.mode)

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(
        description="Toy, reversible pixel-manipulation image encryption. NOT secure."
    )
    p.add_argument("mode", choices=["encrypt", "decrypt"], help="Mode to run")
    p.add_argument("--in", dest="inp", required=True, help="Input image path")
    p.add_argument("--out", dest="out", required=True, help="Output image path")
    p.add_argument("--ops", required=True,
                   help="Comma-separated ops. Ex: 'xor:23, add:7, shuffle:myKey, swap_channels:bgr'")
    args = p.parse_args()

    ops = parse_ops(args.ops)
    if args.mode == "decrypt":
        # Build inverse op list automatically for decryption
        ops = inverse_ops(ops)

    img = Image.open(args.inp)
    out = run_pipeline(img, ops, mode=args.mode)
    out.save(args.out)
    print(f"Done: {args.mode} -> {args.out}")

if __name__ == "__main__":
    main()
