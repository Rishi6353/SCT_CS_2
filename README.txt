# ImageCrypt - Ready to Run

This is a simple, **educational** image encryption/decryption tool using pixel manipulation.

## How to Use

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Encrypt the test image
```bash
python imagecrypt.py encrypt --in test_image.png --out secret.png --ops "xor:77,add:13,swap_channels:bgr"
```

### 3. Decrypt the encrypted image
```bash
python imagecrypt.py decrypt --in secret.png --out restored.png --ops "xor:77,add:13,swap_channels:bgr"
```

`restored.png` should look exactly like `test_image.png`.

---

⚠️ **Note:** This is NOT secure encryption — it's just for learning pixel manipulation.
