from flask import Flask, render_template, request, send_file
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use the RealESRGAN_x4plus architecture
model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# Initialize the RealESRGAN model
model = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model_arch,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=torch.cuda.is_available(),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
def get_readable_size(path):
    size_bytes = os.path.getsize(path)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.2f} MB"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        output_format = request.form.get('format', 'jpg').lower()

        if img_file:
            input_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
            img_file.save(input_path)

            # Read and convert image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                return "Failed to read image."

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.uint8)

            # Enhance image
            with torch.no_grad():
                sr_img, _ = model.enhance(img)
            
                if sr_img is None:
                    return "Image enhancement failed."
            # Save enhanced image using correct extension
            valid_exts = {'jpg': '.jpg', 'png': '.png', 'webp': '.webp'}
            ext = valid_exts.get(output_format, '.jpg')
            # Save enhanced image in selected format
            output_filename = f"enhanced_{Path(img_file.filename).stem}.{ext}"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            
            download_name = output_filename

            # Convert back to BGR and save
            sr_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, sr_bgr)
            
             # Clear memory
            del sr_img
            torch.cuda.empty_cache()
            
            # Get sizes
            original_size = get_readable_size(input_path)
            enhanced_size = get_readable_size(output_path)

            return render_template('index.html',
                original=input_path.replace('\\', '/'),
                enhanced=output_path.replace('\\', '/'),
                original_size=original_size,
                enhanced_size=enhanced_size,
                download_name=download_name)
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
