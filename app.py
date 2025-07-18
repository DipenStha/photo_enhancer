from flask import Flask, render_template, request, send_file
import os
import cv2
import torch
import numpy as np
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
            sr_img, _ = model.enhance(img)
            
            torch.cuda.empty_cache()
            if sr_img is None:
                return "Image enhancement failed."

            # Save enhanced image
            output_filename = f"enhanced_{img_file.filename}"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            sr_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, sr_bgr)
                        # Get sizes
            original_size = get_readable_size(input_path)
            enhanced_size = get_readable_size(output_path)

            return render_template('index.html',
                                  original=input_path.replace('\\', '/'),
                enhanced=output_path.replace('\\', '/'),
                original_size=original_size,
                enhanced_size=enhanced_size)
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
