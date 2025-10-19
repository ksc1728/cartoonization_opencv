import os
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
import tensorflow as tf
import tensorflow_hub as hub

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from models.Transformer import load_pretrained_model, get_available_styles, get_model_filename
    cartoongan_available = True
except ImportError:
    print("CartoonGAN models not available")
    cartoongan_available = False
    def get_available_styles():
        return []

cyclegan_path = os.path.join(project_root, 'models', 'cyclegan', 'pytorch_CycleGAN_and_pix2pix')
sys.path.insert(0, cyclegan_path)

cyclegan_available = False
try:
    from models.cyclegan.pytorch_CycleGAN_and_pix2pix.models import networks
    from models.cyclegan.pytorch_CycleGAN_and_pix2pix.models.base_model import BaseModel
    from models.cyclegan.pytorch_CycleGAN_and_pix2pix.models.cycle_gan_model import CycleGANModel
    from models.cyclegan.pytorch_CycleGAN_and_pix2pix.options.test_options import TestOptions
    print("Successfully imported CycleGAN components")
    cyclegan_available = True
except ImportError as e:
    print(f"Import error: {e}")
    try:
        from pytorch_CycleGAN_and_pix2pix.models import networks
        from pytorch_CycleGAN_and_pix2pix.models.base_model import BaseModel
        from pytorch_CycleGAN_and_pix2pix.models.cycle_gan_model import CycleGANModel
        from pytorch_CycleGAN_and_pix2pix.options.test_options import TestOptions
        print("Successfully imported CycleGAN from package")
        cyclegan_available = True
    except:
        print("Could not import CycleGAN models - these style options will not be available")


app = Flask(__name__, template_folder=os.path.join(project_root, 'templates'))
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join('static', 'outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

class ImageStylizer:
    def __init__(self):
        
        self.available_methods = {
            'neural': False,
            'opencv': True, 
            'cartoongan': False,
            'vangogh': False,
            'monet': False, 
            'cezanne': False,
            'ukiyoe': False
        }
        
        try:
            self.hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
            self.available_methods['neural'] = True
            print("Successfully loaded TensorFlow Hub module for neural style transfer")
        except Exception as e:
            print(f"Error loading TensorFlow Hub module: {str(e)}")
            self.hub_module = None
        
        if cartoongan_available:
            self._load_cartoongan_models()
        
        if cyclegan_available:
            self._load_cyclegan_models()
    
    def get_available_methods(self):
        return self.available_methods
        
    def _load_cartoongan_models(self):
        self.cartoongan_models = {}
        styles = get_available_styles()
        
        for style in styles:
            try:
                model = load_pretrained_model(style)
                if model:
                    self.cartoongan_models[style] = model
                    print(f"Loaded {style} model successfully")
                    self.available_methods['cartoongan'] = True
            except Exception as e:
                print(f"Error loading {style} model: {str(e)}")
    
    def _load_cyclegan_models(self):
        
        self.cyclegan_models = {}
        styles = ['vangogh', 'monet', 'cezanne', 'ukiyoe']
        
        import tempfile
        temp_dir = tempfile.mkdtemp()

        for style in styles:
            try:
                import argparse
                parser = argparse.ArgumentParser()
                parser = TestOptions().initialize(parser)
                
                class Options:
                    pass
                
                opt = Options()
                opt.dataroot = temp_dir
                opt.name = f'style_{style}_pretrained'
                opt.model = 'cycle_gan'
                opt.checkpoints_dir = os.path.join('models', 'cyclegan', 'checkpoints')
                opt.gpu_ids = []
                opt.input_nc = 3
                opt.output_nc = 3
                opt.ngf = 64
                opt.ndf = 64
                opt.netD = 'basic'
                opt.netG = 'resnet_9blocks'
                opt.n_layers_D = 3
                opt.norm = 'instance'
                opt.init_type = 'normal'
                opt.init_gain = 0.02
                opt.no_dropout = True
                opt.dataset_mode = 'single'
                opt.direction = 'AtoB'
                opt.serial_batches = False
                opt.num_threads = 0
                opt.batch_size = 1
                opt.load_size = 256
                opt.crop_size = 256
                opt.max_dataset_size = float("inf")
                opt.preprocess = 'resize_and_crop'
                opt.no_flip = True
                opt.display_winsize = 256
                opt.epoch = 'latest'
                opt.load_iter = 0
                opt.phase = 'test'
                opt.isTrain = False
                opt.eval = True
                opt.verbose = False
                opt.suffix = ''
                
                model = CycleGANModel(opt)
                model.setup(opt)
                
                model_path = os.path.join(opt.checkpoints_dir, opt.name, 'latest_net_G_A.pth')
                print(f"Looking for model at: {model_path}")

                if not os.path.exists(model_path):
                    print(f"Model file not found: {model_path}")
                    alt_paths = [
                        os.path.join(opt.checkpoints_dir, opt.name, 'latest_net_G.pth'),
                        os.path.join(opt.checkpoints_dir, opt.name, f'{opt.epoch}_net_G.pth'),
                        os.path.join(opt.checkpoints_dir, opt.name, f'{opt.epoch}_net_G_A.pth')
                    ]
                    
                    for alt_path in alt_paths:
                        print(f"Trying alternative path: {alt_path}")
                        if os.path.exists(alt_path):
                            model_path = alt_path
                            print(f"Found model at: {model_path}")
                            break
                    else:
                        print(f"No model found for {style}. Skipping.")
                        continue
                
                state_dict = torch.load(model_path, map_location='cpu')
                
                # Handling different state dict formats and filter out unexpected keys
                if hasattr(model, 'netG_A'):
                    # Get the model's state dict to extract required keys
                    model_state_dict = model.netG_A.state_dict()
                    required_keys = set(model_state_dict.keys())
                    
                    # Processing the loaded state dict to match required format
                    if isinstance(state_dict, dict):
                        # Handle full model state dict
                        if 'model.netG' in state_dict:
                            filtered_state_dict = {k: v for k, v in state_dict['model.netG'].items() if k in required_keys}
                        elif 'netG' in state_dict:
                            filtered_state_dict = {k: v for k, v in state_dict['netG'].items() if k in required_keys}
                        else:
                            #state dict - filter out batch norm stats
                            filtered_state_dict = {}
                            for k, v in state_dict.items():
                                # Check if key is in required keys
                                if k in required_keys:
                                    filtered_state_dict[k] = v
                                # If this is a model prefix format
                                elif k.startswith('model.'):
                                    # Extract the part after 'model.'
                                    clean_key = k.replace('model.', '')
                                    if clean_key in required_keys:
                                        filtered_state_dict[clean_key] = v
                    else:
                        # state dict - filter out batch norm stats
                        filtered_state_dict = {k: v for k, v in state_dict.items() if k in required_keys}
                    
                    #filtered state dict
                    missing_keys = set(required_keys) - set(filtered_state_dict.keys())
                    if missing_keys:
                        print(f"Warning: Missing keys in state dict for {style}: {missing_keys}")
                    
                    try:
                        # strict loading first
                        model.netG_A.load_state_dict(filtered_state_dict, strict=True)
                    except Exception as e:
                        print(f"Strict loading failed: {e}")
                        # non-strict loading as fallback
                        model.netG_A.load_state_dict(filtered_state_dict, strict=False)
                        print(f"Loaded with non-strict mode for {style}")
                    
                    print(f"Successfully loaded weights for {style}")
                    model.eval()
                    self.cyclegan_models[style] = model
                    self.available_methods[style] = True
                else:
                    raise AttributeError("Model has no netG_A attribute")
                
            except Exception as e:
                print(f"Failed to load {style} CycleGAN model: {str(e)}")
                continue
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)

    def apply_neural_style(self, content_path, style_path):
       
        # if not self.hub_module:
        #     raise ValueError("Neural style transfer module is not available")
            
        # content_img = tf.constant(load_image(content_path))
        # style_img = tf.constant(load_image(style_path))
        # result = self.hub_module(content_img, style_img)
        # return tf.squeeze(result[0])
        if not self.hub_module:
            raise ValueError("Neural style transfer module is not available")
            
        content_img = load_image_neural_style(content_path)
        style_img = load_image_neural_style(style_path)
        outputs = self.hub_module(tf.constant(content_img), tf.constant(style_img))
        stylized_image = outputs[0]
        
        # Convert and save the stylized image
        stylized_image = tf.squeeze(stylized_image)
        stylized_image = tf.image.convert_image_dtype(stylized_image, tf.uint8)
        stylized_image = np.array(stylized_image)
        return stylized_image

    def apply_opencv_cartoon(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (500, 500))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.medianBlur(gray, 7)
        edges = cv2.Canny(smoothed, 50, 100)
        edges_inverted = cv2.bitwise_not(edges)
        bilateral = cv2.bilateralFilter(img, 10, 100, 100)
        edges_colored = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(bilateral, edges_colored)
        
        return Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

    def apply_cartoongan(self, image_path, style):
        """Apply CartoonGAN style"""
        if style not in self.cartoongan_models:
            raise ValueError(f"Style {style} not available")
            
        model = self.cartoongan_models[style]
        img = Image.open(image_path).convert("RGB")
        
        # Resizing
        w, h = img.size
        ratio = w / h
        load_size = 450
        
        if ratio > 1:
            w = load_size
            h = int(w / ratio)
        else:
            h = load_size
            w = int(h * ratio)
            
        img = img.resize((w, h), Image.BICUBIC)
        img = np.asarray(img)[:, :, [2, 1, 0]] 
        img = transforms.ToTensor()(img).unsqueeze(0)
        img = -1 + 2 * img  
        
        with torch.no_grad():
            output = model(img)[0]
        
        output = output[[2, 1, 0], :, :]  
        output = output.data.cpu().float() * 0.5 + 0.5
        output = np.uint8(output.numpy().transpose(1, 2, 0) * 255)
        
        return Image.fromarray(output)

    def apply_cyclegan(self, image_path, style):
        """Apply CycleGAN style transfer"""
        if style not in self.cyclegan_models:
            raise ValueError(f"Style {style} not available")
            
        model = self.cyclegan_models[style]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  

        with torch.no_grad():
            model.eval()
            fake_img = model.netG_A(img_tensor)[0]

        fake_img = fake_img * 0.5 + 0.5  
        fake_img = fake_img.clamp(0, 1) 
        fake_img_np = fake_img.permute(1, 2, 0).cpu().numpy() *636
        fake_img_pil = Image.fromarray(np.uint8(fake_img_np))

        return fake_img_pil


stylizer = ImageStylizer()

##


@app.route('/', methods=['GET', 'POST'])
def index():
    available_methods = stylizer.get_available_methods()
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            method = request.form.get('method')
        
            if method not in available_methods or available_methods.get(method) is not True:
                return render_template('index.html', 
                                    error=f"The selected method '{method}' is not available.",
                                    anime_styles=get_available_styles(),
                                    available_methods=available_methods)
            
            result_filename = f"{method}_{filename}"
            result_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)
            
            try:
                if method == 'neural':
                    style_file = request.files.get('style_file')
                    if not style_file or not allowed_file(style_file.filename):
                        return render_template('index.html', 
                                           error="Style image missing or invalid",
                                           anime_styles=get_available_styles(),
                                           available_methods=available_methods)
                    
                    style_filename = secure_filename(f'style_{style_file.filename}')
                    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
                    style_file.save(style_path)
                    
                    result = stylizer.apply_neural_style(filepath, style_path)
                    Image.fromarray(np.array(tf.image.convert_image_dtype(result, tf.uint8))).save(result_path)
                
                elif method == 'opencv':
                    result = stylizer.apply_opencv_cartoon(filepath)
                    result.save(result_path)
                
                elif method == 'cartoongan':
                    style = request.form.get('anime_style', 'Hayao')
                    result = stylizer.apply_cartoongan(filepath, style)
                    result.save(result_path)
                
                elif method in ['vangogh', 'monet', 'cezanne', 'ukiyoe']:
                    result = stylizer.apply_cyclegan(filepath, method)
                    result.save(result_path)
                
                
                return render_template('index.html', 
                                    original=f"uploads/{filename}", 
                                    result=f"outputs/{result_filename}",
                                    selected_method=method,
                                    anime_styles=get_available_styles(),
                                    available_methods=available_methods)
            
            except Exception as e:
                return render_template('index.html', 
                                    error=str(e),
                                    anime_styles=get_available_styles(),
                                    available_methods=available_methods)
    
    return render_template('index.html', 
                         anime_styles=get_available_styles(),
                         available_methods=available_methods)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def tensor2im(input_image, imtype=np.uint8):
    
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

def load_image(path, max_size=512):
    img = Image.open(path).convert('RGB')
    
    w, h = img.size
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img = np.array(img) / 255.0
    return img[np.newaxis, ...].astype(np.float32)

def load_image_neural_style(img_path):
    
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
