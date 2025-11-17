
# AI Image Style Transfer Web App

A web-based application for applying artistic and cartoon-style transformations to images using various AI models including OpenCV, Neural Style Transfer, CartoonGAN, and CycleGAN-based styles like Van Gogh, Monet, Cezanne, and Ukiyoe.

<img width="1200" height="700" alt="image" src="https://github.com/user-attachments/assets/00321b04-8a2b-416c-8276-fc7bcc2352ff" />

<img width="1200" height="700" alt="image" src="https://github.com/user-attachments/assets/b276715b-b7b5-4a6a-97ac-d8dadbce97f8" />
<img width="1200" height="700" alt="image" src="https://github.com/user-attachments/assets/c7f1235b-6622-48d7-acb3-7247d1b17ead" />


## Features

- Upload an image and apply different style transfer techniques
- Supports multiple methods:
  - OpenCV Cartoonization
  - Neural Style Transfer (TensorFlow Hub)
  - CartoonGAN (Hayao, Hosoda, Paprika, Shinkai)
  - CycleGAN styles (Van Gogh, Monet, Cezanne, Ukiyoe)
- Dynamic form fields based on selected method
- Preview original and styled images side-by-side
- Built with Flask, HTML, CSS, JavaScript, and Bootstrap

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Image Processing**:
  - OpenCV (bilateral filtering)
  - TensorFlow & TensorFlow Hub (Neural Style Transfer)
  - PyTorch (CycleGAN models)
- **Models**:
  - CartoonGAN: Hayao, Hosoda, Paprika, Shinkai
  - CycleGAN: Van Gogh, Monet, Cezanne, Ukiyoe

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ksc1728/cartoonization_opencv.git
   cd ai-style-transfer
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

4. **Download pre-trained models**:
   - Place CartoonGAN models in `models/cartoongan/`
   - Place CycleGAN models in `models/cyclegan/checkpoints/`
   - Ensure each model folder contains `latest_net_G_A.pth`

5. **Run the app**:
   ```bash
   python app.py
   ```

6. **Access the app**:
   Open your browser and go to `http://localhost:5000`

## 📁 Project Structure

```
ai-style-transfer/
├── static/
│   └── uploads/, outputs/, css/
├── templates/
│   └── index.html
├── models/
│   ├── cartoongan/
│   └── cyclegan/checkpoints/
├── app.py
└── README.md
```

## Usage

1. Upload an image.
2. Select a style transfer method.
3. If using Neural Style Transfer, upload a style image.
4. If using CartoonGAN, select an anime style.
5. Click "Apply Style" to process the image.
6. View the original and styled image previews.

## Notes

- Ensure GPU support is enabled for faster processing (PyTorch and TensorFlow).
- Large images may take longer to process.
- Style image is required only for Neural Style Transfer.
