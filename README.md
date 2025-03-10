AI Black & White Image Colorization

This project is a black-and-white image colorization tool using OpenCV and Deep Learning. It utilizes a pre-trained Caffe model to colorize grayscale images automatically.

🚀 Features

Upload a black-and-white image, and the model will colorize it.

Uses Deep Learning with a pre-trained Caffe model.

Streamlit-based UI for easy interaction.

Error handling for missing files or incorrect formats.

📌 Installation

Clone the repository:

git clone https://github.com/vishaljain147/image-colorization.git

cd your-repo

Install dependencies:

pip install -r requirements.txt

Download Pre-trained Model Files:

Download the required model files:

models_colorization_deploy_v2.prototxt

colorization_release_v2.caffemodel

pts_in_hull.npy

Place them inside the models/ directory.

🎮 Usage

Run the Streamlit app:

streamlit run app.py

🎨 How It Works

Upload a black-and-white image using the UI.

The deep learning model processes the image.

The colorized image is displayed next to the original.

📷 Example

Original	Colorized

🛠 Dependencies

Python 3.x

OpenCV

NumPy

Streamlit

PIL

Install all dependencies with:

pip install -r requirements.txt

⚠ Troubleshooting

Model files not found? Ensure they are correctly placed in the models/ directory.

Streamlit not running? Try reinstalling with:

pip install --upgrade streamlit

📜 License

This project is licensed under the MIT License.
