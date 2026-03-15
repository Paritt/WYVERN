# 🐉 WYVERN

This project provides a GUI for chest X-ray (CXR) inspiration and rotation inspection. 🩻✨

## 🛠️ How It Works

The WYVERN application uses deep learning models to analyze chest X-ray images. Through its graphical user interface, you can:

- Select images (.dcm or .png or .jpg) for inspection
- Click Analyze button to start the process
- We detect inspiration by segmenting the lungs and 9th posterior rib and calculate overlap ratio called 'Rib over Lung (ROL)' to determine if the inspiration is sufficient. If it above a certain threshold, we classify it as 'Adequate Inspiration', otherwise 'Inadequate Inspiration'.
- For rotation, we analyze the x position medial end of clavicles and the spinous process to determine if the patient is rotated. We calculate 'Alpha' which is quantify metric for access symmetry. If the alpha is above a certain threshold, we classify it as 'Rotated', otherwise 'Not Rotated'.

The models in `src/models` are used to perform the image analysis, ensuring accurate and fast inspection. The GUI makes it easy for users to operate without needing to write code.

## 🚀 How to Run

1. 📦 Install the required Python packages:

	```bash
	pip install -r requirements.txt
	```

2. 🧠 Place the required model files in the `src/models` directory.

Once the dependencies and models are in place, you can launch the GUI for CXR inspiration and rotation inspection! 🎉