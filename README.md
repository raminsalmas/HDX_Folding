This programme utilises machine learning (ML) techniques to predict the secondary structures of amino acids in proteins. The model was trained using data from the Hydrogen Deuterium Exchange Mass Spectrometry (HDX-MS), whose resolution was improved using the HDX modeller tool, which is accessible on the HDXsite. (https://hdxsite.nms.kcl.ac.uk/) The gradient tree boosting method was employed for the current version of this model, which outperformed all previous ML methods. Using the ROC-AUC score, the current version's accuracy on the testing set is assessed to be 75%.<br >
If you encounter any issues when executing the code, please file a bug report in the GitHub issues area.<br>

Using the requirements.txt File

To ensure that you have all the necessary dependencies installed before running the HDX_Folding code, we provide a requirements.txt file that lists the required Python packages along 
with their versions. Here's how you can set up the environment using this file:

Create a Virtual Environment (Optional but Recommended)
It's a good practice to create a virtual environment to isolate the project's dependencies. If you're not familiar with virtual environments, you can create one as follows:
<pre>
python -m venv hdx_env
source hdx_env/bin/activate  # On Windows, use: hdx_env\Scripts\activate
</pre>


Install Dependencies

Once you have your virtual environment activated (if you're using one), navigate to the project directory and install the dependencies using the pip command:
<pre>
pip install -r requirements.txt
</pre>
This command will read the requirements.txt file and install the specified packages along with the versions listed.

Verifying Installation

After the installation is complete, you can verify that the required packages are installed in your environment by using the following command:
<pre>
pip list
</pre>
This will display a list of installed packages along with their versions. <be>

If you utilised this approach, please cite the linked publication below: <br>
https://pubs.acs.org/doi/full/10.1021/jasms.3c00145




<!DOCTYPE html>
<html>
<head>
    <title>DeepBack: Predicting Protein Backbone Angles using Deep Learning</title>
</head>
<body>
    <h1>DeepBack: Predicting Protein Backbone Angles using Deep Learning</h1>
    <p>Welcome to the DeepBack repository! DeepBack is a deep learning-based tool designed to predict protein backbone torsion angles. This repository contains the source code, models, and related materials for training and utilizing DeepBack for accurate backbone angle predictions.</p>
    <img src="https://github.com/raminsalmas/DeepBack/blob/main/protein_backbone.png" alt="Protein Backbone">
    
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#introduction">Introduction</a></li>
        <li><a href="#features">Features</a></li>
        <li><a href="#getting-started">Getting Started</a>
            <ul>
                <li><a href="#installation">Installation</a></li>
                <li><a href="#usage">Usage</a></li>
            </ul>
        </li>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#results">Results</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#license">License</a></li>
        <li><a href="#contact">Contact</a></li>
    </ul>

    <h2 id="introduction">Introduction</h2>
    <p>Accurate prediction of protein backbone torsion angles is crucial for understanding protein structure and function. DeepBack leverages the power of deep learning to make precise predictions, aiding in various structural biology applications.</p>
    
    <h2 id="features">Features</h2>
    <ul>
        <li><strong>Deep Learning:</strong> DeepBack utilizes deep neural networks to predict protein backbone torsion angles.</li>
        <li><strong>Easy Integration:</strong> The codebase is designed to be easily integrated into existing bioinformatics workflows.</li>
        <li><strong>Pre-trained Models:</strong> Pre-trained models are provided for immediate use, while allowing fine-tuning for specific datasets.</li>
        <li><strong>Visualization:</strong> Tools to visualize predictions and analyze model performance.</li>
        <li><strong>User-Friendly:</strong> Well-documented code with clear functions and classes for straightforward usage.</li>
    </ul>

    <h2 id="getting-started">Getting Started</h2>
    <h3 id="installation">Installation</h3>
    <ol>
        <li>Clone the repository:<br><code>git clone https://github.com/raminsalmas/DeepBack.git</code></li>
        <li>Navigate to the project directory:<br><code>cd DeepBack</code></li>
        <li>Install required Python libraries:<br><code>pip install -r requirements.txt</code></li>
    </ol>

    <h3 id="usage">Usage</h3>
    <ol>
        <li>Prepare your protein structure data in the appropriate format.</li>
        <li>Modify the configuration settings in <code>config.yaml</code> to specify model parameters.</li>
        <li>Use the provided scripts to train a new model or use a pre-trained model for prediction.</li>
        <li>Visualize predictions and assess model performance using visualization tools.</li>
    </ol>

    <h2 id="dataset">Dataset</h2>
    <p>DeepBack works best when trained on a diverse and representative dataset of protein structures. While a sample dataset is provided, you're encouraged to curate a dataset that suits your specific needs.</p>

    <h2 id="results">Results</h2>
    <p>The DeepBack model achieves state-of-the-art performance in predicting protein backbone torsion angles. You can find detailed results and performance metrics in the <code>results</code> directory.</p>

    <h2 id="contributing">Contributing</h2>
    <p>Contributions to the DeepBack project are welcome! If you find a bug, have a feature request, or want to contribute enhancements, please submit issues and pull requests through GitHub. Review the <a href="https://github.com/raminsalmas/DeepBack/blob/main/CONTRIBUTING.md">contribution guidelines</a> before getting started.</p>

    <h2 id="license">License</h2>
    <p>This project is licensed under the <a href="https://github.com/raminsalmas/DeepBack/blob/main/LICENSE">MIT License</a>.</p>

    <h2 id="contact">Contact</h2>
    <p>For any questions, suggestions, or collaborations, please feel free to contact the project maintainer, Ramin Salmas, at <a href="mailto:ramin.salmas@email.com">ramin.salmas@email.com</a>.</p>
</body>
</html>

