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
