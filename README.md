# Visitor_distribution_prediction
Code used for the paper "Prediction of the distribution of Visitors for Large Events in Smart Cities"<br />
Folder "preprocessing": Contains the main preprocessing steps that were applied on basmati dataset in order to render it easy to process from machine learning algorithms. "papework2.py" is the main python script that creates a variety of datasets of different properties of our choice (such as number off clusters to split the area and timestep length)<br />
"Folder regression_classification": Contains the main two files implementing single output regression and classification. The subfolder datasets contains various datasets of different properties constructed from the preprocessing step. (e.g 30_5 means a thirty minute final timestep constructed from the mean values of its 6 contained 5-minute periods). Code can be run  directly from an open terminal inside this folder, using a command of this type:<br />
python single_output_classifiers.py  -d datasets/15_15min -s 1<br />
where:<br />
-d DIR, --dir DIR<br />
-s SHIFTED, --shifted SHIFTED<br />
SHIFTED is an integer value k and is translated into a shift of the weather and popularity independent variables k timesteps towards the future in order to perceive them as forecasts and not current states.<br />
Depending on the dataset used a uniquely and accordingly named evaluation csv file is created in the respective subfolder of the dataset. 
