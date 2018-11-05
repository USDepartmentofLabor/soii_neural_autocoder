# Neural Network Autocoder for SOII

Contains the code used to train a close relative of the neural network autocoder
described in [Deep neural networks for worker injury autocoding](https://www.bls.gov/iif/deep-neural-networks.pdf).
Compared to the model described in the paper, `examples/big_single_seq_180_lr4e-4.py`
is significantly faster with similar performance, made possible by more efficient 
batching due to the concatenation of all text inputs.

# Requirements
Note: more recent versions of these libraries will likely also work.
* Anaconda (with Python 3.6)
* Tensorflow 1.8
* Keras 2.1.6
* NLTK 3.2.5
* Sklearn

# Training Data
The model expects a CSV training set with the following columns and data:

| column header | description | example |
|---------------|-------------|---------|
|survey_year|The year of the incident|2017|
|occupation_text| The worker's job title   | RN |
|other_text|Optional field indicating the worker's job category | elder care|
|company_name|The primary name of the worker's establishment | ACME hospitals inc. |
|secondary_name| The secondary name of the worker's establishment | ACME holding corp.|
|unit_description|Description of the sampled establishment| hospital staff only|
|nar_activity|A narrative answering "What was the worker doing before the incident occurred"| Helping patient get out of bed |
|nar_event|A narrative answering "What happened"| The patient slipped and employee tried to catch her|
|nar_nature|A narrative answering "What was the injury or illness?"|Employee strained lower back|
|nar_source|A narrative answering "What obect or substance directly harmed the employee?"|Patient and floor|
|naics|The 6 digit 2012 North American Industry Classification System (NAICS) code for the establishment| 622110|
|soc|The 6 digit 2010 Standard Occupational Classification (SOC) code for the worker | 29-1141|
|nature_code|The 2.01 [OIICS](https://www.bls.gov/iif/oshoiics.htm) nature code| 1233 |
|part_code|The 2.01 [OIICS](https://www.bls.gov/iif/oshoiics.htm) part code| 322 |
|event_code|The 2.01 [OIICS](https://www.bls.gov/iif/oshoiics.htm) event code| 7143 |
|source_code|The 2.01 [OIICS](https://www.bls.gov/iif/oshoiics.htm) source code|574|
|sec_source_code|The 2.01 [OIICS](https://www.bls.gov/iif/oshoiics.htm) secondary source code (blank means none present)||
|office|Checkbox indicating the job category of the worker (X or blank)|X|
|sales|Checkbox indicating the job category of the worker (X or blank)|X|
|assembly|Checkbox indicating the job category of the worker (X or blank)|X|
|repair|Checkbox indicating the job category of the worker (X or blank)|X|
|construction|Checkbox indicating the job category of the worker (X or blank)|X|
|health|Checkbox indicating the job category of the worker (X or blank)|X|
|driving|Checkbox indicating the job category of the worker (X or blank)|X|
|food|Checkbox indicating the job category of the worker (X or blank)|X|
|maintenance|Checkbox indicating the job category of the worker (X or blank)|X|
|material_handling|Checkbox indicating the job category of the worker (X or blank)|X|
|farming|Checkbox indicating the job category of the worker (X or blank)|X|
|other|Checkbox indicating the job category of the worker (X or blank)|X|

# Installation 
`pip install git+https://github.com/USDepartmentofLabor/neural_autocoder.git`

# Example Usage
Modify the `data_file` variable in `examples/big_single_seq_180_lr4e-4.py` to point
to an appropriately formatted training dataset, or leave as is to use the dummy
training set included in the module. Then run `big_single_seq_180_lr4e-4.py`.
Model checkpoints and a log of training results will be saved in the same directory.
