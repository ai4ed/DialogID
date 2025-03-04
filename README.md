# DialogID
This is an official implementation for "DialogID: A Dialogic Instruction Dataset for Improving Teaching Effectiveness in Online Environments".

## Dataset Overview

The dataset is split into training, validation, and test sets, which can be found in the `data` directory. Each file contains two columns:
- `text`: the text of the sentence
- `label`: an integer representing the category of the text

The mapping between label numbers and categories is as follows:

```json
{
  "0": "commending",
  "1": "guidance",
  "2": "summarization", 
  "3": "greeting",
  "4": "note-taking",
  "5": "repeating",
  "6": "reviewing",
  "7": "example-giving",
  "8": "others"
}
```

### Data Statistics


The DialogID dataset is constructed from **K-12 online classes** at **TAL Education Group**. Through a rigorous **3-step annotation procedure**, we curated a total of **51,908 annotated samples**, of which **30,431** are identified as effective online dialogic instructions. This dataset provides a rich resource for studying dialogic instruction patterns in educational settings.

The detailed statistics of the DialogID dataset are illustrated in the figure below:

![Dataset Statistics](https://github.com/user-attachments/assets/25531997-1de6-4a4a-a793-06343d7c5be8)


## How to training?

### Install env

1、create conda env

```sh
conda create --name=atc_gpu python=3.7.5
source activate atc_gpu
```

2、install packages

```
conda install tensorflow-gpu==1.13.1  cudatoolkit=10.0.130=0
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements_gpu.txt
```

3、download pre-train models

Because the pre-train models are huge to host on github. All the pre-train models in this paper can download from [here](https://github.com/ymcui/Chinese-BERT-wwm). After download please push the model on `auto_text_classifier/atc/data`. The model path config can see in `src/auto_text_classifier/atc/configs/hf_config.py`


### training & predict

1、baselines

`nohup python -u atc_train.py ->baseline.log & `


2、RoBERTa+AT

`nohup python -u atc_adt_train.py ->baseline_adt.log & `

After training, the results of each model in the test set will be saved to a csv.

### Contact

If you have any quesiton, please don't hesitate to create an issue.



