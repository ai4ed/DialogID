# DialogID
This is an official implementation for "DialogID: A Dialogic Instruction Dataset for Improving Teaching Effectiveness in Online Environments".

## Dataset
The splited training, validation, test sets can be download from [here](), and put the downloaded files to `data` directory. Each file contains two columns:
- `text`: the context.
- `label`: the int number of text, the mapping relation of number to category is following：

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



