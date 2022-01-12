# Masked-EDA: The Effect of Using Masked Language Models in Random Textual Data Augmentation

Code for our paper accepted at CSICC 2021.

```
@inproceedings{rashid2021effect,
  title={The Effect of Using Masked Language Models in Random Textual Data Augmentation},
  author={Rashid, Mohammad Amin and Amirkhani, Hossein},
  booktitle={2021 26th International Computer Conference, Computer Society of Iran (CSICC)},
  pages={1--5},
  year={2021},
  organization={IEEE}
}
```

The original Easy Data Augmentation (EDA) code is shared here:
https://github.com/jasonwei20/eda_nlp

# Masked-EDA Usage
This code can be run on any text classification dataset (Check the example file 'data/tr_2000.tsv' for data structure).

First, install HuggingFace transformers:
<code>pip install transformer</code>

Second, setup a machine learning library of your choice (TensorFlow or PyTorch)
<code>pip install pytorch</code>
or
<code>pip install tensorflow</code>

# Run
All configurations are similar to original EDA (https://github.com/jasonwei20/eda_nlp) except for the type of Masked Language Model you wish to use.

* Note that the mask_model parameter must be one of the listed names. If no model name is provided, it would default to DistilBERT.

Available models: {'bert', 'roberta', 'distilbert'}

<code>python ./augment.py --input='data/tr_2000.tsv' --output='data/tr2000_aug16.tsv' --mask_model='roberta' --num_aug=16 --alpha_sr=0.1 --alpha_rd=0.1 --alpha_ri=0.4 --alpha_rs=0.1</code>
