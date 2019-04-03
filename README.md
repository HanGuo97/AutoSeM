# AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning
Han Guo, Ramakanth Pasunuru, and Mohit Bansal. NAACL 2019

### Dependencies
* The project originally runs in Tensorflow 1.8, but should be compatible for future versions (except TF 2.0).
* Python 3.5
* See `requirements.txt`

### Setup
Download the data from [GLUE](https://github.com/nyu-mll/GLUE-baselines), and follow the pre-processing from authors. A copy of the download script is provided in this repo.
`python download_glue_data.py --data_dir glue_data --tasks all `

To compute the ELMo representations, use either [TF-Hub](https://www.tensorflow.org/hub/) or [AllenNLP](https://allennlp.org).

#### Instructions for TF-hub:
`elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
embeddings = elmo(..., as_dict=True)["elmo"]`

#### Instructions for AllenNLP:
AllenNLP's website includes a very detailed tutorial.

### Training Models
#### To run the stage-1 of the model, use the following script.

`python run_MTL.py --logdir [logdir] --tasks [tasks] --embedding_dim [embedding_dim] --num_units [num_units] --num_layers [num_layers] --dropout_rate [dropout_rate] --learning_rate [learning_rate] --stage [stage]`

#### To run the stage-2 of the model, use the following script.

`python run_MTL.py --logdir [logdir] --tasks [tasks] --embedding_dim [embedding_dim] --num_units [num_units] --num_layers [num_layers] --dropout_rate [dropout_rate] --learning_rate [learning_rate] --stage [stage]`


### Pre-trained Models
Models coming soon.

# Citation
```
@inproceedings{guo2019autosem,
  title={AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning},
  author={Han Guo and Ramakanth Pasunuru and Mohit Bansal},
  booktitle={Proc. of NAACL},
  year={2019}
}
```
