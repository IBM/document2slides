## Run D2S model


Recommend to use miniconda environment.yml.
 Requirements:
-   python = 3.8
-   pytorch = 1.6.0
-   transformers = 3.0
-   faiss-gpu = 1.6


Cache IR results (filtered or prefiltered slides) and evaluate idf-recall
```console
python run.py ir (default)
python run.py ir -filter 1 -slide_json ../input/sciduet_slides_filter.json (default)
python run.py ir -filter 0 -slide_json ../input/sciduet_slides_prefilter.json (prefilter)
python run.py ir -eval 1 (evaluates idf-recall)
```
Train S2S model (default -num_gpus = 2 min recommended or one 32GB gpu)
```console
python run.py train -model_name mytest -ir_type [prefilter|filter] (need to specify -ir_type)
python run.py train -model_name mytest -ir_type [.|.|.] -num_gpus 1 (single gpu machine)
python run.py train -lr 2e-4 -max_epochs 5 -ir_type filter -model_name model_2e-4 (tunable learning rate and max epochs)
```
Test/validate S2S model (runs on prefilter)
```console
python run.py test
python run.py val -s2s_model [specify path to model]
python run.py test -s2s_model [] -file_name test_0 (saves generated slide to file if specified)
```
Compute rouge score for a saved test/val result file
```console
python run.py rouge -result_file ../results/test_0.json
```
For more details on command-line arguments
```console
python run.py -h
python run.py [MODE] -h
```
---
Code inspired by https://yjernite.github.io/lfqa.html
