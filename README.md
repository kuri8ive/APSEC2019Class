## "Class Name Recommendation based on Graph Embedding of Program Elements"

- Here is the repository for [APSEC'19 paper](https://kuri8ive.github.io/preprints/apsec2019.pdf)
- The source code preprocessing the dataset is at `RelationExtractor/src/main/java`
- The source code of the proposed approach is at `core`
- All experiments in the paper are available in `exp.ipynb`

### Requirements

- `openjdk == 11.0.1`
- `python == 3.6.8`
- `pipenv version 2018.11.26`
- `maven == 3.6.2`
- `CUDA == 8.0.61`

### Procedure

1. clone this repository
1. `cd APSEC2019Class`
1. `mkdir data | mkdir data/input data/model data/output data/raw_data`
1. download the [data](http://groups.inf.ed.ac.uk/cup/naturalize/) and put it in `data/raw_data`
1. `cd RelationExtractor | mvn clean install`
1. `cd .. | java -cp RelationExtractor/target/RelationExtractor-1.0-SNAPSHOT-jar-with-dependencies.jar Main`
1. `pipenv install`
1. `pipenv run jupyter notebook`

### BibTeX

```
@inproceedings{kurimoto2019class,
    title={Class Name Recommendation based on Graph Embedding of Program Elements},
    author={Kurimoto, Shintaro and Hayase, Yasuhiro and Yonai, Hiroshi and Ito, Hiroyoshi and Kitagawa, Hiroyuki},
    booktitle={2019 26th Asia-Pacific Software Engineering Conference (APSEC)},
    year={2019}
}
```
