# KAGGLE

Productionized solutions to kaggle problems.

![Productionized Kaggle Solutions](https://media-exp1.licdn.com/dms/image/C5612AQGmGpr38YdJLQ/article-cover_image-shrink_600_2000/0?e=1588204800&v=beta&t=f_881YG8S12TbuMdEHuXpuxAyCLrtaVJxgLxY06aImU)

## PROJECTS

| __Problem__ | __Description__ | __Placement__ | __Solution__ |
| :---: | --- | --- | --- |
| ![wids-datathon-2020](https://github.com/iainwo/kaggle/workflows/wids-datathon-2020/badge.svg?branch=master) | The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. _[Git](./wids-datathon-2020/)._ | 160th / 951 | `Batch Prediction` |

## HOW-TO CREATE NEW PROJECTS

New Projects are created using the Cookiecutter template `./kaggle-data-science/`.
This is a modified version of [Cookiecutter](https://github.com/drivendata/cookiecutter-data-science).
Alterations have been made for Kaggle specific tasks.

```sh
$ pip install cookiecutter
$ cookiecutter kaggle-data-science # run-through the setup
$ cd ./$MY_NEW_PROJ_NAME
$ make create_environment # create project conda env
$ make requirements # install python libs
```
