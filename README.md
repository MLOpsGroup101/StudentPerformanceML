# stuperml


The goal of the project is to analyze how AI-assisted learning influences student outcomes and to develop a robust, reproducible machine learning system capable of predicting exam and overall academic scores.

In this project, PyTorch is used as the primary machine learning framework due to its flexibility, strong support for deep learning, and widespread adoption in both research and production environments. PyTorch will be integrated into the project using a modular structure, where data handling, model definition, training logic, and evaluation are separated into individual components. 

The project uses a publicly available [dataset](https://www.kaggle.com/datasets/ankushnarwade/ai-impact-on-student-performance?resource=download&fbclid=IwY2xjawPNtDZleHRuA2FlbQIxMABzcnRjBmFwcF9pZBAyMjIwMzkxNzg4MjAwODkyAAEe3tk6GmCGyheBcjJQi6rQhsBAd3wHoEWj8EJN7i0fimRVXtH-PFTiWG6jv4w_aem_77mTsWIQ9rs4H_at6RH8Pg) containing 8,000 student records with 26 features, covering demographics, study habits, academic performance indicators, and detailed AI usage metrics. Key features include age, grade level, study hours, attendance, prior exam scores, and several AI-related attributes such as AI usage time, dependency score, ethical usage score, and percentage of AI-generated content. The target variable is the final score.

Before training, the dataset will undergo an initial preprocessing phase. Numerical features such as study hours, attendance percentage, and AI usage metrics will be normalized to ensure stable and efficient model training. Categorical variables including gender, grade level, and AI usage purpose will be encoded into numerical representations suitable for neural networks. Missing values, if present, will be handled using appropriate strategies. The dataset will be split into training, validation, and test sets to enable reliable evaluation and to reduce the risk of overfitting.

A deep neural network is developed as the model to predict student performance based on the provided features from the dataset. The model is trained to predict the final score of the student and then evaluated using MSE (Mean Squared Error) in comparison to a baseline model. The baseline will simply be the mean of the final scores in the training dataset.

A core focus of this project is the application of MLOps best practices. The codebase is structured to support modular development, readability, and scalability. Docker is used to containerize the application, ensuring consistency across development, testing, and deployment environments. This setup enhances reproducibility, collaboration, and maintainability throughout the project lifecycle.

Overall, this project demonstrates how modern MLOps techniques can be combined with deep learning to build production ready machine learning systems.

Notice the project is intended for MacOS systems.

For documentation, see [GitHub pages](https://mlopsgroup101.github.io/StudentPerformanceML/)

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


Run Guide:

To retrieve the dataset without syncing dependencies
```bash
uv run src/stuperml/data.py
```

To launch API in terminal:
```bash
uv run uvicorn src.stuperml.api:app --reload
```
