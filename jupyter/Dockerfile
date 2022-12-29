# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
FROM jupyter/datascience-notebook:latest

# add libraries as needed here
# https://github.com/mementum/backtrader
RUN pip3 install --no-cache shap scikit-learn backtrader

# disable a required password to login
RUN mkdir -p ~/.jupyter/ && printf 'c.NotebookApp.token = "" ' >> ~/.jupyter/jupyter_notebook_config.py
