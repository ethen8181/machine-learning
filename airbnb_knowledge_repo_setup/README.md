Some tooling for converting the repo into a deployable Airbnb Knowledge Repo.
First install the [Airbnb Knowledge Repo](https://github.com/airbnb/knowledge-repo).
```bash
pip install --upgrade knowledge-repo
```

Then, convert the directory to Knowledge Repo format by calling the `knowledge_repo_converter.py` with the first argument as the root directory of machine-learning, and the second argument as the desired path to the knowledge repo.

```bash
python knowledge_repo_converter.py /Documents/machine-learning /Documents/machine-learning/knowledge-repo
```

The knowledge repo will automatically be created, after which it can be previewed with:

```bash
knowledge_repo --repo /Documents/machine-learning/knowledge-repo --port 8888
```
