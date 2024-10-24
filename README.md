# pandapower-heig-ui
### 1. In order to publish in pip increment version in setup.py
### 2. Remove old distribution file in dist folder 
### 3. Create new distribution file
```bash
python3 setup.py sdist bdist_wheel
```
### 3. Puplish on pip You schould have the pip token in your home direcotry

```bash
python3 -m twine upload dist/*
```

