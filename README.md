# pandapower-heig-ui
In order to publish in pip increment version in setup.py

```bash
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
```

You schould have the pip token in your home direcotry