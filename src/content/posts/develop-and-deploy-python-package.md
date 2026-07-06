---
pubDatetime: 2020-05-19
title: "Develop and release your python package"
draft: false
tags:
  - Python
  - Tutorial
description: "Develop and release your python package"
---


## How to update my code in pypi?
```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```
