---
pubDatetime: 2020-05-19
title: "开发并发布你的 Python 包"
draft: false
tags:
  - Python
  - Tutorial
description: "开发并发布你的 Python 包"
---


## How to update my code in pypi?
```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```
