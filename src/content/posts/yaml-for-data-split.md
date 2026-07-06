---
pubDatetime: 2022-12-19
title: "使用 YAML 进行数据划分"
draft: false
tags:
  - Knowledge
description: "通常使用固定随机种子划分数据集，但随着项目开发，数据集会逐渐更新。"
---


## Why do we need YAML for data split?

Normally I used a fixed random seed to split dataset. However, I found that my dataset has been slightly and gradually updated with the development of the project. 

For instance, in the latter experiments, I found that some patients should be excluded. Then should I re-train all the previous expreiments again? if not, how to ensure the following experiments use the same training/validation/testing data with the previous experiment? (The same seed for different lenth of patient list will lead to very different data split)

So let's use a YAML file to split dataset so that we can always have the almost same split for training/validation/testing data.

A complete YAML tutorial could be found at [Real Python](https://realpython.com/python-yaml/)

Difference between YAML, JSON and XML is [here](https://realpython.com/python-yaml/#comparison-with-xml-and-json)
