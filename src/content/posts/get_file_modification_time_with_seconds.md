---
pubDatetime: 2023-01-26
title: "如何获取文件的创建或修改时间（精确到秒）"
draft: false
tags:
  - Python
description: "需要获取文件的创建或修改时间（精确到秒）。"
---


I want to get the fiel creation or docification time **with seconds!!!**

In windows, it seems impossible to see the exect time in Windows explorer, because it only shows the date and hour and minutes, missing the seconds!

Luckily, we have Python!

```
modified = os.path.getmtime(file)
print("Date modified: "+time.ctime(modified))
print("Date modified:",datetime.datetime.fromtimestamp(modified))

# out: 
# Date modified: Tue Apr 21 11:50:46 2015
# Date modified: 2015-04-21 11:50:46


```

## Sort files from Glob by time
`files.sort(key=os.path.getctime)`
from [here](https://stackoverflow.com/questions/168409/how-do-you-get-a-directory-listing-sorted-by-creation-date-in-python).
