# kaggle竞赛流程简介——以手写识别为例

## 下载数据集
[https://www.kaggle.com/c/digit-recognizer/data](https://www.kaggle.com/c/digit-recognizer/data)
数据包含三个csv文档。训练集 train.csv、测试集 test.csv、结果提交模版 sample_submission.csv。

## 数据说明
数据文件train.csv和test.csv包含的手绘位灰度图像，从0到9。每一行代表一个数字，不同的是train.csv中包含标签列，test.csv不包含标签列。

图片的像素大小为 28 x 28 ，每个像素具有与它相关联的单个像素值，表示像素的亮度或暗度，数字越大意味着较暗。这个像素值是0到255（含）之间的整数。也就是说 0 为白色，255 为黑色。

## 建模过程

- **二值化**
	由于训练集的像素值包含了0-255之间不同的值，因此我们需要对每个图片的特征进行二值化。二值化的处理方式是：设置一个阈值，大于该阈值的设置成255，小于该阈值的设置成0。
	```python
	import pandas as pd
	from sklearn import metrics
	from sklearn.model_selection import train_test_split
	from sklearn.neighbors import KNeighborsClassifier
		
	# load train data
	train = pd.read_csv('F:/GitHub/kaggle/Digit Recognizer/train.csv')
		
	# split feature and label
	feature = train.drop('label', axis=1)
	label = train[['label']]
	
	# 设置一个阈值，为0-255之间，大于该值的设置为 255， 小于该值的设置为0
	threshold = 100
	feature[feature >= threshold] = 255
	feature[feature < threshold] = 0
	```

- **分离训练集**
	从训练集中提取一部分做训练，另一部分做验证。
	``` python
	# split train and test data
	train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.2, random_state=0)
	```

- **拟合模型**
	``` python
	# fit classifier model
	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(train_feature, train_label)

	# predict test_feature
	predict_test = clf.predict(test_feature)

	# print result
	print(metrics.classification_report(test_label, predict_test))
	```

- **查看打印的预测结果评估**
	![](https://i.imgur.com/OQNwpzq.png)
	可以看到，该模型的预测成功率为96%，这个准确率在这个项目中算低的，但本次旨在学习如何参加kaggle项目，以及上传预测结果，因此对准确率不做高要求。

- **预测测试数据，并按提交要求保存结果。**
	``` python
	# predict test data,
	test = pd.read_csv('F:/GitHub/kaggle/Digit Recognizer/test.csv')
	
	# 设置一个阈值，为0-255之间，大于该值的设置为 255， 小于该值的设置为0
	threshold = 100
	test[test >= threshold] = 255
	test[test < threshold] = 0
	
	# you need save file as .csv like sample_submission.csv
	predict = clf.predict(test)
	predict = pd.DataFrame(predict)
	predict.to_csv('predict.csv')
	# 注意：这里保存的结果需要把第一行改成模版中的两个字段 ImageId, label, 而且ImageId是从1开始的，不是从0开始。
	```

## 最后在网页界面把刚刚保存的predict.csv文件上传到提交结果的地方即可。
结果上传之后，就会根据你的结果，给出评分，下拉即可看到你在所有人中的排名。
![](https://i.imgur.com/syLdAXT.png)
![](https://i.imgur.com/592Ydcc.png)
大家可以看到，我这个比较水，都排到1716名去了。前十名的预测率已经是100%了。

关于kaggle竞赛流程，大概就是这个样子，对于想提高正确率的，可以在调整模型参数或使用其他算法之后，从新预测结果，再上传就可以了。