# UnsupervisedLearning

数据集：没有任何标签，都具有相同的标签或者没有标签  
如：聚类,鸡尾酒算法  
```Python
[W,s,v]=svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');
```  
应用：大型计算机集群组织，社交网络分析，市场细分，天文数据分析