# good_codes
 feature_gan.py #这个是一个简单的生成对抗网络，其实是一个特征迁移网络，是将market1501在网络提取后的特征，迁移到duke经过网络提取特征后的空间里，寻思能对跨域REID有帮助，但是实验结果并不好。  
 triplet_sperate.py #这个代码实现了如何让REID的market1501数据库，按照triplet loss的形式，形成anchor positive negative 对，形成N×K的mini batch送入网络，N代表行人ID的数量，K代表每个ID具有的图片数量 对这个代码的博客讲解地址为：  https://blog.csdn.net/t20134297/article/details/105712627
