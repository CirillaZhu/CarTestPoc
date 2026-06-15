
公网链接：https://www.modelscope.cn/studios/Cirilia/car_test_poc/summary

### 学习内容

-  跑通智能问答助手代码搭建，项目部署的最小流程
	- MinerU 进行 pdf 版面分析 + OCR识别，输出带类型的 json 块，并抽出图片、关联标题
	- 构造 rag：通过Python代码对 json 数据进行粗筛，按标题切块（chunks）；使用本地模型对数据进行向量化处理，生成 index 库（embedding），向量与切片数据一一对应
	- rag 检索：实现通过 query 数据进行向量化，在 index 库中计算向量相似度，召回 top-k 个相关切片，并带出切片相关的图表
	- 生成回答：接入 Deepseek Api 作为问答模型，先本地检索在 rag 中拿到相关信息，再将信息，提示词与问题一并作为输入，生成回答
	- 部署：本地部署测试成功后，在 Modelscope 线上部署，实现远程访问
