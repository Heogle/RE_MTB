# RE_MTB
[Matching the Blanks: Distributional Similarity for Relation Learning](https://www.aclweb.org/anthology/P19-1279/ "mtblink") 논문을 기반으로 개발하였다.   
Data는 개별적으로 받아야하며 Relation Extraction Benchmars인 TACRED를 사용하였습니다.  
구조는 아래 그림과 같다.   



<img src="https://user-images.githubusercontent.com/31501328/103607001-400acd00-4f5b-11eb-9fe4-79b697e306fb.png" width="600px" height="300px"></img>


### 1. Data Preprocessing
TACRED 데이터를 활용하여 아래와 같이 전처리를 진행합니다.

<pre>
<code>
cd data
python tacred_data_utils.py
</code>
</pre>

### 2. Train
아래와 같이 train을 진행합니다.
<pre>
<code>
CUDA_VISIBLE_DEVICES=0 python main.py --model entity_markers --data tacred
</code>
</pre>

### 3. Evaluation
train된 모델을 기반으로 evaluation을 진행합니다.
<pre>
<code>
CUDA_VISIBLE_DEVICES=0 python main.py --model entity_markers --data tacred --evaluation /path/to/checkpoints
</code>
</pre>
