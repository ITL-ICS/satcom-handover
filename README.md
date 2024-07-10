고마워 ㅎㅎ
선우야!!❤ 빠이팅!!( •̀ ω •́ )✧


🧡DQN 실행해서 학습하고 Reward 값 얻기를 원한다면
→ Past(Get reward graph)

🧡각종 plot을 뽑고 싶다면
→ Now(Various plots)

# dqn-leo-handover-python
DQN based Handover Optimization for LEO Satellites in NTN

**Steps**

Create a custom conda environment

```
conda create -n ntn_handover_drl
conda activate ntn_handover_drl
```

Install the required python libraries
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
conda install -c conda-forge pillow=9.3.0 matplotlib=3.6.2 pyorbital=1.7.3 scipy=1.9.3 requests=2.28.1 pyproj=3.4.1
```

Run the main script
```
python main_leo_handover.py
```
